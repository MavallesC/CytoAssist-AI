import pandas as pd
import numpy as np

# Configuración visual de clases y prioridades clínicas
CLASS_COLORS = {
    "Negative for intraepithelial lesion": "#2ca02c",  # verde
    "ASC-US": "#3a86c8",                              # azul / otro color
    "LSIL": "#ff7f0e",                                # naranja
    "ASC-H": "#d62728",                               # rojo
    "HSIL": "#9467bd",                                # morado
    "SCC": "#000000",                                 # negro
}

CLINICAL_PRIORITY = {
    "Negative for intraepithelial lesion": 0,
    "ASC-US": 1,
    "LSIL": 2,
    "ASC-H": 3,
    "HSIL": 4,
    "SCC": 5,
}

ABNORMAL_CLASSES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]

def build_class_summary(df_pred):
    """
    Genera resumen cuantitativo por clase predicha.
    """
    if len(df_pred) == 0:
        return pd.DataFrame()

    total = len(df_pred)

    summary = (
        df_pred
        .groupby("pred_label")
        .agg(
            n_cells=("pred_label", "size"),
            mean_confidence=("confidence", "mean"),
            std_confidence=("confidence", "std"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        )
        .reset_index()
    )

    summary["percentage"] = summary["n_cells"] / total * 100
    summary["clinical_priority"] = summary["pred_label"].map(CLINICAL_PRIORITY).fillna(-1).astype(int)

    summary = summary.sort_values(
        ["clinical_priority", "n_cells", "mean_confidence"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return summary

def build_roi_summary(df_pred):
    """
    Resume la presencia de células y su prioridad clínica por ROI.
    """
    if len(df_pred) == 0:
        return pd.DataFrame()

    df = df_pred.copy()
    df["is_abnormal"] = df["pred_label"].isin(ABNORMAL_CLASSES).astype(int)
    df["priority"] = df["pred_label"].map(CLINICAL_PRIORITY).fillna(-1).astype(int)

    roi_summary = (
        df
        .groupby("roi_id")
        .agg(
            n_predicted_cells=("pred_label", "size"),
            n_abnormal_cells=("is_abnormal", "sum"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            max_priority=("priority", "max"),
        )
        .reset_index()
    )

    # Identificar la clase dominante de cada ROI
    dominant_roi = (
        df.groupby(["roi_id", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["roi_id", "count"], ascending=[True, False])
        .drop_duplicates("roi_id")
        .rename(columns={"pred_label": "dominant_class_roi"})
    )

    roi_summary = roi_summary.merge(
        dominant_roi[["roi_id", "dominant_class_roi"]],
        on="roi_id",
        how="left"
    )

    roi_summary = roi_summary.sort_values(
        ["max_priority", "n_abnormal_cells", "max_confidence"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return roi_summary

def infer_preliminary_wsi_class(
    df_pred,
    min_cells_for_global_result=100,
    min_conf_priority=0.50,
    min_count_lsil=5,
    min_pct_lsil=10.0
):
    """
    Regla de clasificación preliminar asistida por IA.
    Retorna un diccionario detallando la categoría preliminar recomendada y el sustento clínico.
    """
    disclaimer = "Clasificación preliminar asistida por IA"

    if len(df_pred) == 0:
        return {
            "preliminary_wsi_class": "No concluyente",
            "report_type": "Sin celularidad evaluable",
            "reason": "No se detectaron células candidatas para clasificación.",
            "disclaimer": disclaimer,
            "dominant_cell_class": None,
            "dominant_cell_count": 0,
            "dominant_cell_pct": 0.0,
            "is_critical": False
        }

    total = len(df_pred)
    counts = df_pred["pred_label"].value_counts()
    dominant_class = counts.index[0]
    dominant_count = int(counts.iloc[0])
    dominant_pct = float(dominant_count / total * 100)

    # Filtrar predicciones de alta confianza
    df_high = df_pred[df_pred["confidence"] >= min_conf_priority].copy()

    # 1. Hallazgos prioritarios críticos (SCC, HSIL, ASC-H)
    high_priority = ["SCC", "HSIL", "ASC-H"]
    for cls in high_priority:
        sub = df_high[df_high["pred_label"] == cls]
        if len(sub) > 0:
            return {
                "preliminary_wsi_class": cls,
                "report_type": "Hallazgo prioritario crítico",
                "reason": (
                    f"Se detectó al menos una célula de alta prioridad clasificada como {cls} "
                    f"con confianza >= {min_conf_priority:.0%}. "
                    "Se requiere revisión experta prioritaria inmediata."
                ),
                "disclaimer": disclaimer,
                "dominant_cell_class": dominant_class,
                "dominant_cell_count": dominant_count,
                "dominant_cell_pct": dominant_pct,
                "n_total_predicted_cells": int(total),
                "n_high_conf_cells": int(len(df_high)),
                "is_critical": True
            }

    # 2. Baja celularidad
    if total < min_cells_for_global_result:
        abnormal_count = int(df_pred["pred_label"].isin(ABNORMAL_CLASSES).sum())
        abnormal_pct = float(abnormal_count / total * 100) if total > 0 else 0
        return {
            "preliminary_wsi_class": "No concluyente por baja celularidad clasificada",
            "report_type": "Baja celularidad clasificada",
            "reason": (
                f"Solo se clasificaron {total} células, por debajo del mínimo de {min_cells_for_global_result}. "
                f"Se detectaron {abnormal_count} células anormales ({abnormal_pct:.1f}%). "
                "El resultado debe interpretarse como soporte visual preliminar, no como diagnóstico."
            ),
            "disclaimer": disclaimer,
            "dominant_cell_class": dominant_class,
            "dominant_cell_count": dominant_count,
            "dominant_cell_pct": dominant_pct,
            "n_total_predicted_cells": int(total),
            "n_high_conf_cells": int(len(df_high)),
            "is_critical": False
        }

    # 3. Lesiones de bajo grado (LSIL o ASC-US)
    lsil_all = df_pred[df_pred["pred_label"] == "LSIL"]
    lsil_high = df_high[df_high["pred_label"] == "LSIL"]
    lsil_count = int(len(lsil_all))
    lsil_pct = float(lsil_count / total * 100)

    if len(lsil_high) > 0 and (lsil_count >= min_count_lsil or lsil_pct >= min_pct_lsil):
        return {
            "preliminary_wsi_class": "LSIL",
            "report_type": "Hallazgo preliminar de bajo grado",
            "reason": (
                f"Se detectaron {lsil_count} células LSIL ({lsil_pct:.1f}%), "
                f"con al menos una confirmada con alta confianza (>= {min_conf_priority:.0%}). "
                "Se recomienda revisión citopatológica."
            ),
            "disclaimer": disclaimer,
            "dominant_cell_class": dominant_class,
            "dominant_cell_count": dominant_count,
            "dominant_cell_pct": dominant_pct,
            "n_total_predicted_cells": int(total),
            "n_high_conf_cells": int(len(df_high)),
            "is_critical": False
        }

    # 4. Predominio negativo
    return {
        "preliminary_wsi_class": dominant_class,
        "report_type": "Predominio de células negativas",
        "reason": (
            "No se encontraron clases de alta prioridad que superen los umbrales de relevancia. "
            f"La muestra presenta un predominio de células negativas ({dominant_pct:.1f}%)."
        ),
        "disclaimer": disclaimer,
        "dominant_cell_class": dominant_class,
        "dominant_cell_count": dominant_count,
        "dominant_cell_pct": dominant_pct,
        "n_total_predicted_cells": int(total),
        "n_high_conf_cells": int(len(df_high)),
        "is_critical": False
    }
