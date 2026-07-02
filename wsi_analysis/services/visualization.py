import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Evitar problemas de hilos con la GUI de Matplotlib
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "Negative for intraepithelial lesion": "#2ca02c",  # verde
    "ASC-US": "#00d2ff",                              # celeste / cian
    "LSIL": "#ff7f0e",                                # naranja
    "ASC-H": "#d62728",                               # rojo
    "HSIL": "#9467bd",                                # morado
    "SCC": "#000000",                                 # negro
}

ABNORMAL_CLASSES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]

def save_horizontal_figure(fig, filepath):
    """
    Guarda figuras con orientación horizontal, reduciendo márgenes,
    ajustando leyendas y asegurando títulos claros de forma académica.
    """
    fig.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close(fig)

def generate_thumbnail_wsi_plot(thumb_rgb, filepath):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(thumb_rgb)
    ax.set_title("Thumbnail de Portaobjeto WSI completo", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_mask_foreground_plot(thumb_rgb, fg_mask, filepath):
    overlay = thumb_rgb.copy()
    overlay[fg_mask > 0] = (0.7 * overlay[fg_mask > 0] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(overlay)
    ax.set_title(f"Máscara de Muestra Útil (Foreground - Verde) (~{(fg_mask > 0).mean()*100:.1f}%)", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_mask_artifacts_plot(thumb_rgb, artifact_mask, filepath):
    overlay = thumb_rgb.copy()
    overlay[artifact_mask > 0] = (0.7 * overlay[artifact_mask > 0] + 0.3 * np.array([255, 0, 0])).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(overlay)
    ax.set_title(f"Máscara de Artefactos Grandes Detectados (Rojo) (~{(artifact_mask > 0).mean()*100:.1f}%)", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_mask_clean_plot(thumb_rgb, clean_mask, filepath):
    overlay = thumb_rgb.copy()
    overlay[clean_mask > 0] = (0.7 * overlay[clean_mask > 0] + 0.3 * np.array([0, 0, 255])).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(overlay)
    ax.set_title(f"Máscara Limpia de Regiones Analizables (Azul) (~{(clean_mask > 0).mean()*100:.1f}%)", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_overlay_rois_plot(thumb_rgb, df_rois, filepath):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(thumb_rgb)
    
    for _, r in df_rois.iterrows():
        x1 = r["x1_thumb"]
        y1 = r["y1_thumb"]
        w = r["x2_thumb"] - r["x1_thumb"]
        h = r["y2_thumb"] - r["y1_thumb"]
        rect = plt.Rectangle(
            (x1, y1), w, h, fill=False,
            edgecolor="cyan", linewidth=1.2, alpha=0.8
        )
        ax.add_patch(rect)
        
    ax.set_title(f"Portaobjeto Completo con Rejilla de ROIs Extraídas (N={len(df_rois)})", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_mapa_candidatos_y_clasificados_plot(
    thumb_rgb, df_candidates, df_predictions, df_rois, filepath
):
    """
    Puntos grises = todos los candidatos celulares detectados
    Puntos de colores = células clasificadas
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(thumb_rgb)

    # 1. Dibujar todos los candidatos detectados en gris
    if len(df_candidates) > 0:
        ax.scatter(
            df_candidates["x_thumb"],
            df_candidates["y_thumb"],
            s=15,
            c="gray",
            alpha=0.35,
            label=f"Candidato celular (n={len(df_candidates)})",
            edgecolors="none"
        )

    # 2. Dibujar células clasificadas por color de clase
    if len(df_predictions) > 0:
        for cls, color in CLASS_COLORS.items():
            sub = df_predictions[df_predictions["pred_label"] == cls]
            if len(sub) > 0:
                ax.scatter(
                    sub["x_thumb"],
                    sub["y_thumb"],
                    s=45,
                    c=color,
                    alpha=0.9,
                    label=f"{cls} (n={len(sub)})",
                    edgecolors="white",
                    linewidths=0.5
                )

    ax.set_title("Distribución Espacial de Candidatos Celulares y Clasificaciones Asistidas por IA", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    # Colocar la leyenda fuera a la derecha
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
    save_horizontal_figure(fig, filepath)

def generate_heatmap_densidad_candidatos_plot(
    thumb_rgb, df_candidates, filepath, bins=100, blur_ksize=35, alpha=0.45
):
    """
    Mapa de calor de densidad de todos los candidatos celulares detectados.
    """
    h, w = thumb_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(16, 9))
    
    if len(df_candidates) == 0:
        ax.imshow(thumb_rgb)
        ax.set_title("Mapa de Calor: Sin candidatos detectados", fontsize=16, pad=15, fontweight='bold')
        ax.axis("off")
        save_horizontal_figure(fig, filepath)
        return

    x = df_candidates["x_thumb"].clip(0, w - 1).values
    y = df_candidates["y_thumb"].clip(0, h - 1).values

    heatmap, _, _ = np.histogram2d(y, x, bins=[bins, bins], range=[[0, h], [0, w]])
    heatmap = heatmap.astype(np.float32)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)

    if heatmap.max() > 0:
        heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap_norm = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(thumb_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    ax.imshow(overlay)
    ax.set_title("Mapa de Calor de Densidad de Candidatos Celulares Detectados", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_heatmap_densidad_anormales_plot(
    thumb_rgb, df_predictions, filepath, bins=100, blur_ksize=35, alpha=0.45
):
    """
    Mapa de calor de densidad de candidatos celulares clasificados como anormales.
    """
    h, w = thumb_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(16, 9))
    
    df_abnormal = df_predictions[df_predictions["pred_label"].isin(ABNORMAL_CLASSES)]

    if len(df_abnormal) == 0:
        ax.imshow(thumb_rgb)
        ax.set_title("Mapa de Calor: Sin células anormales detectadas", fontsize=16, pad=15, fontweight='bold')
        ax.axis("off")
        save_horizontal_figure(fig, filepath)
        return

    x = df_abnormal["x_thumb"].clip(0, w - 1).values
    y = df_abnormal["y_thumb"].clip(0, h - 1).values

    heatmap, _, _ = np.histogram2d(y, x, bins=[bins, bins], range=[[0, h], [0, w]])
    heatmap = heatmap.astype(np.float32)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)

    if heatmap.max() > 0:
        heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap_norm = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(thumb_rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    ax.imshow(overlay)
    ax.set_title("Mapa de Calor de Densidad de Células Anormales (ASC-US, LSIL, ASC-H, HSIL, SCC)", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_overlay_rois_prioritarios_plot(
    thumb_rgb, df_rois, roi_summary, filepath, top_n=10
):
    """
    Dibuja los ROIs con mayor prioridad en rojo con etiquetas de conteo de anormales.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(thumb_rgb)

    if len(roi_summary) == 0:
        ax.set_title("ROIs Prioritarios: Sin datos", fontsize=16, pad=15, fontweight='bold')
        ax.axis("off")
        save_horizontal_figure(fig, filepath)
        return

    roi_plot = df_rois.merge(roi_summary, on="roi_id", how="left")
    roi_plot["n_abnormal_cells"] = roi_plot["n_abnormal_cells"].fillna(0)
    roi_plot["max_priority"] = roi_plot["max_priority"].fillna(0)

    # Seleccionar top_n de mayor relevancia clínica
    selected = roi_plot.sort_values(
        ["max_priority", "n_abnormal_cells", "max_confidence"],
        ascending=[False, False, False]
    ).head(top_n)

    # Dibujar todos los ROIs en gris suave
    for _, r in roi_plot.iterrows():
        x1 = r["x1_thumb"]
        y1 = r["y1_thumb"]
        w = r["x2_thumb"] - r["x1_thumb"]
        h = r["y2_thumb"] - r["y1_thumb"]
        rect = plt.Rectangle(
            (x1, y1), w, h, fill=False,
            edgecolor="white", linewidth=0.5, alpha=0.4
        )
        ax.add_patch(rect)

    # Dibujar prioritarios en rojo grueso
    for _, r in selected.iterrows():
        x1 = r["x1_thumb"]
        y1 = r["y1_thumb"]
        w = r["x2_thumb"] - r["x1_thumb"]
        h = r["y2_thumb"] - r["y1_thumb"]
        rect = plt.Rectangle(
            (x1, y1), w, h, fill=False,
            edgecolor="red", linewidth=2.0, alpha=0.9
        )
        ax.add_patch(rect)

        label = f"{r['roi_id']}\nAnorm={int(r['n_abnormal_cells'])}"
        ax.text(
            x1 + 3, y1 + 15, label,
            color="red", fontsize=8, fontweight='bold',
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2")
        )

    ax.set_title(f"ROIs Prioritarios para Revisión Citológica Asistida — Top {top_n}", fontsize=16, pad=15, fontweight='bold')
    ax.axis("off")
    save_horizontal_figure(fig, filepath)

def generate_barplot_resumen_por_clase_plot(summary_by_class, filepath):
    fig, ax = plt.subplots(figsize=(16, 9))

    if len(summary_by_class) == 0:
        ax.text(0.5, 0.5, "Sin datos de predicciones", ha='center', va='center')
        ax.set_title("Distribución de Células por Clase", fontsize=16, pad=15, fontweight='bold')
        save_horizontal_figure(fig, filepath)
        return

    df = summary_by_class.sort_values("n_cells", ascending=False).copy()
    colors = [CLASS_COLORS.get(c, "#7f7f7f") for c in df["pred_label"]]

    bars = ax.bar(df["pred_label"], df["n_cells"], color=colors, edgecolor='grey', linewidth=0.7)
    ax.set_ylabel("Cantidad de células", fontsize=12, labelpad=10)
    ax.set_xlabel("Clase Celular (IA)", fontsize=12, labelpad=10)
    ax.set_title("Distribución Cuantitativa de Células Clasificadas por Clase", fontsize=16, pad=15, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    save_horizontal_figure(fig, filepath)

def generate_barplot_retencion_celular_plot(df_retention, filepath):
    """
    Dibuja un gráfico de barras de retención celular mostrando el embudo de calidad.
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    if len(df_retention) == 0:
        ax.text(0.5, 0.5, "Sin datos de retención celular", ha='center', va='center')
        ax.set_title("Filtro y Retención Celular", fontsize=16, pad=15, fontweight='bold')
        save_horizontal_figure(fig, filepath)
        return

    criterios = df_retention["criterio"].values
    conteos = df_retention["n"].values
    
    # Crear gradiente de color azul-morado para el embudo
    colors = plt.cm.plasma(np.linspace(0.8, 0.2, len(criterios)))

    bars = ax.barh(criterios[::-1], conteos[::-1], color=colors[::-1], edgecolor='grey', linewidth=0.7)
    ax.set_xlabel("Cantidad de Candidatos", fontsize=12, labelpad=10)
    ax.set_title("Embudo de Retención y Filtros de Calidad Celular", fontsize=16, pad=15, fontweight='bold')
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{int(width)}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # offset horizontal
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    save_horizontal_figure(fig, filepath)
