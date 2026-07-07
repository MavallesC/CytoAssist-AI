import os
import json
import pandas as pd
from datetime import datetime

def generate_csv_reports(
    run_dir,
    df_candidates,
    df_predictions,
    summary_by_class,
    summary_by_roi,
    df_retention
):
    """
    Guarda todos los reportes tabulares en formato CSV.
    """
    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # 1. predictions_candidates.csv
    pred_path = os.path.join(predictions_dir, "predictions_candidates.csv")
    df_predictions.to_csv(pred_path, index=False)
    
    # 2. resumen_por_clase.csv
    class_path = os.path.join(predictions_dir, "resumen_por_clase.csv")
    summary_by_class.to_csv(class_path, index=False)
    
    # 3. resumen_por_roi.csv
    roi_path = os.path.join(predictions_dir, "resumen_por_roi.csv")
    summary_by_roi.to_csv(roi_path, index=False)
    
    # 4. resumen_retencion_celular.csv
    ret_path = os.path.join(predictions_dir, "resumen_retencion_celular.csv")
    df_retention.to_csv(ret_path, index=False)

    return {
        "predictions_candidates": pred_path,
        "resumen_por_clase": class_path,
        "resumen_por_roi": roi_path,
        "resumen_retencion_celular": ret_path
    }

def generate_html_report(
    run_dir,
    sample_name,
    run_id,
    preliminary_report,
    summary_by_class,
    summary_by_roi,
    df_retention,
    parameters
):
    """
    Genera el reporte final en formato HTML para impresión o descarga.
    Utiliza una maquetación CSS premium adaptada para uso médico.
    """
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    html_path = os.path.join(report_dir, "reporte_wsi.html")
    
    # Consultar slides y rois en base de datos
    try:
        from wsi_analysis.models import Slide, ROI
        total_slides = Slide.objects.filter(roi__run__run_id=run_id).count()
        total_rois = ROI.objects.filter(run__run_id=run_id).count()
    except Exception:
        total_slides = 0
        total_rois = len(summary_by_roi) if summary_by_roi is not None else 0

    # Formatear filas de tablas para inyectar en HTML
    class_rows = ""
    if len(summary_by_class) > 0:
        for _, row in summary_by_class.iterrows():
            class_rows += f"""
            <tr>
                <td><strong>{row['pred_label']}</strong></td>
                <td>{int(row['n_cells'])}</td>
                <td>{row['percentage']:.2f}%</td>
                <td>{row['mean_confidence']:.2%}</td>
            </tr>
            """
    else:
        class_rows = "<tr><td colspan='4'>No se registraron células predichas.</td></tr>"

    roi_rows = ""
    if len(summary_by_roi) > 0:
        from wsi_analysis.models import ROI
        # Mostrar top 10 ROIs prioritarias
        for _, row in summary_by_roi.head(10).iterrows():
            try:
                roi_obj = ROI.objects.filter(run__run_id=run_id, roi_id=row['roi_id']).first()
                n_slides = roi_obj.slides.count() if roi_obj else 0
            except Exception:
                n_slides = 0
            roi_rows += f"""
            <tr>
                <td><code>{row['roi_id']}</code></td>
                <td>{n_slides}</td>
                <td>{int(row['n_predicted_cells'])}</td>
                <td style="color: #d62728; font-weight: bold;">{int(row['n_abnormal_cells'])}</td>
                <td>{row['mean_confidence']:.2%}</td>
                <td>{int(row['max_priority'])}</td>
            </tr>
            """
    else:
        roi_rows = "<tr><td colspan='6'>No se encontraron ROIs procesadas.</td></tr>"

    retention_rows = ""
    if len(df_retention) > 0:
        for _, row in df_retention.iterrows():
            retention_rows += f"""
            <tr>
                <td>{row['criterio']}</td>
                <td>{int(row['n'])}</td>
                <td>{row['porcentaje_sobre_total_crops']:.1f}%</td>
            </tr>
            """
    else:
        retention_rows = "<tr><td colspan='3'>Sin datos.</td></tr>"

    # Determinar clase de alerta CSS según gravedad
    alert_class = "alert-warning"
    if preliminary_report.get("is_critical", False):
        alert_class = "alert-danger"
    elif "negativas" in preliminary_report.get("report_type", "").lower():
        alert_class = "alert-success"

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>CytoAssist AI - Reporte Analítico WSI</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
            line-height: 1.5;
            background-color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0056b3;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header-title h1 {{
            margin: 0;
            font-size: 28px;
            color: #0056b3;
        }}
        .header-title p {{
            margin: 5px 0 0 0;
            font-size: 14px;
            color: #64748b;
        }}
        .header-meta {{
            text-align: right;
            font-size: 14px;
            color: #475569;
        }}
        .section-title {{
            font-size: 20px;
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .alert {{
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 25px;
            border-left: 5px solid;
        }}
        .alert-success {{
            background-color: #f0fdf4;
            border-color: #16a34a;
            color: #15803d;
        }}
        .alert-warning {{
            background-color: #fffbeb;
            border-color: #d97706;
            color: #b45309;
        }}
        .alert-danger {{
            background-color: #fef2f2;
            border-color: #dc2626;
            color: #b91c1c;
        }}
        .alert h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 18px;
        }}
        .alert p {{
            margin: 0;
            font-size: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            font-size: 14px;
        }}
        th {{
            background-color: #f1f5f9;
            color: #334155;
            font-weight: 600;
        }}
        .disclaimer {{
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 15px;
            font-size: 12px;
            color: #64748b;
            border-radius: 4px;
            margin-top: 40px;
        }}
        .disclaimer strong {{
            color: #475569;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .figure-box {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-box img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
        }}
        @media print {{
            body {{
                background-color: #fff;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                border: none;
                padding: 0;
            }}
            .no-print {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <h1>CytoAssist AI</h1>
                <p>Sistema de Análisis Citopatológico Asistido</p>
            </div>
            <div class="header-meta">
                <strong>ID Muestra:</strong> {sample_name}<br>
                <strong>ID Corrida:</strong> {run_id}<br>
                <strong>Fecha:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M")}
            </div>
        </div>

        <div class="alert {alert_class}">
            <h3>{preliminary_report.get('disclaimer')}</h3>
            <p><strong>Clase Preliminar Estimada:</strong> {preliminary_report.get('preliminary_wsi_class')}</p>
            <p style="margin-top: 10px;"><strong>Sustento Clínico:</strong> {preliminary_report.get('reason')}</p>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px;">
            <div style="background-color: #f1f5f9; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #cbd5e1;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">ROIs Analizados</div>
                <div style="font-size: 24px; font-weight: bold; color: #0056b3;">{total_rois}</div>
            </div>
            <div style="background-color: #f1f5f9; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #cbd5e1;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Slides (Cuadrantes)</div>
                <div style="font-size: 24px; font-weight: bold; color: #0056b3;">{total_slides}</div>
            </div>
            <div style="background-color: #f1f5f9; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #cbd5e1;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Células Evaluadas</div>
                <div style="font-size: 24px; font-weight: bold; color: #0056b3;">{preliminary_report.get('n_total_predicted_cells', 0)}</div>
            </div>
        </div>

        <div class="grid">
            <div>
                <div class="section-title">Distribución por Clase IA</div>
                <table>
                    <thead>
                        <tr>
                            <th>Clase Celular</th>
                            <th>Conteo</th>
                            <th>Porcentaje</th>
                            <th>Confianza Prom.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {class_rows}
                    </tbody>
                </table>
            </div>
            <div>
                <div class="section-title">Parámetros del Pipeline</div>
                <table>
                    <thead>
                        <tr>
                            <th>Parámetro</th>
                            <th>Valor</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Tamaño de Ventana (scanning)</td><td>{parameters.get('tile_size')} px</td></tr>
                        <tr><td>Paso (stride)</td><td>{parameters.get('stride')} px</td></tr>
                        <tr><td>Ancho del Slide (cuadrante)</td><td>{parameters.get('slide_width', 1376)} px</td></tr>
                        <tr><td>Alto del Slide (cuadrante)</td><td>{parameters.get('slide_height', 1020)} px</td></tr>
                        <tr><td>Tamaño del Crop</td><td>{parameters.get('crop_size')} px</td></tr>
                        <tr><td>Fracción de Muestra Útil Mín.</td><td>{parameters.get('min_clean_frac')}</td></tr>
                        <tr><td>Score Mín. de Enfoque (Laplacian)</td><td>{parameters.get('min_focus_score')}</td></tr>
                        <tr><td>Densidad de Bordes Mín.</td><td>{parameters.get('min_edge_density')}</td></tr>
                        <tr><td>Área de Núcleo Permitida</td><td>{parameters.get('min_nucleus_area')} - {parameters.get('max_nucleus_area')} px</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section-title">Embudo de Calidad y Filtrado de Candidatos</div>
        <table>
            <thead>
                <tr>
                    <th>Criterio de Calidad</th>
                    <th>Candidatos Retenidos</th>
                    <th>% sobre total extraídos</th>
                </tr>
            </thead>
            <tbody>
                {retention_rows}
            </tbody>
        </table>

        <div class="section-title">Top 10 ROIs Prioritarios (Mayor Carga Lesional Anormal)</div>
        <table>
            <thead>
                <tr>
                    <th>ID ROI</th>
                    <th>Slides (Sub-recuadros)</th>
                    <th>Células Evaluadas</th>
                    <th>Células Anormales</th>
                    <th>Confianza Promedio</th>
                    <th>Prioridad Clínica Max</th>
                </tr>
            </thead>
            <tbody>
                {roi_rows}
            </tbody>
        </table>

        <div class="disclaimer">
            <strong>ADVERTENCIA Y DESCARGO DE RESPONSABILIDAD:</strong><br>
            Este reporte es una "Clasificación preliminar asistida por IA" orientada exclusivamente a servir de apoyo diagnóstico para el citopatólogo. 
            El software localiza candidatos sospechosos y propone etiquetas celulares, pero no emite un diagnóstico definitivo. 
            Este resultado bajo ninguna circunstancia reemplaza la evaluación clínica y firma experta del profesional de la salud.
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path

def sanitize_json_data(data):
    """
    Recursivamente convierte valores no permitidos en JSON estándar (como NaN o Infinity)
    en None (que se serializa como null en JSON).
    """
    import math
    import numpy as np
    if isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(v) for v in data]
    elif pd.isna(data):
        return None
    elif isinstance(data, float) or isinstance(data, np.floating):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, int) or isinstance(data, np.integer):
        return int(data)
    elif hasattr(data, "dtype"): # Tipos escalares de numpy
        try:
            val_py = data.item()
            if isinstance(val_py, float) and (math.isnan(val_py) or math.isinf(val_py)):
                return None
            return val_py
        except Exception:
            return data
    else:
        return data

def generate_json_report(
    run_dir,
    sample_name,
    run_id,
    preliminary_report,
    summary_by_class,
    summary_by_roi,
    df_retention,
    parameters
):
    """
    Genera el reporte estructurado en formato JSON.
    """
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "reporte_wsi.json")

    # Consultar slides y rois en base de datos para incluir en JSON
    try:
        from wsi_analysis.models import Slide, ROI
        total_slides = Slide.objects.filter(roi__run__run_id=run_id).count()
        total_rois = ROI.objects.filter(run__run_id=run_id).count()
    except Exception:
        total_slides = 0
        total_rois = len(summary_by_roi) if summary_by_roi is not None else 0

    report_data = {
        "sample_name": sample_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_rois": total_rois,
        "total_slides": total_slides,
        "parameters": parameters,
        "preliminary_result": {
            "class": preliminary_report.get("preliminary_wsi_class"),
            "type": preliminary_report.get("report_type"),
            "reason": preliminary_report.get("reason"),
            "disclaimer": preliminary_report.get("disclaimer"),
            "is_critical": preliminary_report.get("is_critical", False)
        },
        "class_summary": summary_by_class.to_dict(orient="records") if len(summary_by_class) > 0 else [],
        "roi_summary": summary_by_roi.to_dict(orient="records") if len(summary_by_roi) > 0 else [],
        "retention_summary": df_retention.to_dict(orient="records") if len(df_retention) > 0 else []
    }

    # Sanitizar report_data para evitar NaN o Infinity que rompen el JSON de SQLite
    report_data = sanitize_json_data(report_data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    return json_path
