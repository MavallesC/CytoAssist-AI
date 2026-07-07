import os
import json
import cv2
import traceback
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from ..models import WSISample, AnalysisRun, ROI, Slide, CandidateCell, Prediction, GeneratedFigure
from .wsi_reader import get_wsi_thumbnail, get_wsi_dimensions, read_wsi_region, clear_wsi_cache
from .masks import foreground_mask_from_thumb, detect_large_artifacts, stain_mask_purple_thumb, build_clean_mask
from .roi_extraction import extract_rois_grid
from .candidate_detection import scan_slides_for_candidates
from .inference import load_inference_artifacts, predict_crop, predict_crops_batch, check_model_files
from .aggregation import build_class_summary, build_roi_summary, infer_preliminary_wsi_class
from .visualization import (
    generate_thumbnail_wsi_plot, generate_mask_foreground_plot, generate_mask_artifacts_plot,
    generate_mask_clean_plot, generate_overlay_rois_plot, generate_mapa_candidatos_y_clasificados_plot,
    generate_heatmap_densidad_candidatos_plot, generate_heatmap_densidad_anormales_plot,
    generate_overlay_rois_prioritarios_plot, generate_barplot_resumen_por_clase_plot,
    generate_barplot_retencion_celular_plot
)
from .reports import generate_csv_reports, generate_html_report, generate_json_report

def run_analysis_pipeline(run_id):
    """
    Función síncrona que ejecuta secuencialmente todo el pipeline de procesamiento de WSI de forma optimizada.
    """
    # Limpiar caché de WSI al iniciar
    clear_wsi_cache()
    
    pipeline_start = time.time()
    timeline = {}
    
    try:
        run = AnalysisRun.objects.get(run_id=run_id)
    except AnalysisRun.DoesNotExist:
        print(f"Error: Corrida {run_id} no existe en la BD.")
        return

    # Actualizar estado a RUNNING
    run.status = 'RUNNING'
    run.started_at = timezone.now()
    run.save()

    sample = run.sample
    params = run.parameters
    wsi_path = sample.file_path

    # Crear directorios de resultados
    run_dir_name = f"sample_{sample.id}_{run_id}"
    run_dir = os.path.join(settings.MEDIA_ROOT, 'results', run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # Subcarpetas según especificación
    qc_dir = os.path.join(run_dir, "qc")
    rois_dir = os.path.join(run_dir, "rois")
    candidates_dir = os.path.join(run_dir, "candidates")
    predictions_dir = os.path.join(run_dir, "predictions")
    figures_dir = os.path.join(run_dir, "figures")
    report_dir = os.path.join(run_dir, "report")

    # Carpetas para guardar imágenes físicas de los crops
    single_cells_out = os.path.join(candidates_dir, "single_cells")
    cell_clusters_out = os.path.join(candidates_dir, "cell_clusters")
    uncertain_out = os.path.join(candidates_dir, "uncertain_candidates")

    for d in [qc_dir, rois_dir, candidates_dir, predictions_dir, figures_dir, report_dir,
              single_cells_out, cell_clusters_out, uncertain_out]:
        os.makedirs(d, exist_ok=True)

    try:
        # 1. Carga o selección del WSI y Generación de Thumbnail
        print(f"[{run_id}] Paso 1: Generando thumbnail del WSI...")
        t_start = time.time()
        thumb_rgb = get_wsi_thumbnail(wsi_path, max_side=2000)
        thumb_path = os.path.join(qc_dir, "thumbnail_wsi.png")
        cv2.imwrite(thumb_path, cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
        timeline["1. Thumbnail & Carga"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 1 completado en {timeline['1. Thumbnail & Carga']:.4f} s")
        
        # 2. Segmentación del foreground (muestra útil)
        print(f"[{run_id}] Paso 2: Calculando máscara de foreground...")
        t_start = time.time()
        fg_mask = foreground_mask_from_thumb(thumb_rgb)
        fg_mask_path = os.path.join(qc_dir, "mask_foreground.png")
        cv2.imwrite(fg_mask_path, fg_mask)
        timeline["2. Máscara Foreground"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 2 completado en {timeline['2. Máscara Foreground']:.4f} s")

        # 3. Detección de artefactos grandes
        print(f"[{run_id}] Paso 3: Calculando máscara de artefactos...")
        t_start = time.time()
        artifact_mask = detect_large_artifacts(thumb_rgb, fg_mask)
        art_mask_path = os.path.join(qc_dir, "mask_artifacts.png")
        cv2.imwrite(art_mask_path, artifact_mask)
        timeline["3. Máscara Artefactos"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 3 completado en {timeline['3. Máscara Artefactos']:.4f} s")

        # 4. Construcción de máscara limpia final
        print(f"[{run_id}] Paso 4: Construyendo máscara limpia...")
        t_start = time.time()
        purple_mask = stain_mask_purple_thumb(thumb_rgb)
        clean_mask = build_clean_mask(purple_mask, artifact_mask, buffer_px=2)
        clean_mask_path = os.path.join(qc_dir, "mask_clean.png")
        cv2.imwrite(clean_mask_path, clean_mask)

        # Generar plots del control de calidad inicial
        generate_thumbnail_wsi_plot(thumb_rgb, os.path.join(qc_dir, "thumbnail_wsi.png"))
        generate_mask_foreground_plot(thumb_rgb, fg_mask, os.path.join(qc_dir, "mask_foreground.png"))
        generate_mask_artifacts_plot(thumb_rgb, artifact_mask, os.path.join(qc_dir, "mask_artifacts.png"))
        generate_mask_clean_plot(thumb_rgb, clean_mask, os.path.join(qc_dir, "mask_clean.png"))

        # Guardar en base de datos las figuras de QC
        GeneratedFigure.objects.create(run=run, figure_type='thumbnail_wsi', image_path=f"results/{run_dir_name}/qc/thumbnail_wsi.png")
        GeneratedFigure.objects.create(run=run, figure_type='mask_foreground', image_path=f"results/{run_dir_name}/qc/mask_foreground.png")
        GeneratedFigure.objects.create(run=run, figure_type='mask_artifacts', image_path=f"results/{run_dir_name}/qc/mask_artifacts.png")
        GeneratedFigure.objects.create(run=run, figure_type='mask_clean', image_path=f"results/{run_dir_name}/qc/mask_clean.png")
        
        timeline["4. Máscara Limpia & QC"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 4 completado en {timeline['4. Máscara Limpia & QC']:.4f} s")

        # 5. Extracción de ROIs
        print(f"[{run_id}] Paso 5: Extrayendo cuadrícula de ROIs...")
        t_start = time.time()
        df_rois = extract_rois_grid(
            wsi_path=wsi_path,
            thumb_rgb=thumb_rgb,
            clean_mask=clean_mask,
            roi_size=200,
            stride=200,
            min_clean_frac=params.get('min_clean_frac', 0.20)
        )

        if df_rois.empty:
            raise ValueError("No se encontraron ROIs que cumplan con el porcentaje mínimo de muestra útil.")

        # Guardar metadatos de ROIs en CSV
        rois_csv_path = os.path.join(rois_dir, "rois_grid.csv")
        df_rois.to_csv(rois_csv_path, index=False)
        
        # Generar plot con rejilla de ROIs
        generate_overlay_rois_plot(thumb_rgb, df_rois, os.path.join(rois_dir, "overlay_rois.png"))
        GeneratedFigure.objects.create(run=run, figure_type='overlay_rois', image_path=f"results/{run_dir_name}/rois/overlay_rois.png")

        # Guardar instancias de ROI en BD
        roi_db_instances = {}
        for _, row in df_rois.iterrows():
            roi_obj = ROI.objects.create(
                run=run,
                roi_id=row['roi_id'],
                x1_thumb=row['x1_thumb'],
                y1_thumb=row['y1_thumb'],
                x2_thumb=row['x2_thumb'],
                y2_thumb=row['y2_thumb'],
                x1_wsi=row['x1_wsi'],
                y1_wsi=row['y1_wsi'],
                x2_wsi=row['x2_wsi'],
                y2_wsi=row['y2_wsi']
            )
            roi_db_instances[row['roi_id']] = roi_obj

        # Subdividir ROIs en Slides
        slide_db_instances = {}
        slides_list = []
        slide_width = params.get('slide_width', 1376)
        slide_height = params.get('slide_height', 1020)

        for roi_id, roi_obj in roi_db_instances.items():
            X1, Y1, X2, Y2 = roi_obj.x1_wsi, roi_obj.y1_wsi, roi_obj.x2_wsi, roi_obj.y2_wsi
            slide_idx = 0
            for y in range(Y1, Y2, slide_height):
                for x in range(X1, X2, slide_width):
                    slide_x1 = x
                    slide_y1 = y
                    slide_x2 = min(x + slide_width, X2)
                    slide_y2 = min(y + slide_height, Y2)
                    
                    # Evitar slides demasiado pequeños
                    if (slide_x2 - slide_x1) >= 64 and (slide_y2 - slide_y1) >= 64:
                        s_id = f"Slide_{roi_obj.roi_id}_{slide_idx:02d}"
                        slide_obj = Slide.objects.create(
                            roi=roi_obj,
                            slide_id=s_id,
                            x1_wsi=slide_x1,
                            y1_wsi=slide_y1,
                            x2_wsi=slide_x2,
                            y2_wsi=slide_y2
                        )
                        slide_db_instances[s_id] = slide_obj
                        slides_list.append({
                            "slide_id": s_id,
                            "roi_id": roi_obj.roi_id,
                            "x1_wsi": slide_x1,
                            "y1_wsi": slide_y1,
                            "x2_wsi": slide_x2,
                            "y2_wsi": slide_y2
                        })
                        slide_idx += 1
            
        timeline["5. Extracción ROIs"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 5 completado en {timeline['5. Extracción ROIs']:.4f} s (ROIs: {len(df_rois)}, Slides: {len(slides_list)})")

        # 6. Detección de candidatos celulares dentro de los Slides
        print(f"[{run_id}] Paso 6: Detectando candidatos celulares en slides...")
        t_start = time.time()
        candidates_raw = scan_slides_for_candidates(wsi_path, slides_list, params)

        if not candidates_raw:
            raise ValueError("No se detectó ningún candidato celular en los slides seleccionados.")
            
        timeline["6. Detección Candidatos"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 6 completado en {timeline['6. Detección Candidatos']:.4f} s (Candidatos crudos: {len(candidates_raw)})")

        # 7. Inferencia y Clasificación con Modelos IA
        print(f"[{run_id}] Paso 7: Clasificando candidatos celulares...")
        t_start = time.time()
        
        # Cargar artefactos reales de la carpeta local
        artefactos_dir = os.path.join(settings.BASE_DIR, "artefactos")
        model_status = check_model_files(artefactos_dir)
        
        model_artifacts = None
        if model_status["json_exists"] and model_status["backbone_exists"] and model_status["catboost_exists"]:
            try:
                model_artifacts = load_inference_artifacts(artefactos_dir)
                print(f"[{run_id}] Modelo IA cargado exitosamente.")
            except Exception as e:
                print(f"[{run_id}] Alerta: Error al inicializar modelo real. Se usará clasificador heurístico: {e}")
        else:
            print(f"[{run_id}] Aviso: No se encontraron los artefactos en {artefactos_dir}. Se usará simulación.")

        # Escalar coordenadas WSI -> thumbnail
        h_wsi, w_wsi = get_wsi_dimensions(wsi_path)
        sx = thumb_rgb.shape[1] / float(w_wsi)
        sy = thumb_rgb.shape[0] / float(h_wsi)

        # Filtrar candidatos que serán clasificados por IA
        cands_to_classify = []
        cands_to_classify_indices = []
        for idx, cand in enumerate(candidates_raw):
            if cand["candidate_type"] in ['single_cell', 'cell_cluster', 'uncertain_candidate']:
                cands_to_classify.append(cand)
                cands_to_classify_indices.append(idx)

        # Inferencia por lotes
        print(f"[{run_id}] Realizando inferencia para {len(cands_to_classify)} candidatos...")
        t_batch_start = time.time()
        batch_crops = [c["crop_image"] for c in cands_to_classify]
        batch_preds = predict_crops_batch(batch_crops, model_artifacts, batch_size=64)
        t_batch_end = time.time()
        print(f"[{run_id}] Inferencia por lotes de IA completada en {t_batch_end - t_batch_start:.4f} s.")

        # Mapear predicciones por índice original
        prediction_results = {}
        for list_idx, orig_idx in enumerate(cands_to_classify_indices):
            prediction_results[orig_idx] = batch_preds[list_idx]

        # Procesar predicciones y guardar imágenes físicas de los crops
        records_candidates = []
        records_predictions = []
        
        # Contadores para el embudo de retención celular
        counts_retention = {
            "total_extracted": len(candidates_raw),
            "nuc_count_1": 0,
            "nuc_count_gt_1": 0,
            "single_cell": 0,
            "cell_cluster": 0,
            "low_quality": 0,
            "artifact_like": 0,
            "uncertain_candidate": 0,
            "classified": 0
        }

        # Guardar en base de datos en una sola transacción
        print(f"[{run_id}] Guardando {len(candidates_raw)} cultivos en disco...")
        t_disk_start = time.time()
        
        candidate_cells_to_create = []
        predictions_data = []

        for idx, cand in enumerate(candidates_raw):
            x_w = cand["x_wsi"]
            y_w = cand["y_wsi"]
            sz = cand["crop_size"]
            
            # Calcular centros sobre el thumbnail
            cx_wsi = x_w + sz / 2.0
            cy_wsi = y_w + sz / 2.0
            x_t = float(cx_wsi * sx)
            y_t = float(cy_wsi * sy)

            crop_img = cand["crop_image"]
            candidate_type = cand["candidate_type"]
            nuc_c = cand["nuc_count"]

            # Contabilizar estadísticas de retención
            if nuc_c == 1:
                counts_retention["nuc_count_1"] += 1
            elif nuc_c > 1:
                counts_retention["nuc_count_gt_1"] += 1

            counts_retention[candidate_type] += 1

            # Guardar el archivo físico del crop si corresponde
            crop_filename = f"crop_{run_id}_{idx:05d}.png"
            
            if candidate_type == 'single_cell':
                cv2.imwrite(os.path.join(single_cells_out, crop_filename), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            elif candidate_type == 'cell_cluster':
                cv2.imwrite(os.path.join(cell_clusters_out, crop_filename), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            elif candidate_type == 'uncertain_candidate':
                cv2.imwrite(os.path.join(uncertain_out, crop_filename), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

            # Preparar CandidateCell
            roi_obj = roi_db_instances.get(cand["roi_id"])
            slide_obj = slide_db_instances.get(cand.get("slide_id"))
            cell_db = CandidateCell(
                run=run,
                roi=roi_obj,
                slide=slide_obj,
                x_slide=cand.get("x_slide"),
                y_slide=cand.get("y_slide"),
                crop_name=crop_filename,
                x_wsi=x_w,
                y_wsi=y_w,
                x_thumb=x_t,
                y_thumb=y_t,
                crop_size=sz,
                nuc_count=nuc_c,
                focus_score=cand["focus_score"],
                edge_density=cand["edge_density"],
                cyto_frac=cand["cyto_frac"],
                candidate_type=candidate_type
            )
            candidate_cells_to_create.append(cell_db)

            # Inferencia recuperada del lote o mock
            if candidate_type in ['single_cell', 'cell_cluster', 'uncertain_candidate']:
                pred_res = prediction_results[idx]
                pred_label = pred_res["pred_label"]
                confidence = pred_res["confidence"]
                probs = pred_res["probabilities"]
                
                counts_retention["classified"] += 1
            else:
                pred_label = "Negative for intraepithelial lesion"
                confidence = 0.0
                probs = {}

            # Guardar datos para crear la predicción correspondiente
            predictions_data.append({
                "cell_index": len(candidate_cells_to_create) - 1,
                "pred_label": pred_label,
                "confidence": confidence,
                "probabilities": probs
            })

            # Guardar metadatos para reportes
            records_candidates.append({
                "roi_id": cand["roi_id"],
                "slide_id": cand.get("slide_id"),
                "x_slide": cand.get("x_slide"),
                "y_slide": cand.get("y_slide"),
                "crop_name": crop_filename,
                "x_wsi": x_w,
                "y_wsi": y_w,
                "x_thumb": x_t,
                "y_thumb": y_t,
                "crop_size": sz,
                "nuc_count": nuc_c,
                "focus_score": cand["focus_score"],
                "edge_density": cand["edge_density"],
                "cyto_frac": cand["cyto_frac"],
                "candidate_type": candidate_type
            })

            if candidate_type in ['single_cell', 'cell_cluster', 'uncertain_candidate']:
                records_predictions.append({
                    "roi_id": cand["roi_id"],
                    "slide_id": cand.get("slide_id"),
                    "x_slide": cand.get("x_slide"),
                    "y_slide": cand.get("y_slide"),
                    "crop_name": crop_filename,
                    "x_wsi": x_w,
                    "y_wsi": y_w,
                    "x_thumb": x_t,
                    "y_thumb": y_t,
                    "nuc_count": nuc_c,
                    "candidate_type": candidate_type,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    **{f"prob_{k.replace('prob_', '')}": v for k, v in probs.items()}
                })

        t_disk_end = time.time()
        print(f"[{run_id}] Escritura en disco completada en {t_disk_end - t_disk_start:.4f} s.")

        print(f"[{run_id}] Guardando registros en la BD...")
        t_db_start = time.time()
        with transaction.atomic():
            created_cells = CandidateCell.objects.bulk_create(candidate_cells_to_create)
            
            predictions_to_create = []
            for pred_item in predictions_data:
                cell_obj = created_cells[pred_item["cell_index"]]
                predictions_to_create.append(
                    Prediction(
                        candidate=cell_obj,
                        pred_label=pred_item["pred_label"],
                        confidence=pred_item["confidence"],
                        probabilities=pred_item["probabilities"]
                    )
                )
            Prediction.objects.bulk_create(predictions_to_create)
            
        t_db_end = time.time()
        print(f"[{run_id}] Escritura en BD completada en {t_db_end - t_db_start:.4f} s.")

        df_cands_meta = pd.DataFrame(records_candidates)
        df_cands_meta.to_csv(os.path.join(candidates_dir, "candidates_metadata.csv"), index=False)

        df_preds_spatial = pd.DataFrame(records_predictions)
        
        timeline["7. Inferencia & Guardado BD"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 7 completado en {timeline['7. Inferencia & Guardado BD']:.4f} s")

        # 8. Agregación de Resultados y Construcción de Reporte Lógico
        print(f"[{run_id}] Paso 8: Agregando estadísticas...")
        t_start = time.time()
        summary_by_class = build_class_summary(df_preds_spatial)
        summary_by_roi = build_roi_summary(df_preds_spatial)
        
        # Calcular el resumen del embudo de retención celular
        rows_retention = [
            {"criterio": "Total Candidatos Detectados", "n": counts_retention["total_extracted"]},
            {"criterio": "Crops con Núcleo = 1", "n": counts_retention["nuc_count_1"]},
            {"criterio": "Crops con Núcleos > 1", "n": counts_retention["nuc_count_gt_1"]},
            {"criterio": "Células Individuales (single_cell)", "n": counts_retention["single_cell"]},
            {"criterio": "Agrupaciones (cell_cluster)", "n": counts_retention["cell_cluster"]},
            {"criterio": "Candidatos Dudosos (uncertain)", "n": counts_retention["uncertain_candidate"]},
            {"criterio": "Baja Calidad (filtrados)", "n": counts_retention["low_quality"]},
            {"criterio": "Artefactos (filtrados)", "n": counts_retention["artifact_like"]},
            {"criterio": "Total Clasificados por IA", "n": counts_retention["classified"]}
        ]
        df_retention = pd.DataFrame(rows_retention)
        df_retention["porcentaje_sobre_total_crops"] = df_retention["n"] / max(1, counts_retention["total_extracted"]) * 100

        # Actualizar base de datos de ROIs con los conteos de anormales
        rois_to_update = []
        for _, r_row in summary_by_roi.iterrows():
            try:
                roi_obj = ROI.objects.get(run=run, roi_id=r_row['roi_id'])
                roi_obj.n_candidates = int(r_row['n_predicted_cells'])
                roi_obj.n_abnormal = int(r_row['n_abnormal_cells'])
                roi_obj.priority_score = float(r_row['max_priority'])
                rois_to_update.append(roi_obj)
            except Exception:
                pass
        
        if rois_to_update:
            with transaction.atomic():
                ROI.objects.bulk_update(rois_to_update, ['n_candidates', 'n_abnormal', 'priority_score'])

        # Determinar clase preliminar asistida
        preliminary_report = infer_preliminary_wsi_class(
            df_preds_spatial,
            min_conf_priority=params.get('min_conf_priority', 0.50),
            min_cells_for_global_result=100
        )
        
        timeline["8. Agregación de Resultados"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 8 completado en {timeline['8. Agregación de Resultados']:.4f} s")

        # 9. Visualización Espacial y plots horizontales
        print(f"[{run_id}] Paso 9: Generando figuras analíticas...")
        t_start = time.time()
        
        mapa_pts_path = os.path.join(figures_dir, "mapa_candidatos_y_clasificados.png")
        generate_mapa_candidatos_y_clasificados_plot(
            thumb_rgb, df_cands_meta, df_preds_spatial, df_rois, mapa_pts_path
        )
        GeneratedFigure.objects.create(run=run, figure_type='mapa_candidatos_y_clasificados', image_path=f"results/{run_dir_name}/figures/mapa_candidatos_y_clasificados.png")

        heatmap_cand_path = os.path.join(figures_dir, "heatmap_densidad_candidatos.png")
        generate_heatmap_densidad_candidatos_plot(thumb_rgb, df_cands_meta, heatmap_cand_path)
        GeneratedFigure.objects.create(run=run, figure_type='heatmap_densidad_candidatos', image_path=f"results/{run_dir_name}/figures/heatmap_densidad_candidatos.png")

        heatmap_anorm_path = os.path.join(figures_dir, "heatmap_densidad_anormales.png")
        generate_heatmap_densidad_anormales_plot(thumb_rgb, df_preds_spatial, heatmap_anorm_path)
        GeneratedFigure.objects.create(run=run, figure_type='heatmap_densidad_anormales', image_path=f"results/{run_dir_name}/figures/heatmap_densidad_anormales.png")

        overlay_prior_path = os.path.join(figures_dir, "overlay_rois_prioritarios.png")
        generate_overlay_rois_prioritarios_plot(thumb_rgb, df_rois, summary_by_roi, overlay_prior_path)
        GeneratedFigure.objects.create(run=run, figure_type='overlay_rois_prioritarios', image_path=f"results/{run_dir_name}/figures/overlay_rois_prioritarios.png")

        barplot_clase_path = os.path.join(figures_dir, "barplot_resumen_por_clase.png")
        generate_barplot_resumen_por_clase_plot(summary_by_class, barplot_clase_path)
        GeneratedFigure.objects.create(run=run, figure_type='barplot_resumen_por_clase', image_path=f"results/{run_dir_name}/figures/barplot_resumen_por_clase.png")

        barplot_ret_path = os.path.join(figures_dir, "barplot_retencion_celular.png")
        generate_barplot_retencion_celular_plot(df_retention, barplot_ret_path)
        GeneratedFigure.objects.create(run=run, figure_type='barplot_retencion_celular', image_path=f"results/{run_dir_name}/figures/barplot_retencion_celular.png")
        
        timeline["9. Generación Figuras"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 9 completado en {timeline['9. Generación Figuras']:.4f} s")

        # 10. Generación de Reportes Estructurados (CSV, JSON, HTML)
        print(f"[{run_id}] Paso 10: Compilando reportes finales...")
        t_start = time.time()
        
        # Exportar CSVs
        generate_csv_reports(run_dir, df_cands_meta, df_preds_spatial, summary_by_class, summary_by_roi, df_retention)
        
        # Exportar JSON
        json_rep_path = generate_json_report(
            run_dir, sample.name, run_id, preliminary_report, summary_by_class, summary_by_roi, df_retention, params
        )
        
        # Exportar HTML
        html_rep_path = generate_html_report(
            run_dir, sample.name, run_id, preliminary_report, summary_by_class, summary_by_roi, df_retention, params
        )

        # Leer reporte JSON final para almacenarlo en la base de datos
        with open(json_rep_path, "r", encoding="utf-8") as f:
            report_dict = json.load(f)

        # Finalización exitosa
        run.results_json = report_dict
        run.status = 'COMPLETED'
        run.completed_at = timezone.now()
        run.error_message = None
        run.save()
        
        timeline["10. Reportes Finales"] = time.time() - t_start
        print(f"[{run_id}] >>> Paso 10 completado en {timeline['10. Reportes Finales']:.4f} s")
        
        # Reportar resumen total de tiempos en el terminal
        print(f"[{run_id}] ===============================================")
        print(f"[{run_id}] RESUMEN DE TIEMPOS DE PROCESAMIENTO:")
        total_time = time.time() - pipeline_start
        for step, dur in timeline.items():
            pct = (dur / total_time) * 100
            print(f"[{run_id}]   - {step}: {dur:.4f} s ({pct:.2f}%)")
        print(f"[{run_id}] TIEMPO TOTAL DEL PIPELINE: {total_time:.4f} s")
        print(f"[{run_id}] ===============================================")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{run_id}] ERROR en el pipeline: {str(e)}")
        print(tb)
        
        run.status = 'FAILED'
        run.completed_at = timezone.now()
        run.error_message = f"{str(e)}\n\n{tb}"
        run.save()
    finally:
        # Limpiar el caché al terminar
        clear_wsi_cache()

def start_analysis_async(run_id):
    """
    Inicia la ejecución del análisis citológico de forma asíncrona.
    Utiliza hilos de Python nativos como fallback seguro y rápido.
    """
    thread = threading.Thread(target=run_analysis_pipeline, args=(run_id,), daemon=True)
    thread.start()
    return thread
