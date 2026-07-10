import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
from django.contrib import messages
from django.conf import settings
from django.utils import timezone
from datetime import datetime

from .models import WSISample, AnalysisRun, ROI, CandidateCell, Prediction, GeneratedFigure
from .forms import WSISampleUploadForm, PipelineParametersForm
from .services.runner import start_analysis_async
from .services.inference import check_model_files

def dashboard(request):
    """
    Vista principal. Lista de muestras procesadas con su estado actual.
    """
    samples = WSISample.objects.all().order_by('-uploaded_at')
    
    # Obtener el último run de cada muestra
    runs_data = []
    running_count = 0
    completed_count = 0
    
    for sample in samples:
        last_run = sample.runs.all().order_by('-started_at').first()
        runs_data.append({
            'sample': sample,
            'last_run': last_run
        })
        if last_run:
            if last_run.status in ['PENDING', 'RUNNING']:
                running_count += 1
            elif last_run.status == 'COMPLETED':
                completed_count += 1

    return render(request, 'wsi_analysis/dashboard.html', {
        'runs_data': runs_data,
        'running_count': running_count,
        'completed_count': completed_count,
    })

def upload_sample(request):
    """
    Vista para subir o seleccionar una imagen WSI y ajustar parámetros.
    """
    if request.method == 'POST':
        sample_form = WSISampleUploadForm(request.POST, request.FILES)
        params_form = PipelineParametersForm(request.POST)

        if sample_form.is_valid() and params_form.is_valid():
            sample = sample_form.save(commit=False)

            # Manejar archivo subido físicamente vs ruta local
            uploaded_file = request.FILES.get('uploaded_file')
            if uploaded_file:
                # Guardar el archivo en la carpeta media/wsi
                wsi_dir = os.path.join(settings.MEDIA_ROOT, 'wsi')
                os.makedirs(wsi_dir, exist_ok=True)
                file_path = os.path.join(wsi_dir, uploaded_file.name)
                
                with open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                sample.file_path = file_path
            else:
                # Validar que exista la ruta local especificada
                local_path = sample_form.cleaned_data.get('file_path')
                if not os.path.exists(local_path):
                    messages.error(request, f"La ruta local especificada no existe: {local_path}")
                    return render(request, 'wsi_analysis/upload.html', {
                        'sample_form': sample_form,
                        'params_form': params_form
                    })
                sample.file_path = local_path

            # Guardar muestra
            sample.save()

            # Crear corrida de análisis
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = AnalysisRun.objects.create(
                sample=sample,
                run_id=run_id,
                status='PENDING',
                parameters=params_form.cleaned_data
            )

            # Iniciar pipeline en segundo plano de manera asíncrona (Thread Fallback)
            start_analysis_async(run.run_id)

            messages.success(request, f"Muestra '{sample.name}' creada y corrida {run_id} iniciada.")
            return redirect('dashboard')
        else:
            print("--- [ERROR] Formulario Inválido ---")
            print("Errores de sample_form:", sample_form.errors.as_data())
            print("Errores de params_form:", params_form.errors.as_data())
            print("-----------------------------------")
    else:
        sample_form = WSISampleUploadForm()
        params_form = PipelineParametersForm()

    return render(request, 'wsi_analysis/upload.html', {
        'sample_form': sample_form,
        'params_form': params_form
    })

def run_detail(request, run_id):
    """
    Vista detallada de una corrida específica de análisis.
    Muestra estadísticas agregadas y figuras de control de calidad.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    figures = run.figures.all()
    
    # Clasificación preliminar
    prelim = run.results_json.get('preliminary_result', {})
    class_summary = run.results_json.get('class_summary', [])
    retention_summary = run.results_json.get('retention_summary', [])
    
    # Agrupar figuras por categorías
    qc_figures = [f for f in figures if f.figure_type in ['thumbnail_wsi', 'mask_foreground', 'mask_artifacts', 'mask_clean']]
    analytic_figures = [f for f in figures if f.figure_type not in ['thumbnail_wsi', 'mask_foreground', 'mask_artifacts', 'mask_clean']]

    # Obtener el top 10 de ROIs prioritarios para mostrar en una tabla
    top_rois = run.rois.all().order_by('-priority_score', '-n_abnormal')[:10]

    # Calcular contadores del flujo
    total_rois = run.rois.count()
    from .models import Slide
    total_slides = Slide.objects.filter(roi__run=run).count()
    total_cells = run.candidates.count()

    return render(request, 'wsi_analysis/detail.html', {
        'run': run,
        'qc_figures': qc_figures,
        'analytic_figures': analytic_figures,
        'prelim': prelim,
        'class_summary': class_summary,
        'retention_summary': retention_summary,
        'top_rois': top_rois,
        'total_rois': total_rois,
        'total_slides': total_slides,
        'total_cells': total_cells,
    })

def run_viewer(request, run_id):
    """
    Visor interactivo espacial.
    Muestra los mapas, heatmaps, ROIs prioritarios y galerías de crops por clase.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    
    # Obtener figuras relevantes
    figures = {fig.figure_type: fig.image_path for fig in run.figures.all()}
    
    # Obtener candidatos categorizados optimizados con slide
    candidates = run.candidates.select_related('prediction', 'slide').all()
    
    # Crops de interés agrupados por clase clínica (mapeo seguro para plantillas)
    crops_by_class = {
        "Negative": [],
        "ASC_US": [],
        "LSIL": [],
        "ASC_H": [],
        "HSIL": [],
        "SCC": []
    }
    
    key_mapping = {
        "Negative for intraepithelial lesion": "Negative",
        "ASC-US": "ASC_US",
        "LSIL": "LSIL",
        "ASC-H": "ASC_H",
        "HSIL": "HSIL",
        "SCC": "SCC"
    }

    # Carpeta relativa en media para servir los archivos a la plantilla
    run_dir_name = f"sample_{run.sample.id}_{run.run_id}"
    
    for cand in candidates:
        if hasattr(cand, 'prediction') and cand.prediction.pred_label in key_mapping:
            lbl = cand.prediction.pred_label
            safe_key = key_mapping[lbl]
            # Solo añadir si no hemos sobrellenado la visualización (ej. max 24 por clase)
            if len(crops_by_class[safe_key]) < 24:
                # Mapear a la carpeta según tipo
                subdir = "uncertain_candidates"
                if cand.candidate_type == "single_cell":
                    subdir = "single_cells"
                elif cand.candidate_type == "cell_cluster":
                    subdir = "cell_clusters"

                crops_by_class[safe_key].append({
                    'crop_name': cand.crop_name,
                    'type': cand.get_candidate_type_display(),
                    'nuc_count': cand.nuc_count,
                    'confidence': cand.prediction.confidence,
                    'x_wsi': cand.x_wsi,
                    'y_wsi': cand.y_wsi,
                    'slide_id': cand.slide.slide_id if cand.slide else "N/A",
                    'x_slide': cand.x_slide if cand.x_slide is not None else "N/A",
                    'y_slide': cand.y_slide if cand.y_slide is not None else "N/A",
                    'src': f"results/{run_dir_name}/candidates/{subdir}/{cand.crop_name}"
                })

    return render(request, 'wsi_analysis/viewer.html', {
        'run': run,
        'figures': figures,
        'crops_by_class': crops_by_class
    })

def model_config(request):
    """
    Verificación del estado de los archivos de modelo en artefactos/
    """
    artefactos_dir = os.path.join(settings.BASE_DIR, "artefactos")
    status = check_model_files(artefactos_dir)
    
    label_map = None
    classes = []
    
    # Intentar leer metadatos de clases
    if status["json_exists"]:
        try:
            json_path = os.path.join(artefactos_dir, "inference_artifacts.json")
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            label_map = cfg.get("label_map", {})
            classes = list(cfg.get("idx_to_label", {}).values())
        except Exception:
            pass

    return render(request, 'wsi_analysis/config.html', {
        'status': status,
        'artefactos_dir': artefactos_dir,
        'label_map': label_map,
        'classes': classes
    })

def delete_sample(request, sample_id):
    """
    Elimina una muestra y todos sus registros físicos/corrida asociados.
    No permite la eliminación si hay un análisis en ejecución o pendiente.
    """
    sample = get_object_or_404(WSISample, id=sample_id)
    
    # Verificar si hay corridas activas (RUNNING o PENDING)
    active_runs = sample.runs.filter(status__in=['PENDING', 'RUNNING'])
    if active_runs.exists():
        messages.error(request, f"No se puede eliminar la muestra '{sample.name}' porque tiene un análisis en ejecución o pendiente.")
        return redirect('dashboard')
        
    name = sample.name
    
    # Borrar carpeta física de resultados si existe
    for run in sample.runs.all():
        run_dir_name = f"sample_{sample.id}_{run.run_id}"
        run_dir = os.path.join(settings.MEDIA_ROOT, 'results', run_dir_name)
        if os.path.exists(run_dir):
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

    sample.delete()
    messages.warning(request, f"Muestra '{name}' eliminada correctamente del sistema.")
    return redirect('dashboard')

# API Endpoints para refresco dinámico / AJAX
def api_run_status(request, run_id):
    """
    Retorna el estado actual de una corrida de análisis en formato JSON.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    return JsonResponse({
        'run_id': run.run_id,
        'status': run.status,
        'completed': run.status in ['COMPLETED', 'FAILED'],
        'started_at': run.started_at.isoformat() if run.started_at else None,
        'completed_at': run.completed_at.isoformat() if run.completed_at else None,
        'error_message': run.error_message
    })

# Servido de Descargas
def download_report(request, run_id):
    """
    Descargar reporte HTML de la corrida.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    run_dir_name = f"sample_{run.sample.id}_{run.run_id}"
    file_path = os.path.join(settings.MEDIA_ROOT, 'results', run_dir_name, 'report', 'reporte_wsi.html')
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='text/html', filename=f"reporte_wsi_{run_id}.html")
    raise Http404("Reporte HTML no encontrado en el servidor.")

def download_predictions(request, run_id):
    """
    Descargar predicciones en CSV.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    run_dir_name = f"sample_{run.sample.id}_{run.run_id}"
    file_path = os.path.join(settings.MEDIA_ROOT, 'results', run_dir_name, 'predictions', 'predictions_candidates.csv')
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='text/csv', filename=f"predicciones_celulas_{run_id}.csv")
    raise Http404("Archivo CSV de predicciones no encontrado.")

def download_json(request, run_id):
    """
    Descargar reporte en formato JSON.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    run_dir_name = f"sample_{run.sample.id}_{run.run_id}"
    file_path = os.path.join(settings.MEDIA_ROOT, 'results', run_dir_name, 'report', 'reporte_wsi.json')
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='application/json', filename=f"reporte_wsi_{run_id}.json")
    raise Http404("Reporte JSON no encontrado.")

def export_sample_crops(request, run_id):
    """
    Exporta la muestra, los ROIs y sus slides correspondientes en un archivo ZIP estructurado:
    WSI_Name/
      ROI_000/
        ROI_000.jpg
        Slide_1.jpg
        Slide_2.jpg
      ...
    """
    import zipfile
    import re
    import cv2
    from .models import Slide
    from .services.wsi_reader import read_wsi_region, clear_wsi_cache

    run = get_object_or_404(AnalysisRun, run_id=run_id)
    wsi_path = run.sample.file_path
    
    # Directorio de exportaciones
    exports_dir = os.path.join(settings.MEDIA_ROOT, 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Nombre del archivo zip basado en el ID de la muestra y la corrida
    zip_filename = f"export_{run.sample.id}_{run.run_id}.zip"
    zip_path = os.path.join(exports_dir, zip_filename)
    
    # Limpiar caracteres inválidos para el nombre de la carpeta raíz dentro del zip
    safe_sample_name = re.sub(r'[^a-zA-Z0-9_]', '_', run.sample.name)
    if not safe_sample_name:
        safe_sample_name = f"sample_{run.sample.id}"
        
    prepare_mode = request.GET.get('prepare') == '1'
    
    if prepare_mode:
        # Si ya existe, no es necesario volver a crearlo
        if os.path.exists(zip_path):
            return JsonResponse({'status': 'ready', 'message': 'El archivo ya estaba listo.'})
            
        import uuid
        temp_zip_filename = f"export_{run.sample.id}_{run.run_id}_{uuid.uuid4().hex}.tmp"
        temp_zip_path = os.path.join(exports_dir, temp_zip_filename)
        clear_wsi_cache()
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Obtener ROIs ordenados
                rois = run.rois.all().order_by('roi_id')
                
                for roi in rois:
                    roi_w = roi.x2_wsi - roi.x1_wsi
                    roi_h = roi.y2_wsi - roi.y1_wsi
                    
                    # 1. Recortar e incorporar la imagen completa del ROI
                    if roi_w > 0 and roi_h > 0:
                        roi_img = read_wsi_region(wsi_path, roi.x1_wsi, roi.y1_wsi, roi_w, roi_h)
                        # Convertir RGB (retornado por wsi_reader) a BGR para guardar correctamente
                        roi_img_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
                        
                        _, roi_jpg = cv2.imencode('.jpg', roi_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        
                        roi_in_zip_path = f"{safe_sample_name}/{roi.roi_id}/{roi.roi_id}.jpg"
                        zip_file.writestr(roi_in_zip_path, roi_jpg.tobytes())
                    
                    # 2. Recortar e incorporar las imágenes de cada Slide dentro de este ROI
                    slides = roi.slides.all().order_by('slide_id')
                    for idx, slide in enumerate(slides, start=1):
                        slide_w = slide.x2_wsi - slide.x1_wsi
                        slide_h = slide.y2_wsi - slide.y1_wsi
                        
                        if slide_w > 0 and slide_h > 0:
                            slide_img = read_wsi_region(wsi_path, slide.x1_wsi, slide.y1_wsi, slide_w, slide_h)
                            slide_img_bgr = cv2.cvtColor(slide_img, cv2.COLOR_RGB2BGR)
                            
                            _, slide_jpg = cv2.imencode('.jpg', slide_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            
                            slide_in_zip_path = f"{safe_sample_name}/{roi.roi_id}/Slide_{idx}.jpg"
                            zip_file.writestr(slide_in_zip_path, slide_jpg.tobytes())
                            
            # Intentar renombrar el archivo temporal de forma segura
            try:
                os.replace(temp_zip_path, zip_path)
            except OSError:
                # Si falla porque el destino ya existe y está bloqueado/utilizado por otro proceso,
                # verificamos si el archivo de destino existe para darlo por bueno
                if os.path.exists(zip_path):
                    try:
                        os.remove(temp_zip_path)
                    except OSError:
                        pass
                else:
                    raise
            return JsonResponse({'status': 'ready', 'message': 'Exportación generada correctamente.'})
            
        except Exception as e:
            if os.path.exists(temp_zip_path):
                try:
                    os.remove(temp_zip_path)
                except OSError:
                    pass
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        finally:
            clear_wsi_cache()
            
    else:
        # Modo descarga directa
        if not os.path.exists(zip_path):
            raise Http404("El archivo ZIP de exportación no ha sido preparado aún.")
        return FileResponse(open(zip_path, 'rb'), content_type='application/zip', filename=f"{safe_sample_name}_export.zip")
