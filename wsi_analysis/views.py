import os
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

    return render(request, 'wsi_analysis/detail.html', {
        'run': run,
        'qc_figures': qc_figures,
        'analytic_figures': analytic_figures,
        'prelim': prelim,
        'class_summary': class_summary,
        'retention_summary': retention_summary,
        'top_rois': top_rois
    })

def run_viewer(request, run_id):
    """
    Visor interactivo espacial.
    Muestra los mapas, heatmaps, ROIs prioritarios y galerías de crops por clase.
    """
    run = get_object_or_404(AnalysisRun, run_id=run_id)
    
    # Obtener figuras relevantes
    figures = {fig.figure_type: fig.image_path for fig in run.figures.all()}
    
    # Obtener candidatos categorizados
    candidates = run.candidates.select_related('prediction').all()
    
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
