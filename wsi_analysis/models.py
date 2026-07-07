from django.db import models

class WSISample(models.Model):
    name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class AnalysisRun(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pendiente'),
        ('RUNNING', 'Ejecutando'),
        ('COMPLETED', 'Completado'),
        ('FAILED', 'Fallido'),
    ]

    sample = models.ForeignKey(WSISample, on_delete=models.CASCADE, related_name='runs')
    run_id = models.CharField(max_length=100, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(null=True, blank=True)
    parameters = models.JSONField(default=dict)
    results_json = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.sample.name} - Run {self.run_id} ({self.status})"

class ROI(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name='rois')
    roi_id = models.CharField(max_length=50)
    x1_thumb = models.IntegerField()
    y1_thumb = models.IntegerField()
    x2_thumb = models.IntegerField()
    y2_thumb = models.IntegerField()
    x1_wsi = models.IntegerField()
    y1_wsi = models.IntegerField()
    x2_wsi = models.IntegerField()
    y2_wsi = models.IntegerField()
    n_candidates = models.IntegerField(default=0)
    n_abnormal = models.IntegerField(default=0)
    priority_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.roi_id} in Run {self.run.run_id}"

class Slide(models.Model):
    roi = models.ForeignKey(ROI, on_delete=models.CASCADE, related_name='slides')
    slide_id = models.CharField(max_length=50)
    x1_wsi = models.IntegerField()
    y1_wsi = models.IntegerField()
    x2_wsi = models.IntegerField()
    y2_wsi = models.IntegerField()

    def __str__(self):
        return f"{self.slide_id} en {self.roi.roi_id}"

class CandidateCell(models.Model):
    CANDIDATE_TYPES = [
        ('single_cell', 'Célula Individual'),
        ('cell_cluster', 'Agrupación Celular'),
        ('low_quality', 'Baja Calidad'),
        ('artifact_like', 'Artefacto'),
        ('uncertain_candidate', 'Candidato Dudoso'),
    ]

    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name='candidates')
    roi = models.ForeignKey(ROI, on_delete=models.SET_NULL, null=True, blank=True, related_name='candidates')
    slide = models.ForeignKey(Slide, on_delete=models.SET_NULL, null=True, blank=True, related_name='candidates')
    x_slide = models.IntegerField(null=True, blank=True)
    y_slide = models.IntegerField(null=True, blank=True)
    crop_name = models.CharField(max_length=255)
    x_wsi = models.IntegerField()
    y_wsi = models.IntegerField()
    x_thumb = models.FloatField()
    y_thumb = models.FloatField()
    crop_size = models.IntegerField(default=128)
    nuc_count = models.IntegerField(default=0)
    focus_score = models.FloatField(default=0.0)
    edge_density = models.FloatField(default=0.0)
    cyto_frac = models.FloatField(default=0.0)
    candidate_type = models.CharField(max_length=50, choices=CANDIDATE_TYPES)

    def __str__(self):
        return f"{self.crop_name} ({self.candidate_type})"

class Prediction(models.Model):
    candidate = models.OneToOneField(CandidateCell, on_delete=models.CASCADE, related_name='prediction')
    pred_label = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    probabilities = models.JSONField(default=dict)

    def __str__(self):
        return f"Prediction: {self.pred_label} ({self.confidence:.2%})"

class GeneratedFigure(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name='figures')
    figure_type = models.CharField(max_length=50) # thumbnail_wsi, mask_foreground, mask_clean, overlay_rois, etc.
    image_path = models.CharField(max_length=1024) # relative to media/ or absolute

    def __str__(self):
        return f"Figure {self.figure_type} for Run {self.run.run_id}"
