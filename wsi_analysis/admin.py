from django.contrib import admin
from .models import WSISample, AnalysisRun, ROI, Slide, CandidateCell, Prediction, GeneratedFigure

admin.site.register(WSISample)
admin.site.register(AnalysisRun)
admin.site.register(ROI)
admin.site.register(Slide)
admin.site.register(CandidateCell)
admin.site.register(Prediction)
admin.site.register(GeneratedFigure)
