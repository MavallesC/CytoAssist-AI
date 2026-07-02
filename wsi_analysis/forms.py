from django import forms
from .models import WSISample

class WSISampleUploadForm(forms.ModelForm):
    file_path = forms.CharField(
        max_length=1024,
        required=False,
        label="Ruta local del WSI",
        help_text="Ruta absoluta al archivo .tif, .tiff o .svs en el servidor/PC (ej. C:\\Dataset\\slide.tif). Dejar en blanco si va a subir un archivo."
    )
    uploaded_file = forms.FileField(
        required=False,
        label="Subir archivo WSI",
        help_text="Seleccione un archivo WSI para subir (recomendado solo para archivos pequeños)."
    )

    class Meta:
        model = WSISample
        fields = ['name']
        labels = {
            'name': 'Nombre de la Muestra',
        }

    def clean(self):
        cleaned_data = super().clean()
        file_path = cleaned_data.get('file_path')
        uploaded_file = cleaned_data.get('uploaded_file')

        if not file_path and not uploaded_file:
            raise forms.ValidationError("Debe proporcionar una ruta local de archivo o subir un archivo WSI.")
        
        return cleaned_data

class PipelineParametersForm(forms.Form):
    tile_size = forms.IntegerField(
        initial=256,
        min_value=64,
        max_value=1024,
        label="Tamaño del Tile (scanning)",
        help_text="Tamaño de la ventana de escaneo en píxeles."
    )
    stride = forms.IntegerField(
        initial=128,
        min_value=32,
        max_value=512,
        label="Paso (stride)",
        help_text="Paso de avance en el escaneo de la imagen."
    )
    crop_size = forms.IntegerField(
        initial=128,
        min_value=32,
        max_value=256,
        label="Tamaño del Crop Celular",
        help_text="Tamaño de salida de las imágenes de candidatos celulares."
    )
    min_clean_frac = forms.FloatField(
        initial=0.20,
        min_value=0.0,
        max_value=1.0,
        label="Fracción Limpia Mínima (ROI)",
        help_text="Porcentaje mínimo de área libre de artefactos para analizar un ROI."
    )
    min_focus_score = forms.FloatField(
        initial=35.0,
        min_value=0.0,
        label="Score de Enfoque Mínimo",
        help_text="Umbral de Laplacian variance para considerar un crop nítido."
    )
    min_edge_density = forms.FloatField(
        initial=0.008,
        min_value=0.0,
        label="Densidad de Bordes Mínima",
        help_text="Densidad de bordes detectada con Canny."
    )
    min_cyto_frac = forms.FloatField(
        initial=0.03,
        min_value=0.0,
        max_value=1.0,
        label="Fracción Mínima de Citoplasma",
        help_text="Porcentaje mínimo de citoplasma en el crop celular."
    )
    min_nucleus_area = forms.IntegerField(
        initial=120,
        min_value=10,
        label="Área Mínima del Núcleo (px)",
        help_text="Área mínima para clasificar un núcleo."
    )
    max_nucleus_area = forms.IntegerField(
        initial=2500,
        min_value=100,
        label="Área Máxima del Núcleo (px)",
        help_text="Área máxima para clasificar un núcleo."
    )
    max_cells_total = forms.IntegerField(
        initial=4000,
        min_value=10,
        max_value=20000,
        label="Límite Total de Candidatos",
        help_text="Límite máximo de crops a extraer de toda la lámina."
    )
    min_conf_priority = forms.FloatField(
        initial=0.50,
        min_value=0.0,
        max_value=1.0,
        label="Confianza Mínima de Prioridad",
        help_text="Confianza mínima requerida para activar alertas de hallazgos críticos (SCC, HSIL, ASC-H)."
    )
