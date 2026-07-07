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
        required=False,
        initial=4000,
        min_value=10,
        max_value=1000000,
        label="Límite Total de Candidatos",
        help_text="Límite máximo de crops a extraer de toda la lámina."
    )
    no_cell_limit = forms.BooleanField(
        required=False,
        initial=False,
        label="Sin límite de candidatos",
        help_text="Desactiva el límite de candidatos para escanear y clasificar todas las células de la muestra (puede tardar considerablemente más tiempo)."
    )
    slide_width = forms.IntegerField(
        initial=1376,
        min_value=256,
        max_value=4096,
        label="Ancho del Slide (WSI)",
        help_text="Ancho de cada recuadro interno (slide) en píxeles WSI."
    )
    slide_height = forms.IntegerField(
        initial=1020,
        min_value=256,
        max_value=4096,
        label="Alto del Slide (WSI)",
        help_text="Alto de cada recuadro interno (slide) en píxeles WSI."
    )
    min_conf_priority = forms.FloatField(
        initial=0.50,
        min_value=0.0,
        max_value=1.0,
        label="Confianza Mínima de Prioridad",
        help_text="Confianza mínima requerida para activar alertas de hallazgos críticos (SCC, HSIL, ASC-H)."
    )

    def clean(self):
        cleaned_data = super().clean()
        no_cell_limit = cleaned_data.get('no_cell_limit')
        max_cells_total = cleaned_data.get('max_cells_total')

        if no_cell_limit:
            cleaned_data['max_cells_total'] = 500000
        elif max_cells_total is None:
            self.add_error('max_cells_total', "Debe ingresar un límite de candidatos si no activa la opción sin límite.")
            
        return cleaned_data
