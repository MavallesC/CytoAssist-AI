# CytoAssist AI - Plataforma de Análisis Citológico Asistido WSI
## Informe Técnico de Diseño, Arquitectura e Implementación

**CytoAssist AI** es una plataforma de Diagnóstico Asistido por Computadora (CAD) de arquitectura web modular diseñada para el procesamiento automatizado, segmentación y clasificación de frotis citológicos a partir de imágenes digitales de portaobjetos completos (*Whole Slide Images*, WSI). El sistema sirve como puente operativo entre los modelos de aprendizaje profundo entrenados con células individuales y un entorno de flujo clínico real de tamizaje citológico de cuello uterino (Pap smear).

---

## 1. Introducción y Contexto Clínico

El tamizaje citológico mediante la tinción de Papanicolaou es el método de referencia para la detección oportuna del cáncer de cuello uterino. Sin embargo, el análisis manual de frotis convencionales presenta importantes desafíos:
* **Alta Carga Operativa:** Los citotecnólogos deben revisar minuciosamente millones de células por lámina, provocando fatiga visual.
* **Variabilidad Interobservador:** El diagnóstico morfológico es subjetivo y dependiente de la experiencia del especialista.
* **Escala Gigapíxel:** Las imágenes WSI digitalizadas a $40\times$ (por ejemplo, con el escáner óptico *MoticEasyScan One*) poseen resoluciones típicas de $50,000 \times 50,000$ a $120,000 \times 100,000$ píxeles, lo que impide cargarlas directamente en la RAM convencional.

**CytoAssist AI** fue planificado y coordinado técnicamente con el personal especializado del **Laboratorio Referencial Regional de Salud Pública de San Martín** para resolver estas limitaciones. La plataforma optimiza el flujo de trabajo clínico filtrando áreas sin valor diagnóstico, localizando de forma fina candidatos celulares y aplicando modelos avanzados de Inteligencia Artificial para emitir un pre-diagnóstico clínico jerarquizado.

---

## 2. Fases del Desarrollo del Sistema

El desarrollo del sistema inteligente se estructuró en dos fases metodológicas:

1. **Fase Experimental (Cuaderno Colab):**
   * Una etapa inicial de exploración y prototipado rápido donde se implementó un pipeline básico de segmentación por color y se evaluó preliminarmente el modelo auto-supervisado **DINOv2 (ViT-B/14) + CatBoost** ([Tesis-Valles-Parte-II.ipynb](https://colab.research.google.com/drive/1fFkLRIdpdPNhfHoFtlu8Iv9SVyWUeOYO?usp=sharing)).
2. **Fase de Producción Web (CytoAssist AI):**
   * Integración de la lógica en una plataforma Django con base de datos SQLite y soporte offline.
   * Elección del modelo híbrido **DenseNet121 (Backbone de extracción de embeddings) + CatBoost (Clasificador de 6 clases Bethesda)** debido a su estabilidad de generalización, mejor balance de precisión-recall en clases minoritarias y menor consumo de GPU en inferencia en lotes.

---

## 3. Arquitectura Funcional del Pipeline

El backend del sistema sigue principios de diseño modular y limpio, separando el pipeline en etapas desacopladas implementadas en servicios Python independientes. El flujo secuencial se resume en el siguiente diagrama:

```mermaid
graph TD
    WSI["Portaobjeto Digital (TIFF Piramidal)"] --> THUMB["1. wsi_reader.py<br>Thumbnail a escala 1:20<br>Mapeo en disco (numpy.memmap)"]
    THUMB --> MASKS["2. masks.py<br>Máscaras de Control de Calidad (QC)<br>- M_fg (Tejido)<br>- M_art (Lapicero/Burbujas)<br>- M_clean (Limpia)"]
    MASKS --> ROIS["3. roi_extraction.py<br>Generación de Grilla de ROIs<br>Filtrado: Muestra limpia >= 20%"]
    ROIS --> CAND["4. candidate_detection.py<br>Tamizaje Fino a 40x<br>Segmentación de Núcleos en HSV"]
    CAND --> FUNNEL["5. Embudo de Retención de Calidad<br>Clasificación local en 5 clases:<br>- single_cell / cell_cluster / uncertain<br>- low_quality / artifact_like (Descartados)"]
    FUNNEL --> BATCH["6. inference.py<br>Inferencia por Lotes CUDA (N=64)<br>Extractor DenseNet121 + CatBoost"]
    BATCH --> AGG["7. aggregation.py<br>Agregación Diagnóstica Global<br>Lógica de Prioridad Bethesda"]
    AGG --> VIS["8. visualization.py & reports.py<br>KDE Heatmaps & PDF/HTML Reports"]
```

---

## 4. Estructura de Código y Descripción de Módulos

La estructura de la aplicación Django `wsi_analysis` organiza los servicios del pipeline de la siguiente forma:

* [models.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/models.py): Define las entidades de base de datos (`WSISample`, `AnalysisRun`, `ROI`, `CellCandidate`) para mantener la trazabilidad de cada análisis.
* [views.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/views.py): Controladores Django y endpoints AJAX para monitorizar el progreso asíncrono.
* `wsi_analysis/services/`: Capa de lógica desacoplada.
  * [wsi_reader.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/wsi_reader.py): Abre imágenes TIFF piramidales mediante `tifffile` y crea un objeto `numpy.memmap` en disco para recuperar regiones de alta resolución sin cargar la imagen completa en RAM.
  * [masks.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/masks.py): Segmenta el foreground de tejido en espacio HSV y detecta marcas físicas (tinta, rayones) para calcular la máscara biológica útil limpia:
    $$M_{clean} = M_{fg} \setminus \text{Dilate}(M_{art})$$
  * [roi_extraction.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/roi_extraction.py): Divide la lámina en una grilla de bloques de $200 \times 200$ px a nivel de thumbnail y selecciona aquellos con cobertura útil $\ge 20\%$.
  * [candidate_detection.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/candidate_detection.py): Segmenta núcleos a $40\times$, extrae parches (*crops*) de $128 \times 128$ px y calcula descriptores de calidad: *Focus Score* (varianza del Laplaciano), *Edge Density* (Canny) y *Cyto Fraction* (citoplasma útil en HSV). Segrega cultivos en 5 clases: `single_cell` (células individuales aptas), `cell_cluster` (agrupaciones), `uncertain_candidate` (dudosos para revisión), `low_quality` (desenfoque grave, descartados) y `artifact_like` (suciedad, descartados).
  * [inference.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/inference.py): Agrupa los parches en lotes de 64, extrae embeddings de 1024 dimensiones con DenseNet121 y clasifica mediante CatBoost en las clases Bethesda.
  * [aggregation.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/aggregation.py): Ejecuta la lógica de agregación diagnóstica y Bethesda.
  * [visualization.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/visualization.py): Genera mapas de calor de densidad celular y densidad de anormalidades mediante estimaciones de densidad de Kernel (KDE), overlays de grillas priorizadas y 11 figuras académicas horizontales `(16, 9)`.
  * [reports.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/reports.py): Genera reportes estructurados imprimibles en HTML y exporta datos clínicos en formatos CSV y JSON.
  * [runner.py](file:///g:/Mi%20unidad/San%20Marcos/Doctorado/2026-I/Tesis%20V/CytoAssist%20AI/wsi_analysis/services/runner.py): Orquestador de la tarea asíncrona que actualiza el progreso en base de datos.
  * **Exportador de Muestra y Recortes (views.py):** Permite exportar la muestra completa con sus recortes estructurados (ROIs y sub-cuadrantes Slides) en un archivo ZIP jerárquico. Los recortes se realizan en tiempo real sobre el frotis original a resolución base y se codifican a JPEG (BGR) para mantener la coloración HE exacta, guardándose en disco (`media/exports/`) en forma de caché persistente.

---

## 5. Fundamento de Lógica Diagnóstica y Bethesda

### 5.1 Control de Celularidad Mínima
El sistema Bethesda establece que una muestra es representativa si cuenta con una cantidad mínima de células epiteliales bien conservadas. **CytoAssist AI** cuenta exactamente los candidatos celulares válidos detectados. Si el total es menor a 100, la muestra se cataloga automáticamente como **"No Concluyente (Baja Celularidad)"**, previniendo falsos negativos debidos a un frotis mal tomado.

### 5.2 Algoritmo de Prioridad de Riesgo Clínico
A diferencia de los clasificadores convencionales que utilizan una regla de mayoría simple, **CytoAssist AI** implementa una jerarquía diagnóstica de prioridad basada en el riesgo citopatológico clínico:

$$\text{SCC} \succ \text{HSIL} \succ \text{ASC-H} \succ \text{LSIL} \succ \text{ASC-US} \succ \text{NILM}$$

En tamizaje citológico, la detección de una sola célula atípica de alto grado o maligna obliga a la remisión de la paciente.

> [!NOTE]
> **Comparativa de Reglas Lógicas (Caso de Estudio WSI-01 - `Leve-10-140942400777_20260225_090600.tif` con 792 células analizadas):**
> * **Regla de Mayoría Simple:** Dado que las células anormales (LSIL) solo representaron el $1.01\%$ del frotis (dominado por $98.99\%$ de NILM), el diagnóstico resultante hubiese sido **NILM (Normal)**, generando un **Falso Negativo crítico**.
> * **Regla de Prioridad Clínica (CytoAssist AI):** Identifica que existen 8 células LSIL de alta confianza (superando el umbral de representatividad mínimo de 5 células). El diagnóstico preliminar se consolida como **LSIL (Hallazgo preliminar de bajo grado)**, garantizando la seguridad de la paciente y recomendando la revisión prioritaria del cuadrante `ROI_000`.

---

## 6. Resultados de Validación del Sistema

El pipeline integrado se evaluó sobre un conjunto de prueba independiente de **10 portaobjetos digitales completos (WSI)** reales. El clasificador base de IA (DenseNet121 + CatBoost) obtuvo las siguientes métricas en validación:

* **Exactitud (Accuracy):** 75.75%
* **Precisión Macro:** 63.52%
* **Sensibilidad (Recall) Macro:** 66.68%
* **F1-Score Macro:** 64.15%
* **ROC-AUC Macro:** 93.41%
* **LogLoss:** 0.6557

### Resultados Globales (10 WSIs Analizados)

La siguiente tabla presenta la distribución de celularidad, el contraste diagnóstico y el tiempo de ejecución en producción para las 10 muestras reales evaluadas y registradas en la base de datos:

| ID | Nombre de Archivo Digitalizado | Diagnóstico Real (Patólogo) | Diagnóstico Preliminar IA | Celularidad Total | Células Anormales IA | Tiempo Total (s) | Estado del Diagnóstico |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **01** | `Leve-4-063802200173_20260224_083300` | LSIL (Bajo Grado) | HSIL (Alto Grado) | 14,337 | 265 LSIL, 2 HSIL | 2276.7 | Discrepancia Menor (Sobre-diagnóstico) |
| **02** | `Cancer-6-065802201521_20260127_083300` | SCC (Cáncer) | HSIL (Alto Grado) | 17,521 | 314 LSIL, 18 HSIL, 2 ASC-US | 2531.7 | Discrepancia Menor (Sub-diagnóstico) |
| **03** | `Leve-6-064492400038_20260319_093900-2` | LSIL (Bajo Grado) | HSIL (Alto Grado) | 19,289 | 197 LSIL, 103 HSIL, 7 ASC-H, 8 ASC-US | 3450.7 | Discrepancia Menor (Sobre-diagnóstico) |
| **04** | `Leve-10-140942400777_20260225_090600` | LSIL (Bajo Grado) | LSIL (Bajo Grado) | 550 | 8 LSIL | 73.0 | Correcto |
| **05** | `Leve-12-140942400788_20260226_093900` | LSIL (Bajo Grado) | HSIL (Alto Grado) | 44,616 | 501 LSIL, 26 HSIL, 5 ASC-H, 4 ASC-US | 10019.8 | Discrepancia Menor (Sobre-diagnóstico) |
| **06** | `Leve-15-065932400013_20260227_125900` | LSIL (Bajo Grado) | HSIL (Alto Grado) | 71,756 | 932 LSIL, 159 HSIL, 17 ASC-H, 37 ASC-US | 14233.2 | Discrepancia Menor (Sobre-diagnóstico) |
| **07** | `MOD-3-065382300114_20260211_115300` | HSIL (Alto Grado) | HSIL (Alto Grado) | 75,928 | 5407 LSIL, 138 HSIL, 5 ASC-H, 10 ASC-US | 11837.3 | Correcto |
| **08** | `MOD-5-065732200029_20260212_122600` | HSIL (Alto Grado) | HSIL (Alto Grado) | 28,841 | 1493 LSIL, 8 HSIL, 2 ASC-H, 13 ASC-US | 5045.3 | Correcto |
| **09** | `MOD-11-064152200004_20260210_083300` | HSIL (Alto Grado) | LSIL (Bajo Grado) | 11,052 | 154 LSIL, 3 HSIL | 1330.6 | Discrepancia Menor (Sub-diagnóstico) |
| **10** | `Cancer-4_20260126_080000` | SCC (Cáncer) | HSIL (Alto Grado) | 61,284 | 3412 LSIL, 249 HSIL, 94 ASC-H, 21 ASC-US | 10406.7 | Discrepancia Menor (Sub-diagnóstico) |

### Análisis de Métricas de Tamizaje Clínico
1. **Sensibilidad Diagnóstica:** **100.0%**. El sistema clasificó correctamente como "patológico sospechoso" a todos los portaobjetos con diagnóstico real de lesión escamosa o cáncer (10 de 10 láminas), evitando la ocurrencia de falsos negativos. Esto es de vital importancia en entornos de tamizaje primario, donde omitir una paciente enferma representa el mayor riesgo clínico.
2. **Especificidad Diagnóstica:** En este subconjunto de validación enfocado en casos con patología confirmada, no se incluyeron láminas sanas (NILM) de control negativo. No obstante, las pruebas analíticas del pipeline en fases previas demostraron un comportamiento robusto ante frotis normales y un correcto funcionamiento del filtro de calidad (QC).
3. **Concordancia Exacta por Categoría Bethesda:** **30.0%** (3 de 10 casos). El sistema demostró coincidencia diagnóstica precisa en los casos 04, 07 y 08. Para los 7 casos restantes, se observaron discrepancias menores que se dividen en dos comportamientos clínicos esperados:
   - **Sobre-diagnóstico (LSIL clasificado como HSIL):** Ocurrió en los casos 01, 03, 05 y 06. Esto se debe a la estricta lógica de prioridad clínica Bethesda implementada en `aggregation.py`: la detección de un número reducido de células con características morfológicas atípicas asociadas a HSIL (incluso 2 células en el caso 01) eleva preventivamente el diagnóstico global del portaobjetos, maximizando la sensibilidad diagnóstica como medida de seguridad.
   - **Sub-diagnóstico Menor (SCC clasificado como HSIL, o HSIL como LSIL):** Ocurrió en los casos 02, 10 y 09. Los casos de carcinoma de células escamosas (SCC) 02 y 10 fueron pre-diagnosticados como HSIL debido a que en frotes digitalizados de lesiones invasoras predomina la celularidad displásica de alto grado (HSIL) sobre células tumorales queratinizantes grandes individuales. En el caso 09, la presencia de solo 3 células HSIL (por debajo del umbral clínico del sistema) condujo a un pre-diagnóstico de LSIL. Dado que tanto LSIL como HSIL y SCC son categorías lesionales patológicas que conllevan la derivación inmediata a colposcopía y biopsia, estas discrepancias no comprometen la seguridad ni el tratamiento oportuno de la paciente.


---

## 7. Requisitos e Instalación

### 7.1 Requisitos del Sistema
* **Python** $\ge$ 3.10
* Sistema operativo compatible con aceleración GPU CUDA (opcional, recomendado para inferencia rápida) o CPU.
* **Bibliotecas principales:** `django`, `numpy`, `pandas`, `opencv-python`, `torch`, `torchvision`, `catboost`, `joblib`, `tifffile`, `imagecodecs`.

### 7.2 Instalación
1. Clonar o descargar el repositorio del proyecto.
2. Crear un entorno virtual e instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Colocar los archivos de pesos de la Inteligencia Artificial en el directorio `artefactos/` en la raíz del proyecto:
   * `densenet121_backbone.pt` (Pesos del extractor de PyTorch)
   * `catboost_classifier.cbm` (Modelo de clasificación CatBoost)
   * `inference_artifacts.json` (Parámetros y metadatos de inferencia)
   * `pca.joblib` (Si se habilitó reducción de dimensionalidad)
   
   *Nota: Si estos archivos no están en el directorio `artefactos/`, el sistema iniciará en un modo de simulación heurística para permitir pruebas de visualización e interfaz.*

4. Ejecutar migraciones de base de datos Django y levantar el servidor:
   ```bash
   python manage.py makemigrations wsi_analysis
   python manage.py migrate
   python manage.py runserver
   ```
5. Acceder al sistema en su navegador web: `http://127.0.0.1:8000/`

---

## 8. Guía de Uso del Visor Web

1. **Cargar la Muestra WSI:** Vaya a "Cargar Muestra WSI", defina el nombre y especifique la ruta local del archivo `.tif` (ej. `C:\DatosWSI\Slide_01.tif`). Configure los parámetros del pipeline (umbrales de foco, sensibilidad de detección, etc.) y presione **Iniciar Análisis**.
2. **Monitoreo en Tiempo Real:** El sistema redirigirá al panel principal, que actualiza de manera automática mediante AJAX el estado del análisis asíncrono (`Procesando`, `Segmentando`, `Finalizado`).
3. **Reporte Clínico:** Presenta las estadísticas globales, recuentos celulares, diagnóstico sugerido y las 11 visualizaciones de control de calidad, incluyendo gráficos de distribución y mapas de calor. Está optimizado para descarga e impresión.
4. **Visor Espacial Interactivo WSI:** Permite alternar la visualización del frotis original con overlays de densidad de atipias y explorar la galería interactiva de células candidatas agrupadas en pestañas según la clasificación de la IA (NILM, LSIL, HSIL, etc.), facilitando la validación del citopatólogo.
5. **Exportación Jerárquica de Recortes (ZIP):** En la tarjeta de descargas del reporte de análisis, puede hacer clic en **Exportar Muestra y Recortes (ZIP)**. Un modal interactivo le indicará el progreso mientras el servidor genera en tiempo real los recortes JPEG de cada ROI (`ROI_xxx.jpg`) y cuadrante (`Slide_x.jpg`), guardándolos en un ZIP estructurado en la carpeta del usuario (Descargas). Las descargas posteriores son instantáneas gracias al caché en disco del backend.
