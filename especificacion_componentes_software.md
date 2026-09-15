# CytoAssist AI - Especificación de Componentes y Versiones de Software

**Proyecto:** CytoAssist AI - Plataforma de Análisis Citológico Asistido sobre Portaobjetos Digitales Completos (WSI)  
**Fecha:** Septiembre 2026  
**Entorno de Ejecución:** Python 3.12 (Arquitectura x64, soporte para GPU CUDA / CPU)

---

## 1. Ficha Técnica Consolidada de Componentes y Dependencias

A continuación se detallan los componentes, bibliotecas y herramientas que integran la plataforma de software con sus respectivas versiones exactas y requerimientos mínimos:

| Capa / Módulo | Componente / Librería | Versión Exacta | Versión Mínima | Función Principal en el Sistema |
| :--- | :--- | :---: | :---: | :--- |
| **Lenguaje Base** | **Python** | `3.12.x` | $\ge 3.10$ | Intérprete base y ejecución del pipeline general |
| **Framework Web** | **Django** | `5.2.17` | $\ge 5.0$ | Arquitectura backend, ORM, enrutamiento, transacciones atómicas |
| **Base de Datos** | **SQLite** | `3.x` | $\ge 3.35$ | Persistencia de muestras WSI, ROIs, recuentos celulares y diagnósticos |
| **Deep Learning** | **PyTorch (`torch`)** | `2.9.1` | $\ge 2.0$ | Extracción de representaciones latentes (embeddings 1024-d) |
| **Visión / DL** | **Torchvision** | `0.24.1` | $\ge 0.15$ | Preprocesamiento tensorial, capas convolucionales y normalización |
| **Clasificador ML** | **CatBoost (`catboost`)** | `1.2.10` | $\ge 1.1$ | Clasificación de árboles de gradiente en 6 categorías Bethesda |
| **Machine Learning** | **Scikit-Learn (`scikit-learn`)** | `1.6.1` | $\ge 1.2$ | Métricas de rendimiento (ROC-AUC, F1, Recall) y preprocesamiento |
| **Serialización** | **Joblib** | `1.5.2` | $\ge 1.2$ | Carga y serialización de objetos de apoyo y transformadores |
| **Visión Artificial** | **OpenCV (`opencv-python`)** | `4.12.0.88` | $\ge 4.7$ | Segmentación HSV, cálculo de foco (Laplaciano) y bordes Canny |
| **Lectura WSI** | **Tifffile** | `2026.6.1` | $\ge 2023.0$ | Lectura de imágenes piramidales gigapíxel y mapeo de memoria |
| **Códecs de Imagen** | **ImageCodecs** | `2026.6.6` | $\ge 2023.0$ | Decodificación de formatos comprimidos JPEG/LZW/Deflate en TIFF |
| **Manejo de Imágenes** | **Pillow (`PIL`)** | `12.0.0` | $\ge 9.5$ | Recorte celular (crops 128×128) y generación de miniaturas |
| **Patología Digital** | **OpenSlide Python** | `1.4.2` | $\ge 1.2$ | Interoperabilidad con formatos estándar de patología digital |
| **Cálculo Numérico** | **NumPy** | `2.2.6` | $\ge 1.24$ | Operaciones matriciales, máscaras binarias y arrays en memoria |
| **Estructuras Datos** | **Pandas** | `2.3.3` | $\ge 2.0$ | Estructuración tabular y consolidación de candidatos celulares |
| **Cálculo Científico** | **SciPy** | `1.16.3` | $\ge 1.10$ | Estimación de densidad por kernel (KDE) para mapas de calor |
| **Visualización** | **Matplotlib** | `3.10.6` | $\ge 3.7$ | Generación de gráficos analíticos y figuras médicas |
| **Cache / Colas** | **Redis (`redis`)** | `8.0.1` | $\ge 5.0$ | Caché de sesión y cola de mensajería asíncrona |

---

## 2. Descripción por Capas del Sistema de Software

### 2.1. Capa de Plataforma Web y Persistencia
* **Django (v5.2.17):** Proporciona la estructura modular del sistema mediante el patrón MTV (Model-Template-View). Gestiona la seguridad, el enrutamiento de peticiones, el panel de control y las transacciones atómicas de base de datos (`transaction.atomic`) para consolidar diagnósticos de forma consistente.
* **SQLite 3:** Motor de base de datos relacional ligero e integrado que almacena los estados de procesamiento (`PENDING`, `PROCESSING`, `COMPLETED`), inventario de ROIs detectadas y predicciones celulares individuales.

### 2.2. Capa de Inteligencia Artificial (Modelos y Clasificación)
* **Backbone Extractor de Características:** **DenseNet-121** pre-entrenado y ajustado sobre el dataset *CRIC Cervix Collection*, exportado como pesos PyTorch (`densenet121_backbone_final.pt`) con un vector de salida de **1024 dimensiones**.
* **Clasificador Final:** **CatBoostClassifier (v1.2.10)** guardado en formato binario CBM (`catboost_final.cbm`), configurado con 500 iteraciones, profundidad 6 y tasa de aprendizaje ~0.059.
* **Categorías Bethesda Soportadas (6 clases):**
  1. `NILM` (Negativo para lesión intraepitelial o malignidad)
  2. `ASC-US` (Células escamosas atípicas de significado indeterminado)
  3. `LSIL` (Lesión intraepitelial escamosa de bajo grado)
  4. `ASC-H` (Células escamosas atípicas, no se puede descartar alto grado)
  5. `HSIL` (Lesión intraepitelial escamosa de alto grado)
  6. `SCC` (Carcinoma de células escamosas)

### 2.3. Capa de Visión por Computador y Procesamiento WSI
* **Tifffile (v2026.6.1) + ImageCodecs (v2026.6.6):** Lectura directa de imágenes gigapíxel escaneadas por el equipo *MoticEasyScan One*, utilizando mapeo en memoria (`numpy.memmap`) para evitar el desbordamiento de memoria RAM.
* **OpenCV (v4.12.0.88):** Segmentación cromática en espacio HSV para separar tejido útil (Foreground) de artefactos (tinta/burbujas) y cálculo de nitidez mediante varianza del Laplaciano (*Focus Score*).

### 2.4. Capa Frontend y Experiencia de Usuario
* **HTML5 + CSS3 (Vanilla):** Interfaz médica moderna con diseño *dark mode*, tipografía estilizada y diseño adaptable.
* **JavaScript (ES6+):** Polling asíncrono para actualización del progreso del análisis sin recargar la página, visor dinámico de imágenes celulares clasificadas y exportador estructurado de recortes en archivo comprimido ZIP.

---

## 3. Especificación de Dependencias (`requirements.txt`)

```text
django>=5.0,<=5.2.17
numpy==2.2.6
pandas==2.3.3
scipy==1.16.3
matplotlib==3.10.6
opencv-python==4.12.0.88
Pillow==12.0.0
torch==2.9.1
torchvision==0.24.1
catboost==1.2.10
scikit-learn==1.6.1
joblib==1.5.2
tifffile==2026.6.1
imagecodecs==2026.6.6
redis==8.0.1
```
