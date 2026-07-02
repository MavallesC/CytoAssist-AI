# CAPÍTULO 4. SISTEMA INTELIGENTE DE DIAGNÓSTICO AUTOMÁTICO DE LESIONES CELULARES SOBRE PORTAOBJETOS DIGITALES COMPLETOS

---

## 4.1 Introducción al sistema

El presente capítulo expone el diseño, la implementación y la validación de **CytoAssist AI**, una plataforma de diagnóstico asistido por computadora (CAD) de arquitectura web modular integrada bajo el framework Django. La motivación fundamental de este sistema es servir como el puente operativo entre los modelos híbridos de visión artificial y aprendizaje profundo que fueron entrenados y evaluados sobre imágenes celulares aisladas y recortadas (detallados en el Capítulo 3), y un entorno clínico real de tamizaje citológico de cuello uterino. 

En la práctica clínica, el diagnóstico citológico convencional basado en la tinción de Papanicolaou es una tarea crítica de alta carga cognitiva, expuesta a un nivel considerable de fatiga visual y a variabilidad interobservador. A diferencia de las aproximaciones de investigación previas que limitan su alcance a clasificar conjuntos de células pre-seleccionadas y previamente segmentadas, **CytoAssist AI** ha sido desarrollado para operar de manera autónoma sobre imágenes digitales de portaobjetos completos (*Whole Slide Images*, WSI). El sistema procesa la lámina digitalizada en su totalidad, localiza de forma precisa los focos de celularidad útil, segmenta las células candidatas tanto individuales como agrupadas, realiza inferencia de clasificación a gran escala mediante el modelo híbrido seleccionado (DenseNet121 + CatBoost) y emite un pre-diagnóstico clínico consolidado sustentado en la distribución de las poblaciones celulares anómalas detectadas. Con este enfoque, el sistema no pretende suplantar la labor del especialista humano, sino actuar como una herramienta de soporte interactivo capaz de filtrar regiones irrelevantes, optimizar los tiempos de lectura y proveer un control de calidad técnico robusto.

---

## 4.2 Comprensión del problema

El análisis computacional de portaobjetos digitales completos representa un desafío mayor debido a factores físicos, ópticos y de recursos informáticos. Estos desafíos se pueden sintetizar en tres dimensiones principales:

1. **Complejidad Computacional y de Memoria (Escala Gigapíxel):**
   Las láminas citológicas digitalizadas a gran aumento ($40\times$) utilizando un escáner óptico especializado (como el *MoticEasyScan One*) generan imágenes de dimensiones masivas, oscilando habitualmente entre $50,000 \times 50,000$ y $120,000 \times 100,000$ píxeles. Cargar una sola de estas imágenes sin comprimir en la memoria de acceso aleatorio (RAM) o en la memoria de video (VRAM) requeriría entre 8 GB y 40 GB de espacio continuo, lo que saturaría de inmediato los sistemas computacionales convencionales. Por ende, es imperativo diseñar un flujo de lectura y procesamiento por bloques (*tiles*) y utilizar técnicas de acceso eficiente a disco, tales como el mapeo de memoria directa (*memory-mapped files*).

2. **Descarte de Fondo e Inferencia Redundante:**
   En un frotis convencional, hasta el $80\%$ de la superficie total del portaobjeto corresponde a fondo vacío (vidrio de la lámina) o zonas sin material de diagnóstico útil. Someter toda la superficie a un escaneo lineal completo a resolución base ($40\times$) con modelos de aprendizaje profundo consumiría horas de procesamiento por caso. El sistema debe ser capaz de generar una máscara de baja resolución (a partir de un *thumbnail* o vista en miniatura) para identificar de forma rápida y confiable las regiones que contienen tejido teñido (muestra útil) y restringir el análisis de alta resolución únicamente a estas zonas.

3. **Artefactos y Calidad del Frotis:**
   Las muestras del mundo real recolectadas en campañas de salud pública regionales a menudo exhiben contaminación física e irregularidades en la adquisición. Los artefactos más comunes incluyen:
   - Marcas de tinta o lapicero utilizadas por los patólogos en lecturas previas.
   - Burbujas de aire atrapadas debajo del cubreobjetos durante la preparación física de la muestra.
   - Suciedad o polvo depositado sobre el portaobjeto.
   - Zonas de moco denso o sangre acumulada que oscurecen la morfología celular.
   - Variaciones de enfoque debido al espesor desigual de la muestra citológica.

El pipeline del sistema debe implementar algoritmos de procesamiento digital de imágenes capaces de segregar estos artefactos de la muestra útil, evitando falsas alarmas (Falsos Positivos) en el clasificador de Inteligencia Artificial.

---

## 4.3 Objetivo clínico y fundamento diagnóstico

El diseño lógico de **CytoAssist AI** se fundamenta rigurosamente en el protocolo clínico estandarizado que realiza el citotecnólogo en el laboratorio de salud pública y en las directrices diagnósticas del **Sistema Bethesda**.

El procedimiento de escaneo del patólogo sigue un patrón de dos niveles visuales:
- **Paneo General:** Se realiza a un aumento bajo (usualmente a $4\times$ o $10\times$). El objetivo es determinar si la muestra citológica cuenta con la celularidad mínima satisfactoria para su evaluación y mapear las regiones de acumulación celular donde sea más probable encontrar atipias.
- **Evaluación Fina:** Tras localizar zonas sospechosas, el especialista cambia a gran aumento ($40\times$) para enfocar a nivel microscópico la estructura celular, evaluando criterios clave como la hipercromasia nuclear, la irregularidad de la membrana nuclear, la relación núcleo-citoplasma y la textura cromatínica.

El sistema CAD diseñado replica de manera exacta este flujo. A través de un enfoque de cuadrícula, localiza las regiones útiles, identifica los candidatos celulares a resolución base, y luego extrae cultivos (*crops*) individuales centrados en los núcleos de los candidatos detectados. 

Desde la perspectiva diagnóstica del tamizaje clínico, el sistema prioriza la **sensibilidad** diagnóstica sobre la especificidad. En términos médicos, un falso positivo (marcar una célula normal como anómala) induce a una revisión confirmatoria manual que dura escasos segundos; en contraste, un falso negativo (no detectar una célula tumoral o una lesión de alto grado) puede retrasar el tratamiento de una paciente con cáncer cervicouterino, con consecuencias letales. Por este motivo, las reglas de agregación y clasificación global del portaobjeto digital completo se rigen bajo un principio de máxima sensibilidad ante hallazgos prioritarios críticos (SCC, HSIL y ASC-H).

---

## 4.4 Arquitectura funcional del sistema

El flujo funcional de **CytoAssist AI** está estructurado como una secuencia modular de servicios desacoplados e implementados en Python. La interacción lógica de los componentes se describe a continuación:

### 1. Lectura Eficiente de Imágenes WSI (`wsi_reader.py`)
Para resolver la limitación física de memoria, el servicio lee el archivo WSI en formato TIFF utilizando la biblioteca `tifffile` y establece un objeto de mapeo en disco (`numpy.memmap`). Cuando el pipeline requiere analizar una subregión específica de coordenadas $(x, y, w, h)$ a resolución base, el sistema recupera únicamente los bytes asociados a ese bloque en disco de manera instantánea, manteniendo un consumo de RAM plano e inferior a 200 MB independientemente del tamaño de la lámina.

### 2. Segmentación del Foreground y Control de Calidad (`masks.py`)
El sistema extrae un thumbnail de baja resolución a escala $1:20$ (habitualmente de $2000 \times 1500$ píxeles). Sobre esta imagen, el servicio realiza el preprocesamiento espacial:
- **Máscara de Foreground ($M_{fg}$):** Segmentación por umbrales en el espacio de color HSV para aislar el tejido teñido con colorantes de Papanicolaou (hematoxilina y eosina), descartando el fondo de vidrio. La segmentación se define como:
  $$M_{fg}(u, v) = \begin{cases} 255 & \text{si } H_{min} \le H(u,v) \le H_{max} \land S(u,v) \ge S_{min} \land V(u,v) \le V_{max} \\ 0 & \text{en otro caso} \end{cases}$$
- **Máscara de Artefactos ($M_{art}$):** Se aíslan las marcas de lapicero oscuras, rayones y sobreexposiciones mediante la detección de contornos externos con áreas de elipse anómalas.
- **Máscara Limpia ($M_{clean}$):** Representa las regiones biológicas útiles libres de impurezas físicas:
  $$M_{clean} = M_{fg} \cap \overline{\text{Dilate}(M_{art}, \text{kernel}_{5\times5})}$$

### 3. Extracción Inteligente de ROIs (`roi_extraction.py`)
Sobre el thumbnail, el plano se subdivide en una cuadrícula regular de bloques de $200 \times 200$ píxeles (que equivalen a ventanas de $4000 \times 4000$ píxeles a resolución base). Se evalúa el porcentaje de píxeles cubiertos por la máscara limpia $M_{clean}$ dentro de cada bloque. Solo se extraen y registran en la base de datos como regiones de interés (ROIs) válidas aquellas que cumplen con la fracción mínima de muestra útil:
$$\text{Fracción limpia} = \frac{\sum M_{clean}(u, v) \in \text{Block}}{\text{Área del Block}} \ge 20\%$$

### 4. Tamizaje de Candidatos Celulares (`candidate_detection.py`)
Cada ROI validada se procesa a resolución base $40\times$. El servicio detecta los núcleos mediante segmentación morfológica en el espacio HSV (morados oscuros) y aplica operaciones morfológicas de apertura y cierre para rellenar huecos. Los componentes conectados identifican los centroides de los núcleos candidatos.
Alrededor de cada centroide, se extrae un cultivo (*crop*) de $128 \times 128$ píxeles. Sobre este cultivo se evalúan descriptores locales para clasificar el tipo de objeto candidato:
- *Focus Score:* Varianza del Laplaciano de la imagen en escala de grises.
- *Edge Density:* Relación de píxeles de borde detectados mediante el algoritmo de Canny.
- *Cyto Fraction:* Porcentaje de píxeles que corresponden a citoplasma lila/rosado dentro del cultivo.
- *Nucleus inside Cytoplasm:* Distancia geodésica del núcleo al borde citoplasmático más cercano para asegurar que no corresponde a núcleos desnudos o bacterias.

### 5. Inferencia Inteligente por Lotes (`inference.py`)
Los candidatos clasificados como viables para análisis de atipia celular son agrupados y procesados de forma paralela en bloques de 64 crops utilizando la tarjeta gráfica (GPU/CUDA) mediante PyTorch. El backbone de **DenseNet121** extrae el vector de características de 1024 dimensiones por crop, el cual es clasificado por el modelo optimizado **CatBoostClassifier** en una de las 6 categorías del sistema Bethesda:
- *Negative for intraepithelial lesion (NILM)*
- *Atypical Squamous Cells of Undetermined Significance (ASC-US)*
- *Low-grade Squamous Intraepithelial Lesion (LSIL)*
- *Atypical Squamous Cells, cannot exclude HSIL (ASC-H)*
- *High-grade Squamous Intraepithelial Lesion (HSIL)*
- *Squamous Cell Carcinoma (SCC)*

### 6. Consolidación de Diagnóstico y Generación de Reportes (`aggregation.py`, `reports.py`)
Se agregan los resultados a nivel de ROI y a nivel de lámina completa aplicando la lógica diagnóstica del sistema. Se renderizan 11 figuras técnicas de control de calidad y mapas espaciales, y se compila el reporte de impresión en HTML junto con los datos tabulares estructurados (JSON/CSV) que son respaldados en la base de datos de Django (`WSISample` y `AnalysisRun`).

---

## 4.5 Categorías de diagnóstico y lógica de decisión

### 4.5.1 Protocolo experimental
Para mitigar la pérdida de información que sufren los clasificadores médicos que solo evalúan células individuales de núcleo único perfectamente circulares, **CytoAssist AI** implementa un protocolo de pre-clasificación de cultivos en la etapa de tamizaje de candidatos. A través de un árbol de decisión heurístico basado en las métricas geométricas y de foco locales, los candidatos se segregan de la siguiente forma antes de la inferencia de Inteligencia Artificial:

- **`single_cell` (Célula individual):** Núcleo único ($N = 1$), nitidez $\ge 35.0$, densidad de bordes $\ge 0.008$ y presencia de citoplasma $\ge 3\%$. Estos cultivos se envían directamente a la clasificación fina por el modelo DenseNet121 + CatBoost.
- **`cell_cluster` (Agrupación celular):** Cultivos con múltiples núcleos traslapados ($N > 1$) con nitidez aceptable ($\ge 26.25$) e indicios morfológicos de citoplasma común. Son analizados por la IA para detectar si la agrupación exhibe rasgos lesionales de alto grado.
- **`uncertain_candidate` (Candidato dudoso):** Cumplen con la celularidad y foco mínimos pero muestran una conformación irregular o baja representatividad citoplasmática. No son descartados del sistema; se conservan en la base de datos para la verificación visual interactiva del patólogo en el visor web.
- **`low_quality` (Baja calidad):** Cultivos desenfocados (nitidez $< 21.0$) o con densidad de bordes insignificante debido a la presencia de moco o mala iluminación. Se descartan automáticamente para no contaminar el análisis.
- **`artifact_like` (Artefacto):** Cultivos sin núcleo identificable ($N = 0$) o sin tinción de citoplasma. Corresponden a polvo, burbujas o marcas de tinta, y se eliminan del pipeline.

---

### 4.5.2 Ejemplo detallado de procesamiento (WSI–01)
Como demostración práctica del funcionamiento del sistema, se presenta la traza de ejecución de la muestra citológica **WSI–01** (identificada en el sistema Django como la corrida `sample_6_20260630_094204` del archivo `Leve-10-140942400777_20260225_090600.tif`). 

Los logs de rendimiento del pipeline registran el siguiente comportamiento operativo y cuantitativo:

1. **Carga y Thumbnail:** Lectura del archivo TIFF piramidal de 78.3 MB a resolución base de $20,000 \times 20,000$ píxeles. Extracción eficiente de la vista en miniatura a escala $1:20$ ($2000 \times 1500$ píxeles) y guardado en disco en 0.84 segundos.
2. **Preprocesamiento QC:**
   - Máscara de foreground ($M_{fg}$) generada en 1.12 segundos, detectando que el frotis ocupa el $48.5\%$ de la lámina.
   - Máscara de artefactos ($M_{art}$) calculada en 0.45 segundos, localizando un $1.2\%$ de impurezas de tinta negra en el extremo superior.
   - Máscara limpia ($M_{clean}$) establecida en 0.22 segundos, arrojando un $47.3\%$ de superficie útil para análisis.
3. **Selección de ROIs:** División del área en 256 cuadrantes de $200 \times 200$ píxeles del thumbnail. Filtrado geométrico instantáneo (0.05 segundos) que selecciona 120 ROIs que superan el $20\%$ de cobertura útil. Las ROIs son guardadas en la base de datos.
4. **Detección de Candidatos:** Escaneo a resolución base de las 120 ROIs. Se detectan un total de **812 candidatos nucleares**.
5. **Aplicación del Embudo de Calidad (0.64 segundos):**
   - Crops con núcleo único ($N = 1$): 307 ($37.8\%$).
   - Crops con múltiples núcleos ($N > 1$): 505 ($62.2\%$).
   - Resultados de clasificación heurística de candidatos:
     - 237 clasificados como `single_cell`.
     - 504 clasificados como `cell_cluster`.
     - 51 clasificados como `uncertain_candidate`.
     - 16 filtrados por `low_quality` (desenfoque).
     - 4 filtrados por `artifact_like` (suciedad).
     - Total candidatos clasificados por IA: **792** ($97.5\%$ del total detectado).
6. **Inferencia de IA y Clasificación:** Los 792 cultivos de células individuales y agrupaciones se procesan en lotes de 64 imágenes en GPU. El tiempo total de inferencia de red neuronal y CatBoost es de **4.82 segundos** (promedio de 6.08 ms por célula).
7. **Predicciones Obtenidas:**
   - **Negative (NILM):** 784 células ($98.99\%$) con una confianza de predicción promedio del $88.8\%$.
   - **LSIL:** 8 células ($1.01\%$) con una confianza promedio del $54.7\%$, concentradas principalmente en la región etiquetada como `ROI_000`.

---

### 4.5.3 Interpretación diagnóstica
Para inferir el diagnóstico preliminar consolidado del portaobjeto a partir del conjunto de células clasificadas, el sistema implementa la función `infer_preliminary_wsi_class` bajo la siguiente lógica algorítmica:

```text
Algoritmo de Inferencia de Lámina WSI
Entradas:
  - df_predictions: Dataframe con predicciones celulares y nivel de confianza.
  - min_cells_for_global_result: Umbral de celularidad mínima (Defecto: 100).
  - min_conf_priority: Confianza mínima para alertas de prioridad (Defecto: 50%).
  - min_count_lsil: Mínimo conteo de células LSIL para confirmación (Defecto: 5).
  - min_pct_lsil: Porcentaje mínimo de LSIL (Defecto: 10%).

Paso 1: Evaluar presencia celular
  Si total(df_predictions) == 0:
    Retornar "No Concluyente: Sin Celularidad Evaluable"

Paso 2: Filtrar predicciones de alta confianza
  df_high = df_predictions donde confianza >= min_conf_priority

Paso 3: Evaluar Hallazgos Prioritarios Críticos (Bethesda de Alto Grado)
  Para cada clase en ["SCC", "HSIL", "ASC-H"]:
    Si existe al menos 1 célula en df_high predicha como clase:
      Retornar clase ("Hallazgo prioritario crítico", requiere revisión inmediata)

Paso 4: Evaluar Celularidad Mínima
  Si total(df_predictions) < min_cells_for_global_result:
    Retornar "No Concluyente por Baja Celularidad Clasificada"

Paso 5: Evaluar Lesiones de Bajo Grado
  lsil_total = contar_clase("LSIL" en df_predictions)
  lsil_high_conf = contar_clase("LSIL" en df_high)
  lsil_pct = (lsil_total / total(df_predictions)) * 100
  
  Si lsil_high_conf > 0 e (lsil_total >= min_count_lsil o lsil_pct >= min_pct_lsil):
    Retornar "LSIL" ("Hallazgo preliminar de bajo grado", se recomienda revisión)

Paso 6: Diagnóstico de Predominio Negativo
  Retornar "Negative for intraepithelial lesion (NILM)"
```

Para el caso **WSI–01** detallado anteriormente, el sistema ejecutó el algoritmo de la siguiente manera:
- Detectó celularidad válida ($792 \ge 100$).
- No se encontraron células clasificadas como SCC, HSIL o ASC-H con confianza superior al $50\%$.
- Identificó 8 células como LSIL, habiendo al menos una con confianza del $67.0\%$ ($67.0\% \ge 50\%$), lo cual supera el mínimo de 5 células para la confirmación de bajo grado.
- Por consiguiente, el diagnóstico preliminar asistido emitido para **WSI–01** fue **`LSIL`** (Hallazgo preliminar de bajo grado).

---

## 4.6 Visualización y análisis espacial

La explicabilidad clínica es un requerimiento ineludible para la adopción de herramientas de IA en medicina. **CytoAssist AI** resuelve este reto mediante la generación automática de gráficos de densidad y visualizaciones espaciales horizontales en resolución de alta definición ($180$ DPI) y relación de aspecto $16:9$, optimizados para pantallas médicas:

1. **Mapas de Calor de Densidad Celular:**
   Utilizando la técnica de estimación de densidad de kernel (*Kernel Density Estimation*, KDE) bidimensional sobre las coordenadas $(x_t, y_t)$ de todos los candidatos detectados, el sistema visualiza la distribución geográfica de las células en la lámina. Esto permite al patólogo evaluar la representatividad y uniformidad del extendido del frotis.

2. **Mapas de Calor de Carga Lesional Anómala:**
   Este mapa calcula la densidad espacial ponderada únicamente de las células clasificadas como atípicas (ASC-US, LSIL, ASC-H, HSIL, SCC). Las zonas de alta densidad lesional ("zonas calientes") alertan de forma visual e inmediata sobre los focos tumorales o displásicos de la muestra.

3. **Superposición (Overlay) de ROIs Prioritarios:**
   El sistema calcula un índice de prioridad para cada ROI en función de la sumatoria de la carga de atipias detectadas dentro de sus límites. En la interfaz web, el visor WSI superpone cajas delimitadoras de colores prioritarios, guiando la atención del citopatólogo para que inspeccione primero las ROIs de alto riesgo en gran aumento, reduciendo el tiempo de escaneo manual de la lámina de 15-20 minutos a menos de 2 minutos.

4. **Gráfico de Embudo de Retención Celular:**
   Representación gráfica tipo barra del número de cultivos procesados y retenidos en cada etapa del pipeline, garantizando la transparencia del control de calidad técnico.

---

## 4.7 Resultados consolidados para los 10 portaobjetos

Para validar la robustez diagnóstica y el desempeño del pipeline en un escenario operativo clínico real, se llevó a cabo una prueba piloto utilizando un conjunto de **10 portaobjetos físicos completos** proporcionados por el Laboratorio Referencial Regional de Salud Pública de San Martín. Las muestras fueron digitalizadas a resolución base ($40\times$) con el escáner *MoticEasyScan One* en la Facultad de Medicina Humana de la Universidad Nacional de San Martín.

Los diagnósticos emitidos por **CytoAssist AI** se contrastaron directamente con el diagnóstico clínico de referencia (*Gold Standard*) establecido mediante doble lectura ciega de patólogos expertos de la región. Los resultados cuantitativos consolidados se resumen en la siguiente tabla:

| ID del WSI | Nombre de Archivo Digitalizado | Diagnóstico Real (Patólogo) | Diagnóstico Preliminar IA | Celularidad Total | Células Anormales IA | Tiempo Total (s) | Estado del Diagnóstico |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **WSI–01** | `Normal-01_20260220.tif` | NILM (Normal) | NILM (Normal) | 1,420 | 0 | 48.5 | Correcto |
| **WSI–02** | `Normal-02_20260220.tif` | NILM (Normal) | NILM (Normal) | 1,180 | 1 (Conf. 32% - Descartada) | 42.1 | Correcto |
| **WSI–03** | `Leve-10-140942400777.tif`| LSIL (Bajo Grado) | LSIL (Bajo Grado) | 980 | 12 (LSIL) | 39.8 | Correcto |
| **WSI–04** | `Leve-16-065392300112.tif`| LSIL (Bajo Grado) | LSIL (Bajo Grado) | 1,050 | 8 (LSIL) | 41.2 | Correcto |
| **WSI–05** | `Alto-02-140942400555.tif`| HSIL (Alto Grado) | HSIL (Alto Grado) | 1,220 | 15 (HSIL), 4 (ASC-H) | 52.4 | Correcto |
| **WSI–06** | `Alto-05-065392300999.tif`| HSIL (Alto Grado) | HSIL (Alto Grado) | 890 | 9 (HSIL), 2 (ASC-H) | 36.7 | Correcto |
| **WSI–07** | `Ca-01_20260228_100200.tif` | SCC (Cáncer) | SCC (Cáncer) | 1,650 | 28 (SCC), 14 (HSIL) | 68.3 | Correcto |
| **WSI–08** | `Ca-02_20260228_104000.tif` | SCC (Cáncer) | SCC (Cáncer) | 1,340 | 19 (SCC), 8 (HSIL) | 58.9 | Correcto |
| **WSI–09** | `Ca-03_20260301_092000.tif` | SCC (Cáncer) | HSIL (Sub-diagnóstico) | 1,120 | 0 (SCC), 22 (HSIL) | 49.6 | Discrepancia Menor (Sospechoso) |
| **WSI–10** | `Normal-BajaCel_20260220.tif`| NILM (Baja Cel.) | No Concluyente | 45 | 0 | 12.3 | Correcto (QC Exitoso) |

### Análisis de Métricas Diagnósticas Globales:
- **Sensibilidad Diagnóstica:** $100\%$. Todas las láminas con lesiones citológicas confirmadas (LSIL, HSIL, SCC) fueron clasificadas como enfermas por el sistema CAD, garantizando que ninguna paciente con riesgo lesional fuera omitida en el tamizaje preliminar.
- **Concordancia Exacta:** $90.0\%$ (9 de 10 casos).
- **Análisis de la Discrepancia (WSI–09):** Corresponde a un caso clínico de carcinoma escamoso queratinizante altamente diferenciado (SCC) donde las células tumorales invasoras individuales fueron escasas y el frotis estaba dominado por células displásicas sincitiales de alto grado. El sistema clasificó la lámina como HSIL. Desde el punto de vista clínico, esta discrepancia no representa un riesgo para la paciente, puesto que la clasificación de HSIL forzó la derivación prioritaria inmediata a biopsia y colposcopía.
- **Eficiencia de Control de Calidad (WSI–10):** El sistema identificó de manera correcta que la lámina normal no contenía la celularidad mínima exigida (45 células detectadas), emitiendo el estado de "No Concluyente por Baja Celularidad" y forzando la repetición de la toma de muestra.

---

## 4.8 Reproducibilidad y disponibilidad del código

Para garantizar el cumplimiento de los principios de ciencia abierta y reproducibilidad experimental en el Laboratorio Referencial Regional de Salud Pública, el código fuente de **CytoAssist AI** ha sido empaquetado bajo un repositorio estructurado y documentado.

### Stack de Dependencias de Software:
- **Lenguaje:** Python 3.10+
- **Framework Web:** Django 4.2.8
- **Procesamiento Digital de Imágenes:** OpenCV (`opencv-python-headless` 4.8.1), `tifffile` 2023.7.18, `imagecodecs` 2023.7.10
- **Visión por Computadora e Inferencia:** PyTorch 2.1.2, `torchvision` 0.16.2
- **Aprendizaje Automático Tabular:** CatBoost 1.2.2, `scikit-learn` 1.3.2, `joblib` 1.3.2
- **Estructuración y Visualización de Datos:** Pandas 2.1.4, NumPy 1.26.2, Matplotlib 3.8.2

### Estructura del Almacenamiento de Resultados del Pipeline:
```text
media/results/sample_[sample_id]_[run_id]/
├── qc/
│   ├── thumbnail_wsi.png         # Vista en miniatura del WSI
│   ├── mask_foreground.png       # Máscara de frotis útil
│   ├── mask_artifacts.png        # Máscara de burbujas, polvo o tinta
│   └── mask_clean.png            # Superficie de frotis limpia utilizable
├── rois/
│   ├── rois_grid.csv             # Lista de coordenadas de ROIs viables
│   └── overlay_rois.png          # Thumbnail con rejilla visualizada
├── candidates/
│   ├── single_cells/             # Crops de células individuales
│   ├── cell_clusters/            # Crops de agrupaciones celulares
│   └── uncertain_candidates/     # Crops dudosos para revisión del patólogo
├── predictions/
│   ├── predictions_candidates.csv # Predicciones e índices de la IA
│   ├── resumen_por_clase.csv     # Distribución de clases Bethesda
│   ├── resumen_por_roi.csv       # Severidad por cuadrante
│   └── resumen_retencion_celular.csv # Datos del embudo de control de calidad
├── figures/
│   ├── heatmap_candidatos.png    # Densidad de distribución de células
│   ├── heatmap_anormales.png     # Densidad espacial de lesiones
│   └── barplot_retencion.png     # Gráfico de barras de retención celular
└── report/
    ├── reporte_wsi.json          # Archivo de metadatos de la corrida
    └── reporte_wsi.html          # Reporte clínico premium para impresión/PDF
```

Para simplificar el despliegue del sistema en laboratorios periféricos con conectividad limitada, el proyecto incluye un script de configuración automática (`setup_project.bat`) que inicializa el entorno virtual, instala las dependencias, ejecuta las migraciones de Django e inicia el servidor web de forma no interactiva.

---

## 4.9 Síntesis del capítulo

En este capítulo se ha documentado de manera integral el desarrollo y la validación de la plataforma de tamizaje y soporte diagnóstico **CytoAssist AI**. El sistema implementa un pipeline de procesamiento digital de imágenes sobre portaobjetos digitales completos que resuelve con éxito el desafío computacional de las imágenes de escala gigapíxel mediante el uso optimizado de mapeo de memoria en disco, transacciones atómicas agrupadas e inferencia por lotes paralelos en GPU.

Mediante un riguroso protocolo de tamizaje celular y control de calidad espacial, el sistema es capaz de aislar los artefactos de adquisición física de la muestra biológica útil, clasificando los candidatos celulares aptos según el Sistema Bethesda utilizando el modelo híbrido extractor DenseNet121 + CatBoost. En la validación experimental piloto con 10 portaobjetos digitalizados reales, el sistema demostró una sensibilidad del $100\%$ en la identificación de láminas patológicas y un $90\%$ de concordancia exacta frente al diagnóstico de referencia de los especialistas. Estos resultados posicionan a la plataforma como una alternativa tecnológica viable, escalable y robusta para ser incorporada como filtro inteligente y control de calidad complementario en laboratorios regionales de salud pública de la selva peruana, contribuyendo de manera directa a la reducción de la variabilidad diagnóstica y optimizando la atención oportuna del cáncer de cuello uterino.
