import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

doc = Document()

# Page setup
section = doc.sections[0]
section.top_margin = Inches(0.8)
section.bottom_margin = Inches(0.8)
section.left_margin = Inches(0.8)
section.right_margin = Inches(0.8)

# Styles
PRIMARY_COLOR = RGBColor(15, 76, 129)      # #0F4C81 (Classic Blue / Medical)
SECONDARY_COLOR = RGBColor(51, 65, 85)     # Slate 700
TEXT_DARK = RGBColor(30, 41, 59)          # Slate 800
MUTED_COLOR = RGBColor(100, 116, 139)      # Slate 500

# Document Title
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after = Pt(4)
title_run = title_p.add_run("CytoAssist AI")
title_run.font.name = "Arial"
title_run.font.size = Pt(22)
title_run.font.bold = True
title_run.font.color.rgb = PRIMARY_COLOR

subtitle_p = doc.add_paragraph()
subtitle_p.paragraph_format.space_before = Pt(0)
subtitle_p.paragraph_format.space_after = Pt(14)
sub_run = subtitle_p.add_run("Especificación Técnica de Componentes y Versiones de Software")
sub_run.font.name = "Arial"
sub_run.font.size = Pt(14)
sub_run.font.bold = True
sub_run.font.color.rgb = SECONDARY_COLOR

# Meta info box
meta_table = doc.add_table(rows=3, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_data = [
    ("Proyecto:", "CytoAssist AI - Plataforma de Diagnóstico Citológico Asistido (WSI)"),
    ("Fecha del Documento:", "Septiembre 2026"),
    ("Entorno de Ejecución:", "Python 3.12 (x64) con soporte de Aceleración CUDA / CPU")
]
for i, (k, v) in enumerate(meta_data):
    row = meta_table.rows[i]
    cell_k = row.cells[0]
    cell_v = row.cells[1]
    cell_k.width = Inches(1.8)
    cell_v.width = Inches(5.0)
    
    pk = cell_k.paragraphs[0]
    pk.paragraph_format.space_before = Pt(2)
    pk.paragraph_format.space_after = Pt(2)
    rk = pk.add_run(k)
    rk.font.name = "Arial"
    rk.font.size = Pt(9.5)
    rk.font.bold = True
    rk.font.color.rgb = SECONDARY_COLOR
    
    pv = cell_v.paragraphs[0]
    pv.paragraph_format.space_before = Pt(2)
    pv.paragraph_format.space_after = Pt(2)
    rv = pv.add_run(v)
    rv.font.name = "Arial"
    rv.font.size = Pt(9.5)
    rv.font.color.rgb = TEXT_DARK
    
    set_cell_background(cell_k, "F1F5F9")
    set_cell_background(cell_v, "F8FAFC")
    set_cell_margins(cell_k, top=60, bottom=60, left=100, right=100)
    set_cell_margins(cell_v, top=60, bottom=60, left=100, right=100)

set_table_borders(meta_table, "CBD5E1")

doc.add_paragraph().paragraph_format.space_after = Pt(10)

# Heading 1
def add_custom_heading(text, level=1):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    r.font.name = "Arial"
    r.font.bold = True
    if level == 1:
        r.font.size = Pt(13)
        r.font.color.rgb = PRIMARY_COLOR
    elif level == 2:
        r.font.size = Pt(11)
        r.font.color.rgb = SECONDARY_COLOR
    return h

add_custom_heading("1. Ficha Técnica Consolidada de Componentes de Software", level=1)

p_intro = doc.add_paragraph()
p_intro.paragraph_format.space_after = Pt(8)
r_intro = p_intro.add_run("La siguiente tabla detalla cada uno de los paquetes, librerías y dependencias que forman parte del entorno operativo del software, indicando la versión exacta instalada y la función que desempeña en el pipeline:")
r_intro.font.name = "Arial"
r_intro.font.size = Pt(10)
r_intro.font.color.rgb = TEXT_DARK

# Table of components
components = [
    ("Lenguaje Base", "Python", "3.12.x", ">= 3.10", "Intérprete y entorno de ejecución del pipeline"),
    ("Framework Web", "Django", "5.2.17", ">= 5.0", "Backend MVC, enrutamiento, ORM y transacciones"),
    ("Base de Datos", "SQLite", "3.x", ">= 3.35", "Persistencia relacional de WSI, ROIs y recuentos"),
    ("Deep Learning", "PyTorch (torch)", "2.9.1", ">= 2.0", "Extracción de embeddings con DenseNet-121"),
    ("Visión / DL", "Torchvision", "0.24.1", ">= 0.15", "Normalización tensorial y capas convolucionales"),
    ("Clasificador ML", "CatBoost", "1.2.10", ">= 1.1", "Clasificación de 6 clases Bethesda por GBDT"),
    ("Machine Learning", "Scikit-Learn", "1.6.1", ">= 1.2", "Métricas (ROC-AUC, F1), transformadores"),
    ("Serialización", "Joblib", "1.5.2", ">= 1.2", "Persistencia de transformadores y pipelines"),
    ("Visión Artificial", "OpenCV", "4.12.0.88", ">= 4.7", "Segmentación HSV, Laplaciano de foco y Canny"),
    ("Lectura WSI", "Tifffile", "2026.6.1", ">= 2023.0", "Decodificación piramidal y memmap gigapíxel"),
    ("Códecs Imagen", "ImageCodecs", "2026.6.6", ">= 2023.0", "Decodificación rápida JPEG/LZW en TIFF"),
    ("Manejo Imagen", "Pillow (PIL)", "12.0.0", ">= 9.5", "Extracción de cultivos (128x128) y miniaturas"),
    ("Patología Digital", "OpenSlide Python", "1.4.2", ">= 1.2", "Interoperabilidad con formatos WSI estándar"),
    ("Cálculo Numérico", "NumPy", "2.2.6", ">= 1.24", "Matrices gigapíxel, arrays y máscaras binarias"),
    ("Estructuras Datos", "Pandas", "2.3.3", ">= 2.0", "Tablas y consolidación de candidatos celulares"),
    ("Cálculo Científico", "SciPy", "1.16.3", ">= 1.10", "Estimación KDE para mapas de calor"),
    ("Visualización", "Matplotlib", "3.10.6", ">= 3.7", "Generación de gráficos y mapas espaciales"),
    ("Cache / Colas", "Redis", "8.0.1", ">= 5.0", "Caché en memoria y mensajería en background")
]

table = doc.add_table(rows=len(components)+1, cols=5)
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ["Capa / Módulo", "Componente", "Versión Exacta", "Versión Mín.", "Función en el Sistema"]
col_widths = [Inches(1.2), Inches(1.3), Inches(1.0), Inches(0.9), Inches(2.4)]

# Format Header
hdr_row = table.rows[0]
for idx, text in enumerate(headers):
    cell = hdr_row.cells[idx]
    cell.width = col_widths[idx]
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in [2, 3] else WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9)
    r.font.bold = True
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_background(cell, "0F4C81")
    set_cell_margins(cell, top=80, bottom=80, left=80, right=80)

# Format Data Rows
for row_idx, data in enumerate(components, start=1):
    row = table.rows[row_idx]
    bg_color = "F8FAFC" if row_idx % 2 == 0 else "FFFFFF"
    for col_idx, val in enumerate(data):
        cell = row.cells[col_idx]
        cell.width = col_widths[col_idx]
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [2, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        if col_idx in [1, 2]:
            r.font.bold = True
        r.font.color.rgb = TEXT_DARK
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=50, bottom=50, left=70, right=70)

set_table_borders(table, "CBD5E1")

# Section 2
add_custom_heading("2. Descripción Detallada por Capas de Software", level=1)

sections_data = [
    ("2.1. Capa Web y Persistencia de Datos", [
        ("Django 5.2.17:", " Núcleo de la plataforma web. Implementa el patrón Modelo-Plantilla-Vista, sistema de autenticación, ORM para consultas estructuradas, gestión de transacciones atómicas (transaction.atomic) para garantizar consistencia diagnóstica, y enrutamiento modular."),
        ("SQLite 3:", " Motor de base de datos embebido de alto rendimiento. Almacena metadatos del WSI, coordenadas de ROIs útiles, características celulares extraídas, marcas de tiempo y el diagnóstico preliminar consolidado.")
    ]),
    ("2.2. Capa de Inteligencia Artificial y Clasificación", [
        ("Backbone Extractor de Características (DenseNet-121):", " Red convolucional profunda implementada en PyTorch 2.9.1. Recibe cultivos celulares normalizados (224x224 px) y extrae vectores densos de 1024 características semánticas."),
        ("Clasificador CatBoost (v1.2.10):", " Algoritmo de Gradient Boosted Decision Trees configurado con 500 árboles, profundidad 6 y regularización L2. Clasifica las células en 6 categorías del sistema Bethesda (NILM, ASC-US, LSIL, ASC-H, HSIL, SCC).")
    ]),
    ("2.3. Capa de Visión por Computador y Procesamiento WSI", [
        ("Tifffile 2026.6.1 & ImageCodecs 2026.6.6:", " Motor de lectura gigapíxel que permite abrir archivos TIFF piramidales provenientes del escáner MoticEasyScan One mediante mapeo en memoria (memmap), reduciendo el consumo de memoria RAM a fracciones mínimas."),
        ("OpenCV 4.12.0.88:", " Procesamiento de imágenes para segmentación en espacio de color HSV (separación de tejido vs. artefactos de tinta/burbujas) y cálculo del Focus Score mediante la varianza del operador Laplaciano.")
    ]),
    ("2.4. Capa de Frontend e Interacción Clínica", [
        ("HTML5 & CSS3 (Vanilla Dark Mode):", " Interfaz web clínica intuitiva, sin dependencias pesadas de frameworks externos, con diseño adaptable, paleta de colores de alto contraste y tipografía moderna."),
        ("JavaScript (ES6+):", " Mecanismo asíncrono para monitoreo del avance del análisis por lotes en segundo plano, visor de galería celular interactiva y módulo de descarga estructurada en archivo ZIP.")
    ])
]

for sec_title, bullets in sections_data:
    add_custom_heading(sec_title, level=2)
    for bold_part, text_part in bullets:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r_bold = p.add_run(bold_part)
        r_bold.font.name = "Arial"
        r_bold.font.size = Pt(9.5)
        r_bold.font.bold = True
        r_bold.font.color.rgb = SECONDARY_COLOR
        
        r_text = p.add_run(text_part)
        r_text.font.name = "Arial"
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = TEXT_DARK

# Section 3
add_custom_heading("3. Anexo: Archivo de Requerimientos (requirements.txt)", level=1)

p_req_intro = doc.add_paragraph()
p_req_intro.paragraph_format.space_after = Pt(4)
r_req_intro = p_req_intro.add_run("Lista para instalación directa y replicabilidad del entorno:")
r_req_intro.font.name = "Arial"
r_req_intro.font.size = Pt(9.5)

req_text = (
    "django>=5.0,<=5.2.17\n"
    "numpy==2.2.6\n"
    "pandas==2.3.3\n"
    "scipy==1.16.3\n"
    "matplotlib==3.10.6\n"
    "opencv-python==4.12.0.88\n"
    "Pillow==12.0.0\n"
    "torch==2.9.1\n"
    "torchvision==0.24.1\n"
    "catboost==1.2.10\n"
    "scikit-learn==1.6.1\n"
    "joblib==1.5.2\n"
    "tifffile==2026.6.1\n"
    "imagecodecs==2026.6.6\n"
    "redis==8.0.1"
)

p_code = doc.add_paragraph()
p_code.paragraph_format.space_before = Pt(4)
p_code.paragraph_format.space_after = Pt(12)
r_code = p_code.add_run(req_text)
r_code.font.name = "Consolas"
r_code.font.size = Pt(8.5)
r_code.font.color.rgb = RGBColor(15, 23, 42)

output_path = r"c:\Users\JhosepSF\Documents\Jhosep\Trabajos\Softwares\CytoAssist AI\Especificacion_Componentes_Software_CytoAssist_AI.docx"
doc.save(output_path)
print(f"Documento Word guardado exitosamente en: {output_path}")
