import os
import cv2
import numpy as np
import pandas as pd
from .wsi_reader import read_wsi_region

def detect_nuclei_mask(tile_rgb):
    """
    Máscara de núcleos en HSV (tonos morados/azules oscuros).
    """
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    
    # Rango morado/azul de núcleos
    mask = ((H >= 115) & (H <= 175) & (S >= 35) & (V <= 220)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def detect_cytoplasm_mask(tile_rgb):
    """
    Máscara de citoplasma en HSV (tonos rosados/lilas típicos de Papanicolaou).
    """
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    mask = (
        (
            ((H >= 145) & (H <= 179)) |   # rosado/magenta
            ((H >= 0) & (H <= 18))   |   # rosa rojizo al cruzar 179->0
            ((H >= 115) & (H <= 145))    # lila
        ) &
        (S >= 12) &
        (V >= 80)
    ).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def crop_focus_score(crop_rgb):
    """
    Métrica de nitidez (varianza del Laplaciano).
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def crop_edge_density(crop_rgb, low=40, high=120):
    """
    Densidad de bordes detectados en el crop.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    return edges, float((edges > 0).mean())

def nucleus_inside_cytoplasm(crop_nuc_mask, crop_cyto_mask, ring=6):
    """
    Verifica si el núcleo principal tiene solapamiento con el citoplasma.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (crop_nuc_mask > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return False

    # Encontrar el componente de núcleo más grande
    best_i = None
    best_area = -1
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_i = i

    if best_i is None:
        return False

    nuc_component = (labels == best_i).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    nuc_dil = cv2.dilate(nuc_component, k, iterations=1)

    overlap = ((nuc_dil > 0) & (crop_cyto_mask > 0)).mean()
    return float(overlap) > 0.01

def detect_edges_dilation_tile(tile_rgb):
    """
    Detecta bordes Canny y los dilata/cierra para el tile completo.
    """
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_d = cv2.dilate(edges, k, iterations=1)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, k, iterations=1)
    return edges_d

def nucleus_inside_cell_like_region(crop_edges_d, crop_nuc_mask):
    """
    Verifica si existen bordes de estructuras celulares alrededor del núcleo
    usando la máscara de bordes ya dilatada.
    """
    num, _, stats, centroids = cv2.connectedComponentsWithStats(
        (crop_nuc_mask > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return False

    best_i = None
    best_area = -1
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_i = i

    if best_i is None:
        return False

    cx, cy = centroids[best_i]
    cx, cy = int(round(cx)), int(round(cy))

    h, w = crop_edges_d.shape
    r = 14
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r + 1), min(h, cy + r + 1)

    local_edges = crop_edges_d[y1:y2, x1:x2]
    return float((local_edges > 0).mean()) > 0.02

def classify_candidate_type(
    nuc_count, focus_score, edge_density, cyto_frac,
    nuc_in_cyto, has_cell_like_region, params
):
    """
    Clasifica el candidato en uno de los 5 tipos requeridos.
    """
    min_focus = params.get('min_focus_score', 35.0)
    min_edge = params.get('min_edge_density', 0.008)
    min_cyto = params.get('min_cyto_frac', 0.03)

    # 1. Célula individual
    if (nuc_count == 1 and 
        focus_score >= min_focus and 
        edge_density >= min_edge and 
        cyto_frac >= min_cyto and 
        nuc_in_cyto and 
        has_cell_like_region):
        return 'single_cell'
    
    # 2. Agrupación Celular (múltiples núcleos con calidad aceptable)
    if nuc_count > 1:
        if focus_score >= min_focus * 0.75 and edge_density >= min_edge * 0.75 and cyto_frac >= min_cyto * 0.5:
            return 'cell_cluster'

    # 3. Baja Calidad
    if focus_score < min_focus * 0.6 or edge_density < min_edge * 0.5:
        return 'low_quality'

    # 4. Artefacto
    if nuc_count == 0 or cyto_frac < min_cyto * 0.2:
        return 'artifact_like'

    # 5. Candidato dudoso pero conservado para revisión
    return 'uncertain_candidate'

def extract_crops_from_tile(
    tile_rgb, tile_x, tile_y, params
):
    """
    Escanea un tile del WSI y detecta todos los candidatos celulares de forma optimizada.
    Retorna una lista de metadatos de candidatos y sus imágenes asociadas.
    """
    crop_size = params.get('crop_size', 128)
    min_nucleus_area = params.get('min_nucleus_area', 120)
    max_nucleus_area = params.get('max_nucleus_area', 2500)
    
    # Precalcular máscaras y bordes a nivel de tile completo para optimizar
    nuc = detect_nuclei_mask(tile_rgb)
    cyto = detect_cytoplasm_mask(tile_rgb)
    tile_edges_d = detect_edges_dilation_tile(tile_rgb)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (nuc > 0).astype(np.uint8), connectivity=8
    )

    candidates = []
    half = crop_size // 2
    H, W = tile_rgb.shape[:2]

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_nucleus_area or area > max_nucleus_area:
            continue

        cx, cy = centroids[i]
        cx = int(round(cx))
        cy = int(round(cy))

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Evitar crops que excedan los bordes de la ventana de escaneo
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue

        crop = tile_rgb[y1:y2, x1:x2]
        crop_nuc = nuc[y1:y2, x1:x2]
        crop_cyto = cyto[y1:y2, x1:x2]
        crop_edges_d = tile_edges_d[y1:y2, x1:x2]

        # Contar núcleos válidos dentro del crop
        n2, _, stats2, _ = cv2.connectedComponentsWithStats(
            (crop_nuc > 0).astype(np.uint8), connectivity=8
        )

        nuc_count = 0
        for j in range(1, n2):
            a2 = stats2[j, cv2.CC_STAT_AREA]
            if a2 >= min_nucleus_area and a2 <= max_nucleus_area:
                nuc_count += 1

        focus_score = crop_focus_score(crop)
        _, edge_density = crop_edge_density(crop)
        cyto_frac = float((crop_cyto > 0).mean())

        nuc_in_cyto = nucleus_inside_cytoplasm(crop_nuc, crop_cyto, ring=6)
        has_cell_like_region = nucleus_inside_cell_like_region(crop_edges_d, crop_nuc)

        candidate_type = classify_candidate_type(
            nuc_count=nuc_count,
            focus_score=focus_score,
            edge_density=edge_density,
            cyto_frac=cyto_frac,
            nuc_in_cyto=nuc_in_cyto,
            has_cell_like_region=has_cell_like_region,
            params=params
        )

        # Filtro de descarte extremo para no almacenar basura total
        if focus_score < (params.get('min_focus_score', 35.0) * 0.4) and candidate_type == 'low_quality':
            continue

        candidates.append({
            "x_wsi": int(tile_x + x1),
            "y_wsi": int(tile_y + y1),
            "crop_size": int(crop_size),
            "nuc_count": int(nuc_count),
            "focus_score": float(focus_score),
            "edge_density": float(edge_density),
            "cyto_frac": float(cyto_frac),
            "has_cyto": int(cyto_frac >= params.get('min_cyto_frac', 0.03)),
            "nuc_in_cyto": int(nuc_in_cyto),
            "has_cell_like_region": int(has_cell_like_region),
            "candidate_type": candidate_type,
            "crop_image": crop  # Se guarda la matriz de la imagen para guardarse después
        })

    return candidates

def scan_rois_for_candidates(
    wsi_path,
    df_rois,
    params,
    progress_callback=None
):
    """
    Orquesta el escaneo de todas las ROIs y extrae los candidatos celulares.
    """
    tile_size = params.get('tile_size', 256)
    stride = params.get('stride', 128)
    max_cells_total = params.get('max_cells_total', 4000)

    all_candidates = []
    total_rois = len(df_rois)

    for idx, r in df_rois.iterrows():
        roi_id = r["roi_id"]
        X1, Y1, X2, Y2 = int(r["x1_wsi"]), int(r["y1_wsi"]), int(r["x2_wsi"]), int(r["y2_wsi"])

        roi_candidates = 0

        for yy in range(Y1, Y2 - tile_size + 1, stride):
            # Pre-leer la fila completa del ROI a esta altura yy para evitar decodificar las mismas tiras/celdas repetidamente
            row_width = X2 - X1
            if row_width < tile_size:
                continue
                
            row_image = read_wsi_region(wsi_path, X1, yy, row_width, tile_size)
            if row_image.shape[0] != tile_size or row_image.shape[1] != row_width:
                continue

            for xx in range(X1, X2 - tile_size + 1, stride):
                # Extraer tile mediante slicing rápido de memoria
                x_offset = xx - X1
                tile = row_image[:, x_offset : x_offset + tile_size]

                candidates = extract_crops_from_tile(tile, xx, yy, params)

                for cand in candidates:
                    cand.update({"roi_id": roi_id})
                    all_candidates.append(cand)
                    roi_candidates += 1

                if len(all_candidates) >= max_cells_total:
                    break
            if len(all_candidates) >= max_cells_total:
                break

        if progress_callback:
            progress_callback(idx + 1, total_rois, len(all_candidates))

        if len(all_candidates) >= max_cells_total:
            print(f"Llegado al límite máximo de candidatos ({max_cells_total})")
            break

    return all_candidates
