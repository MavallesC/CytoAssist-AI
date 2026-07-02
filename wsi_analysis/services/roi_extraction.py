import pandas as pd
import numpy as np
from .wsi_reader import get_wsi_dimensions

def extract_rois_grid(
    wsi_path,
    thumb_rgb,
    clean_mask,
    roi_size=200,          # tamaño del ROI en thumbnail
    stride=200,            # igual -> sin solapamiento
    min_clean_frac=0.2,    # % mínimo de área limpia
):
    """
    Divide el WSI en una cuadrícula de ROIs basados en el thumbnail y la máscara limpia.
    Retorna un DataFrame de pandas con las coordenadas del thumbnail y WSI.
    """
    th, tw = thumb_rgb.shape[:2]
    H0, W0 = get_wsi_dimensions(wsi_path)

    sx = tw / float(W0)
    sy = th / float(H0)
    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy

    rois = []
    idx = 0

    for y in range(0, th - roi_size + 1, stride):
        for x in range(0, tw - roi_size + 1, stride):
            # Patch de máscara limpia en thumbnail
            patch = clean_mask[y:y + roi_size, x:x + roi_size]
            frac = (patch > 0).mean()

            if frac < min_clean_frac:
                continue

            # Coordenadas WSI escala completa
            X1 = int(x * inv_sx)
            Y1 = int(y * inv_sy)
            X2 = int((x + roi_size) * inv_sx)
            Y2 = int((y + roi_size) * inv_sy)

            roi_id = f"ROI_{idx:03d}"
            idx += 1

            rois.append({
                "roi_id": roi_id,
                "x1_thumb": int(x),
                "y1_thumb": int(y),
                "x2_thumb": int(x + roi_size),
                "y2_thumb": int(y + roi_size),
                "x1_wsi": X1,
                "y1_wsi": Y1,
                "x2_wsi": X2,
                "y2_wsi": Y2
            })

    return pd.DataFrame(rois)
