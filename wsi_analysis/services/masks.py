import cv2
import numpy as np

def foreground_mask_from_thumb(thumb_rgb):
    """
    Máscara simple y robusta para separar 'material' vs fondo.
    Trabaja en HSV: el fondo suele ser muy brillante y de baja saturación.
    """
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # El frotis citológico suele tener saturación > 20 y brillo < 245
    mask = ((S > 20) & (V < 245)).astype(np.uint8) * 255

    # Limpieza morfológica (quita puntitos y rellena huecos)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    return mask

def detect_large_artifacts(thumb_rgb, fg_mask):
    """
    Detecta artefactos grandes sobre el thumbnail:
    - zonas muy oscuras (sangre / manchas de tinta)
    - zonas sobresaturadas (quemadas por el microscopio)
    - rayones / líneas largas en el portaobjetos
    """
    h, w = thumb_rgb.shape[:2]

    # --- 1. Zonas muy oscuras ---
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < 40).astype(np.uint8) * 255

    # --- 2. Zonas sobresaturadas ---
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    _, S, V = cv2.split(hsv)
    bright_mask = ((V > 245) & (S < 30)).astype(np.uint8) * 255

    # --- 3. Rayones / estructuras lineales ---
    edges = cv2.Canny(gray, 50, 150)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    scratches = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=1)
    scratches = (scratches > 0).astype(np.uint8) * 255

    # --- Combinar artefactos ---
    artifact_mask = np.zeros((h, w), dtype=np.uint8)
    artifact_mask[dark_mask > 0] = 255
    artifact_mask[bright_mask > 0] = 255
    artifact_mask[scratches > 0] = 255

    # Limitar solo a zonas con frotis (foreground)
    artifact_mask[fg_mask == 0] = 0

    # Limpieza morfológica para unir regiones
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_CLOSE, k, iterations=2)

    return artifact_mask

def stain_mask_purple_thumb(thumb_rgb):
    """
    Filtra las regiones con tinción morada/azul (donde hay material celular denso).
    """
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Rango morado aproximado en HSV
    purple = ((H >= 120) & (H <= 175) & (S >= 25) & (V <= 245)).astype(np.uint8) * 255

    # Unir regiones y remover pequeños ruidos
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, k, iterations=2)
    purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, k, iterations=1)
    return purple

def build_clean_mask(base_mask, artifact_mask, buffer_px=2):
    """
    clean_mask = base_mask - (artefactos dilatados)
    base_mask: puede ser fg_mask (foreground) o purple_mask (zonas teñidas)
    buffer_px: margen de seguridad para alejarse de los artefactos
    """
    base = (base_mask > 0).astype(np.uint8)
    art = (artifact_mask > 0).astype(np.uint8)

    if buffer_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_px + 1, 2 * buffer_px + 1))
        art = cv2.dilate(art, k, iterations=1)

    clean = base.copy()
    clean[art > 0] = 0

    # Suavizado para rellenar huecos
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k2, iterations=1)

    return (clean * 255).astype(np.uint8)
