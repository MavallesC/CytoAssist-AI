import os
import cv2
import numpy as np
import tifffile
import warnings

# Silenciar advertencias de tiff
warnings.filterwarnings("ignore", category=UserWarning)

def get_wsi_dimensions(wsi_path):
    """
    Retorna (height, width) de la resolución base del WSI.
    Si el archivo no existe o falla, retorna dimensiones simuladas.
    """
    if not os.path.exists(wsi_path):
        # Dimensiones simuladas por defecto
        return 20000, 20000

    try:
        with tifffile.TiffFile(wsi_path) as tif:
            series = tif.series[0]
            shape = series.shape
            return int(shape[0]), int(shape[1])
    except Exception:
        # Fallback a dimensiones de simulación si no es un TIFF válido
        return 20000, 20000

def get_wsi_thumbnail(wsi_path, max_side=2000):
    """
    Genera un thumbnail eficiente:
    - Si el TIFF tiene niveles piramidales, usa el nivel más pequeño.
    - Si no, y es una imagen grande, lee de forma estriada/celda a celda sin cargar en RAM.
    - Si el archivo no existe, genera un thumbnail sintético.
    Retorna: img_rgb (H, W, 3) uint8
    """
    if not os.path.exists(wsi_path):
        return _generate_mock_thumbnail(max_side)

    try:
        with tifffile.TiffFile(wsi_path) as tif:
            series = tif.series[0]
            page = tif.pages[0]
            
            # Caso 1: Tiene pirámide de resolución
            if hasattr(series, "levels") and len(series.levels) > 1:
                img = series.levels[-1].asarray()
                # Asegurar 3 canales
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[-1] == 4:
                    img = img[..., :3]
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h, w = img.shape[:2]
                scale = max(h, w) / float(max_side)
                if scale > 1:
                    new_w = int(w / scale)
                    new_h = int(h / scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return img
                
            # Caso 2: Es plano (1 nivel)
            H, W = page.shape[:2]
            scale = max(H, W) / float(max_side)
            
            # Si la imagen es pequeña, la cargamos directamente completa
            if scale <= 1.0 or max(H, W) <= 4000:
                img = series.asarray()
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[-1] == 4:
                    img = img[..., :3]
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h, w = img.shape[:2]
                if scale > 1:
                    new_w = int(w / scale)
                    new_h = int(h / scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return img
            
            # Caso 3: Es una imagen plana gigante (WSI sin niveles de resolución)
            # Decodificamos y reescalamos únicamente las tiras/celdas necesarias
            W_thumb = int(W / scale)
            H_thumb = int(H / scale)
            
            thumb = np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8)
            jpegtables = page.jpegtables
            jpegheader = page.jpegheader
            fh = tif.filehandle
            
            if page.is_tiled:
                tilewidth = page.tilewidth
                tilelength = page.tilelength
                tiles_x = (W + tilewidth - 1) // tilewidth
                
                # Caching por fila de tiles
                last_tile_y = -1
                last_tile_row_stitched = None
                
                for y_thumb in range(H_thumb):
                    y_src = int(y_thumb * scale)
                    tile_y = y_src // tilelength
                    row_in_tile = y_src % tilelength
                    
                    if y_src >= H:
                        continue
                        
                    if tile_y != last_tile_y:
                        tile_y_strips = []
                        for tx in range(tiles_x):
                            tile_idx = tile_y * tiles_x + tx
                            offset = page.dataoffsets[tile_idx]
                            bytecount = page.databytecounts[tile_idx]
                            fh.seek(offset)
                            raw_data = fh.read(bytecount)
                            res = page.decode(raw_data, tile_idx, jpegtables=jpegtables, jpegheader=jpegheader)
                            tile_arr = np.squeeze(res[0], axis=0)
                            if tile_arr.ndim == 2:
                                tile_arr = cv2.cvtColor(tile_arr, cv2.COLOR_GRAY2RGB)
                            elif tile_arr.shape[-1] == 4:
                                tile_arr = tile_arr[..., :3]
                            tile_y_strips.append(tile_arr)
                        
                        last_tile_row_stitched = np.concatenate(tile_y_strips, axis=1)
                        last_tile_y = tile_y
                        
                    row = last_tile_row_stitched[row_in_tile, :W]
                    row_reshaped = row[np.newaxis, ...]
                    row_resized = cv2.resize(row_reshaped, (W_thumb, 1), interpolation=cv2.INTER_AREA)
                    thumb[y_thumb] = row_resized[0]
            else:
                rowsperstrip = page.rowsperstrip
                last_strip_idx = -1
                last_strip_arr = None
                
                for y_thumb in range(H_thumb):
                    y_src = int(y_thumb * scale)
                    strip_idx = y_src // rowsperstrip
                    row_in_strip = y_src % rowsperstrip
                    
                    if y_src >= H:
                        continue
                        
                    if strip_idx != last_strip_idx:
                        offset = page.dataoffsets[strip_idx]
                        bytecount = page.databytecounts[strip_idx]
                        fh.seek(offset)
                        raw_data = fh.read(bytecount)
                        res = page.decode(raw_data, strip_idx, jpegtables=jpegtables, jpegheader=jpegheader)
                        last_strip_arr = np.squeeze(res[0], axis=0)
                        if last_strip_arr.ndim == 2:
                            last_strip_arr = cv2.cvtColor(last_strip_arr, cv2.COLOR_GRAY2RGB)
                        elif last_strip_arr.shape[-1] == 4:
                            last_strip_arr = last_strip_arr[..., :3]
                        last_strip_idx = strip_idx
                        
                    row = last_strip_arr[row_in_strip, :W]
                    row_reshaped = row[np.newaxis, ...]
                    row_resized = cv2.resize(row_reshaped, (W_thumb, 1), interpolation=cv2.INTER_AREA)
                    thumb[y_thumb] = row_resized[0]
                    
            return thumb
            
    except Exception as e:
        print(f"Error al generar thumbnail eficiente de WSI real ({e}), usando simulación...")
        return _generate_mock_thumbnail(max_side)

_WSI_CACHE = {}

def clear_wsi_cache():
    """Limpia el caché de archivos WSI abiertos."""
    global _WSI_CACHE
    for tif in _WSI_CACHE.values():
        try:
            tif.close()
        except Exception:
            pass
    _WSI_CACHE.clear()

def read_wsi_region(wsi_path, x, y, w, h):
    """
    Lee una región del WSI en coordenadas de alta resolución de manera eficiente.
    Si el WSI no existe, genera contenido sintético (células simuladas).
    Retorna: img_rgb (h, w, 3) uint8
    """
    if not os.path.exists(wsi_path):
        return _generate_mock_region(x, y, w, h)

    global _WSI_CACHE
    if wsi_path not in _WSI_CACHE:
        try:
            _WSI_CACHE[wsi_path] = tifffile.TiffFile(wsi_path)
            print(f"[wsi_reader] Cached TiffFile object for: {wsi_path}")
        except Exception as e:
            print(f"[wsi_reader] No se pudo abrir el TIFF ({e}). Usando generador simulado.")
            return _generate_mock_region(x, y, w, h)

    try:
        tif = _WSI_CACHE[wsi_path]
        page = tif.pages[0]
        fh = tif.filehandle
        
        H, W = page.shape[:2]
        
        x1 = max(0, min(x, W - 1))
        y1 = max(0, min(y, H - 1))
        x2 = max(0, min(x + w, W))
        y2 = max(0, min(y + h, H))
        
        req_w = x2 - x1
        req_h = y2 - y1
        
        if req_w <= 0 or req_h <= 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
            
        region = np.zeros((req_h, req_w, 3), dtype=np.uint8)
        jpegtables = page.jpegtables
        jpegheader = page.jpegheader
        
        if page.is_tiled:
            tilewidth = page.tilewidth
            tilelength = page.tilelength
            tiles_x = (W + tilewidth - 1) // tilewidth
            
            start_tile_y = y1 // tilelength
            end_tile_y = (y2 - 1) // tilelength
            start_tile_x = x1 // tilewidth
            end_tile_x = (x2 - 1) // tilewidth
            
            for ty in range(start_tile_y, end_tile_y + 1):
                tile_y1 = ty * tilelength
                tile_y2 = min(H, (ty + 1) * tilelength)
                oy1 = max(y1, tile_y1)
                oy2 = min(y2, tile_y2)
                
                for tx in range(start_tile_x, end_tile_x + 1):
                    tile_x1 = tx * tilewidth
                    tile_x2 = min(W, (tx + 1) * tilewidth)
                    ox1 = max(x1, tile_x1)
                    ox2 = min(x2, tile_x2)
                    
                    tile_idx = ty * tiles_x + tx
                    offset = page.dataoffsets[tile_idx]
                    bytecount = page.databytecounts[tile_idx]
                    
                    fh.seek(offset)
                    raw_data = fh.read(bytecount)
                    
                    res = page.decode(raw_data, tile_idx, jpegtables=jpegtables, jpegheader=jpegheader)
                    tile_arr = np.squeeze(res[0], axis=0)
                    if tile_arr.ndim == 2:
                        tile_arr = cv2.cvtColor(tile_arr, cv2.COLOR_GRAY2RGB)
                    elif tile_arr.shape[-1] == 4:
                        tile_arr = tile_arr[..., :3]
                        
                    src_y1 = oy1 - tile_y1
                    src_y2 = oy2 - tile_y1
                    src_x1 = ox1 - tile_x1
                    src_x2 = ox2 - tile_x1
                    
                    dst_y1 = oy1 - y1
                    dst_y2 = oy2 - y1
                    dst_x1 = ox1 - x1
                    dst_x2 = ox2 - x1
                    
                    region[dst_y1:dst_y2, dst_x1:dst_x2] = tile_arr[src_y1:src_y2, src_x1:src_x2]
        else:
            rowsperstrip = page.rowsperstrip
            start_strip = y1 // rowsperstrip
            end_strip = (y2 - 1) // rowsperstrip
            
            for s in range(start_strip, end_strip + 1):
                strip_y1 = s * rowsperstrip
                strip_y2 = min(H, (s + 1) * rowsperstrip)
                oy1 = max(y1, strip_y1)
                oy2 = min(y2, strip_y2)
                
                offset = page.dataoffsets[s]
                bytecount = page.databytecounts[s]
                
                fh.seek(offset)
                raw_data = fh.read(bytecount)
                
                res = page.decode(raw_data, s, jpegtables=jpegtables, jpegheader=jpegheader)
                strip_arr = np.squeeze(res[0], axis=0)
                if strip_arr.ndim == 2:
                    strip_arr = cv2.cvtColor(strip_arr, cv2.COLOR_GRAY2RGB)
                elif strip_arr.shape[-1] == 4:
                    strip_arr = strip_arr[..., :3]
                    
                src_y1 = oy1 - strip_y1
                src_y2 = oy2 - strip_y1
                src_x1 = x1
                src_x2 = x2
                
                dst_y1 = oy1 - y1
                dst_y2 = oy2 - y1
                dst_x1 = 0
                dst_x2 = req_w
                
                region[dst_y1:dst_y2, dst_x1:dst_x2] = strip_arr[src_y1:src_y2, src_x1:src_x2]
                
        # Asegurar tamaño exacto solicitado
        if region.shape[0] != h or region.shape[1] != w:
            region = cv2.resize(region, (w, h))
        return region
        
    except Exception as e:
        print(f"[wsi_reader] Error al obtener región eficiente ({e}), usando simulación...")
        return _generate_mock_region(x, y, w, h)



def _generate_mock_thumbnail(max_side=2000):
    """
    Genera una imagen de thumbnail simulada de una muestra citológica.
    Fondo brillante con 3 o 4 manchas de tejido morado/rosado.
    """
    # Determinar dimensiones
    th, tw = int(max_side * 0.75), max_side
    # Fondo casi blanco (típico de microscopio)
    img = np.ones((th, tw, 3), dtype=np.uint8) * 248

    # Dibujar manchas de frotis citológico
    np.random.seed(42)
    n_spots = 5
    for _ in range(n_spots):
        cx = np.random.randint(int(tw * 0.2), int(tw * 0.8))
        cy = np.random.randint(int(th * 0.2), int(th * 0.8))
        rx = np.random.randint(80, 250)
        ry = np.random.randint(80, 250)
        angle = np.random.randint(0, 180)

        # Crear una elipse suave morado/rosado (frotis)
        mask = np.zeros((th, tw), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
        # Suavizar bordes de la mancha
        mask = cv2.GaussianBlur(mask, (71, 71), 0)

        # Aplicar coloración hematoxilina-eosina típica (mezcla morada/rosa)
        color = np.array([215, 170, 220]) # BGR -> RGB: morado rosáceo
        for c in range(3):
            img[..., c] = np.clip(
                img[..., c] * (1 - mask / 255.0 * 0.35) + color[c] * (mask / 255.0 * 0.35),
                0, 255
            ).astype(np.uint8)

    # Añadir un par de líneas que simulen suciedad o rayones en el vidrio (artefactos)
    cv2.line(img, (int(tw*0.1), int(th*0.1)), (int(tw*0.15), int(th*0.4)), (20, 20, 20), 2)
    cv2.line(img, (int(tw*0.55), int(th*0.8)), (int(tw*0.8), int(th*0.82)), (10, 15, 10), 3)

    return img

def _generate_mock_region(x, y, w, h):
    """
    Genera un tile simulado con células (núcleos oscuros y citoplasmas tenues).
    Usa el residuo de (x, y) como semilla para que los tiles sean consistentes.
    """
    # Semilla reproducible basada en coordenadas
    seed = (x * 73856093) ^ (y * 19349663)
    np.random.seed(seed & 0xFFFFFFFF)

    # Fondo claro del microscopio
    img = np.ones((h, w, 3), dtype=np.uint8) * 242

    # Decidir si la región tiene muestra celular (simular distribución espacial)
    # Por ejemplo, si está en cierta región sinusoidal
    # Dibujar citoplasmas (más grandes y claros, rosados/morados)
    n_cells = np.random.randint(15, 45)
    for _ in range(n_cells):
        cx = np.random.randint(10, w - 10)
        cy = np.random.randint(10, h - 10)
        cr = np.random.randint(15, 45) # Radio citoplasma

        # Citoplasma lila/rosado
        color_cyto = np.array([np.random.randint(200, 230), np.random.randint(140, 175), np.random.randint(200, 240)])
        cv2.circle(img, (cx, cy), cr, color_cyto.tolist(), -1)
        # Borde citoplasma difuso
        cv2.circle(img, (cx, cy), cr, (color_cyto * 0.95).astype(int).tolist(), 1)

        # Dibujar núcleo(s) - A veces 1, raramente más
        nuc_count = np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03])
        for n in range(nuc_count):
            nx = cx + np.random.randint(-10, 10) if nuc_count > 1 else cx
            ny = cy + np.random.randint(-10, 10) if nuc_count > 1 else cy
            nr = np.random.randint(4, 9) # Radio núcleo

            # Núcleo oscuro (azul/morado intenso)
            color_nuc = np.array([np.random.randint(90, 120), np.random.randint(40, 70), np.random.randint(120, 150)])
            cv2.circle(img, (nx, ny), nr, color_nuc.tolist(), -1)
            cv2.circle(img, (nx, ny), nr, (color_nuc * 0.8).astype(int).tolist(), 1)

    # Agregar algo de ruido de fondo
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Agregar un artefacto negro ocasionalmente
    if np.random.rand() < 0.05:
        # Dibujar mancha negra
        cv2.circle(img, (np.random.randint(0, w), np.random.randint(0, h)), np.random.randint(5, 15), (20, 20, 25), -1)

    return img
