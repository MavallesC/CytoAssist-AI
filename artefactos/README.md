# Directorio de Artefactos de Inferencia IA

Coloque en esta carpeta los siguientes archivos:
1. `inference_artifacts.json`
2. El checkpoint del backbone de PyTorch (ej. `densenet121_backbone.pt`)
3. El clasificador CatBoost (ej. `catboost_classifier.cbm`)
4. El reductor PCA `pca.joblib` (si fue configurado)

El sistema cargará estos modelos de forma dinámica para realizar la clasificación citológica sobre la lámina WSI.
