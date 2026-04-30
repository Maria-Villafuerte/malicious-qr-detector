# Malicious QR Code Detector
## Clasificación de Códigos QR Maliciosos con CNN

**Proyecto - Implementación Fase 2**
Fabiola Contreras - 22787 | María Villafuerte - 22129

---

## 1. Descripción del modelo

El modelo clasifica imágenes de códigos QR en dos categorías —**normal** y **malware**— a partir del análisis directo de los píxeles, sin decodificar el contenido del QR. Este enfoque pre-escaneo permite detectar la amenaza antes de que el usuario quede expuesto al contenido malicioso.

### Arquitectura

CNN ligera entrenada desde cero sobre imágenes de **128×128 píxeles en escala de grises**.

```
Entrada: (B, 1, 128, 128)

ConvBlock(1  → 32)   128×128 → 64×64    [Conv2d + BN + ReLU + MaxPool]
ConvBlock(32 → 64)    64×64  → 32×32
ConvBlock(64 → 128)   32×32  → 16×16
ConvBlock(128→ 256)   16×16  →  8×8
Conv(256→256) + BN + ReLU               [sin pooling]
AdaptiveAvgPool(4×4)           →  4×4

Flatten → 4096
FC(4096→512) → ReLU → Dropout(0.4)
FC(512→128)  → ReLU → Dropout(0.4)
FC(128→2)

Salida: logits [normal, malware]
```

**Parámetros entrenables: 3,142,242**

### Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Dataset | 200,000 imágenes (100K normal / 100K malware) |
| Split | 70% train / 15% val / 15% test (estratificado) |
| Augmentación | Flip H, Flip V, Rotación ±10° |
| Optimizador | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Early stopping | patience=7 (activado en época 19) |
| Dispositivo | CUDA (Google Colab) |

---

## 2. Métricas de rendimiento

### Resultados en test set (30,000 imágenes)

| Métrica | Valor |
|---|---|
| Test Loss | 0.000694 |
| Accuracy | 0.9999 |
| AUC | 0.9999998 |
| Precision | 0.9998 |
| Recall | 0.9999 |
| F1-Score | 0.9999 |

### Por clase

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1.00 | 1.00 | 1.00 | 15,000 |
| Malware | 1.00 | 1.00 | 1.00 | 15,000 |
| **Macro avg** | **1.00** | **1.00** | **1.00** | **30,000** |

La matriz de confusión registró **4 errores** en 30,000 predicciones: 3 falsos positivos y 1 falso negativo.

### Curva ROC

AUC = 0.9999998. La curva sube verticalmente desde el origen, indicando separabilidad casi perfecta entre clases: el modelo asigna probabilidades cercanas a 0 para imágenes normales y cercanas a 1 para malware, sin solapamiento entre distribuciones.

---

## 3. Interpretación de métricas en el contexto del problema

### 3.1 ¿Son válidos los resultados perfectos?

Los resultados de ~99.99% de accuracy y AUC ≈ 1.0 son reales para el dataset utilizado, pero requieren contexto. Para determinar si el modelo aprendió patrones visuales genuinos del QR o características espurias del proceso de generación del dataset, se realizó un análisis de robustez evaluando el mismo modelo bajo distintas condiciones, sin re-entrenamiento.

### 3.2 Análisis de robustez

| Condición | Accuracy | AUC | ΔAcc | Estado |
|---|---|---|---|---|
| Baseline (original) | 0.9999 | 1.0000 | — | ✅ |
| Re-encodeado PNG | 0.9999 | 1.0000 | 0.0000 | ✅ |
| Ruido gaussiano (σ=15) | 0.9999 | 1.0000 | +0.0000 | ✅ |
| JPEG compresión (q=30) | 0.9998 | 1.0000 | −0.0001 | ✅ |
| Rotación 15° | 0.8927 | 0.9880 | −0.1072 | ⚠️ |
| Blur gaussiano (r=2) | 0.5000 | 0.5000 | −0.4999 | ❌ |
| Recorte 10% | 0.5029 | 0.5934 | −0.4970 | ❌ |

### 3.3 Interpretación por condición

**Re-encodeado PNG — sin impacto**
Al re-encodear las imágenes en memoria se eliminan todos los metadatos PNG (chunks tEXt, pHYs, sRGB) y se normaliza la tabla de compresión. El accuracy idéntico al baseline descarta que el modelo esté leyendo el fingerprint del generador a través de los metadatos del archivo: aprendió algo de los píxeles, no de la estructura del archivo.

**Ruido gaussiano y JPEG — sin impacto**
El modelo es robusto ante degradaciones de intensidad. Esto es consistente con la naturaleza binaria de los QRs (solo píxeles en 0 y 255): la estructura de módulos del QR sobrevive bien a ruido y compresión.

**Rotación 15° — caída moderada (−10.7%)**
Caída esperable: los códigos QR tienen orientación canónica y el modelo fue entrenado con augmentación de rotaciones pequeñas (±10°). Esta vulnerabilidad es relevante para un sistema de producción que maneje QRs capturados con cámara en distintos ángulos.

**Blur y recorte — colapso a clasificación aleatoria**
El resultado más significativo del análisis. El modelo colapsa a accuracy ≈ 0.50 ante blur gaussiano de radio 2 y recorte del 10% del frame, indicando que depende de la nitidez y completitud de los módulos del QR. Un blur leve sobre una imagen 128×128 destruye suficiente detalle como para eliminar la señal aprendida. Esto limita la aplicabilidad directa a imágenes escaneadas en condiciones reales.

### 3.4 Discusión

Los resultados perfectos en el test set son reales **para el dataset utilizado** (benign-and-malicious-qr-codes, Kaggle), cuyas imagenes son sinteticas, perfectamente binarias y capturadas en condiciones controladas. Sin embargo, el análisis de robustez revela que estos resultados no se generalizarían directamente a QRs reales escaneados con cámara, donde el blur y el recorte parcial son condiciones habituales.

En el contexto de la literatura revisada, el enfoque de análisis directo de pixeles (sin decodificar la URL) -propuesto tambien por Trad y Chehab (2025)- logró en ese trabajo un AUC de 0.9133 con LightGBM sobre el mismo tipo de features. Nuestros resultados superiores en el dataset controlado son consistentes con la mayor capacidad representacional de una CNN frente a features manuales, pero la brecha se reduciría en condiciones reales, en linea con lo que la literatura indica.

La limitación de generalización identificada responde directamente a la brecha de investigación señalada en el marco teorico del proyecto: ninguno de los estudios revisados evalua el rendimiento bajo condiciones de degradación visual, y este analisis constituye un aporte metodológico concreto en esa dirección.

---

## 4. Archivos del proyecto

```
├── cnn_model.py                     # Arquitectura QRCNN, QRDataset, funciones de train/eval
├── modelado.ipynb                   # Entrenamiento, evaluación y visualizaciones
├── robustness_analysis.ipynb        # Análisis de robustez bajo degradaciones
└── outputs/
    ├── best_model.pt                # Pesos del modelo (mejor val_loss)
    ├── training_cnn_history.csv     # Loss y accuracy por época
    ├── test_metrics.json            # Métricas finales en test
    ├── test_predictions.csv         # Predicciones y probabilidades por imagen
    └── robustness_results.csv       # Métricas de robustez por condición
```
