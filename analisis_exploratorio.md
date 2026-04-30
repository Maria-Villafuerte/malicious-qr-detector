# Malicious QR Code Detector

**Proyecto - Implementación Fase 1**
Fabiola Contreras - 22787 | María Villafuerte - 22129

---

## Descripción del Proyecto

Sistema de clasificación de códigos QR en dos categorías: **benignos** y **maliciosos**, utilizando análisis exploratorio de datos, ingeniería de características y modelos de machine learning.

---

## Dataset

**Fuente:** Kaggle - Benign and Malicious QR Codes

| Característica | Detalle |
|---|---|
| Total de imágenes | 200,000 |
| Clase `normal` (benigno) | 100,000 imágenes |
| Clase `malware` (malicioso) | 100,000 imágenes |
| Formato | PNG, escala de grises |
| Resolución predominante | 410×410 px (rango 370–490 px) |
| Tamaño promedio de archivo | ~928 bytes |

### Estructura del Dataset

```
kaggle-qr-codes/
  QR codes/
    benign/benign/        → 100,000 imágenes benignas
    malicious/malicious/  → 100,000 imágenes maliciosas
```

---

## Análisis Exploratorio de Datos (EDA)

### 1. Distribución de Clases

El dataset presenta un balance perfecto 50/50 entre clases, lo que elimina la necesidad de técnicas de balanceo como SMOTE y permite usar accuracy como métrica confiable.

![alt text](./assets/image.png)

---

### 2. Análisis de Dimensiones de Imagen

Las imágenes son cuadradas (ancho = alto) en ambas clases. Las distribuciones de tamaño son casi idénticas entre clases malignas y benignas.

| Métrica | Malware | Normal |
|---|---|---|
| Media de ancho | 405.86 px | 407.92 px |
| Desv. estándar | 12.21 px | 8.90 px |
| Rango | 370–490 px | 370–450 px |

**Conclusión:** Las dimensiones de imagen **no son una característica discriminativa**. El modelo debe aprender del contenido visual, no del tamaño.

![alt text](./assets/image-1.png)
---

### 3. Análisis de Tamaño de Archivo

- Malware: media 922.48 bytes, rango 718–1234 bytes
- Normal: media 933.33 bytes, rango 708–1101 bytes
- **Test Mann-Whitney U: p-value ≈ 0.0000** → diferencia estadísticamente significativa

Aunque existe significancia estadística, la separación práctica es mínima. Las distribuciones se superponen casi completamente. El tamaño de archivo tiene **poder discriminativo marginal** como característica individual.

![alt text](./assets/image-2.png)
---

### 4. Estadísticas de Píxeles (muestra de 500 imágenes por clase)

| Métrica | Malware | Normal |
|---|---|---|
| Media de píxel | 149.60 | 148.56 |
| Desv. estándar | 125.55 | 127.13 |
| Proporción de píxeles oscuros | 0.411 | 0.408 |

**Conclusión:** Las estadísticas globales de píxeles muestran diferencias mínimas entre clases. Las clases son prácticamente indistinguibles usando solo métricas globales de brillo.
![alt text](./assets/image-3.png)

---

## Ingeniería de Características

Se extrajeron **34 características** por imagen a partir de una muestra de 1,000 imágenes por clase (2,000 en total).

### Características Extraídas

| Grupo | Características | Descripción |
|---|---|---|
| Estadísticas globales | 5 | `mean_pixel`, `std_pixel`, `dark_ratio`, `skewness`, `kurtosis` |
| Entropía | 1 | Complejidad/aleatoriedad de la imagen binarizada |
| Coeficientes DCT | 10 | Energía frecuencial en perfiles horizontal y vertical (`dct_h_0`–`dct_h_4`, `dct_v_0`–`dct_v_4`) |
| LBP (Local Binary Patterns) | 10 | Distribución de patrones de textura (`lbp_0`–`lbp_9`) |
| Análisis por cuadrantes | 8 | Media y desv. estándar por región (superior-izq, superior-der, inferior-izq, inferior-der) |

---

## Selección de Características

### Importancia de Características (Random Forest)

Las 15 características más importantes identificadas con un clasificador Random Forest:

![alt text](./assets/image-4.png)
| Rank | Característica | Importancia |
|---|---|---|
| 1 | `q_br_mean` | 0.0741 |
| 2 | `q_br_std` | 0.0436 |
| 3 | `dct_h_2` | 0.0427 |
| 4 | `lbp_8` | 0.0361 |
| 5 | `dct_h_1` | 0.0360 |
| 6 | `dct_v_1` | 0.0359 |
| 7 | `dct_v_3` | 0.0350 |
| 8 | `dct_v_2` | 0.0337 |
| 9 | `dct_h_3` | 0.0325 |
| 10 | `lbp_3` | 0.0309 |
| 11 | `dct_h_4` | 0.0302 |
| 12 | `q_tr_mean` | 0.0298 |
| 13 | `entropy` | 0.0287 |
| 14 | `dark_ratio` | 0.0284 |
| 15 | `dct_v_4` | 0.0284 |

### Resultado de la Selección

Usando `SelectFromModel` con umbral de importancia media:

- **Características seleccionadas: 12 de 34** (reducción del 64.7%)
- Características seleccionadas: `dct_h_1`, `dct_h_2`, `dct_h_3`, `dct_h_4`, `dct_v_1`, `dct_v_2`, `dct_v_3`, `lbp_3`, `lbp_8`, `q_tr_mean`, `q_br_mean`, `q_br_std`

| Configuración | Accuracy CV | Desv. estándar |
|---|---|---|
| 34 características (todas) | 0.7290 | ±0.0090 |
| 12 características (seleccionadas) | 0.7325 | ±0.0127 |

El subconjunto de 12 características **mantiene e incluso mejora levemente** el accuracy respecto al conjunto completo.

![alt text](./assets/image-5.png)

---

## Separabilidad de Clases (PCA 2D)

![alt text](./assets/image-6.png)

Las clases presentan **solapamiento significativo** en el espacio proyectado 2D.

**Conclusión:** Las clases **no son linealmente separables** en el espacio de características tabulares. Esto implica que los patrones visuales/espaciales que distinguen QR benignos de maliciosos requieren representaciones más ricas que las estadísticas globales.

---

## Conclusiones y Estrategia de Modelado

### Hallazgos Clave

1. **Calidad del Dataset:** Balance perfecto, sin valores nulos, 200K muestras suficientes para deep learning.

2. **Características Tabulares:**
   - Las estadísticas globales son insuficientes para clasificación confiable.
   - Los coeficientes DCT y descriptores LBP capturan patrones estructurales no evidentes en estadísticas globales.
   - Las características espaciales por cuadrante son relevantes (especialmente el cuadrante inferior-derecho).
   - El subconjunto de 12 características mantiene el rendimiento del conjunto completo de 34.

3. **Separabilidad:** Las clases no son linealmente separables en el espacio de características → los patrones discriminativos son de naturaleza espacial/visual compleja.

### Estrategia para el Modelado

| Modelo | Descripción |
|---|---|
| **Modelo Principal (CNN)** | Red convolucional con entrada de imágenes 128×128 en escala de grises. Aprende representaciones espaciales automáticamente. |
| **Modelo Baseline** | Random Forest o Gradient Boosting sobre las 12 características seleccionadas, para comparación. |

---

## Archivos Generados

| Archivo | Descripción |
|---|---|
| `data_procesada/qr_full_dataset.csv` | Metadatos completos de las 200K imágenes |
| `data_procesada/features_sample.csv` | 34 características extraídas de muestra de 2K imágenes |
| `data_procesada/feature_selection.json` | Resumen de importancia y características seleccionadas |
