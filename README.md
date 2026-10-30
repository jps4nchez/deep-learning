# Workshops de Deep Learning

Talleres prácticos sobre redes neuronales densas (MLP) en datos tabulares. Cada notebook contiene el código base del curso, las celdas de apoyo añadidas para resolver las actividades y las respuestas escritas en celdas de texto.

## Estructura

```
workshop/
├── WorkShop_1/                               DNNs en datos tabulares (TensorFlow/Keras)
│   ├── DNN_Regression_Price of Houses.ipynb  Parte 1: regresión del precio de viviendas
│   ├── DNN_Clasificacion_de_Viviendas.ipynb  Parte 2: clasificación (¿precio sobre la mediana?)
│   ├── kc_house_data_yr.xlsx                 King County (Seattle), 21 613 ventas 2014–2015
│   └── housepricedata.csv                    1 460 viviendas, 10 variables + etiqueta 0/1
└── WorkShop_2/                               ¿ML clásico o deep learning? (PyTorch + scikit-learn)
    ├── Ejercicio_ML_vs_DL_HIGGS.ipynb        Búsqueda de un Higgs pesado con datos simulados
    ├── higgs_2M.parquet                      2,1 millones de choques (se descarga solo, ~405 MB)
    └── resultados_curva.json                 Resultados del experimento 1 (los genera el notebook)
```

## Entorno de ejecución

| Notebook | Dónde se ejecutó | Hardware |
|---|---|---|
| `DNN_Regression_Price of Houses.ipynb` | Local (Linux) | CPU de 8 núcleos, 7,7 GB RAM, sin GPU |
| `DNN_Clasificacion_de_Viviendas.ipynb` | Local (Linux) | CPU de 8 núcleos, 7,7 GB RAM, sin GPU |
| `Ejercicio_ML_vs_DL_HIGGS.ipynb` | **Google Colab** | **GPU NVIDIA Tesla T4** (2 núcleos de CPU) |

El notebook HIGGS **se corrió en Google Colab con una GPU Tesla T4**. Con GPU, el notebook se ejecuta en modo completo, con hasta 2 000 000 de datos de entrenamiento. Sin GPU activa automáticamente un *modo rápido* (máximo 300 000 datos, sin las celdas de 2 millones), así que en un equipo sin GPU los resultados no coincidirán con los que están guardados. En Colab: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.

Versiones locales: Python 3.12, TensorFlow 2.21, PyTorch 2.15, scikit-learn 1.9.1, pandas 3.1 (ver `../pyproject.toml`).

## Cómo ejecutarlos

Entorno local con [uv](https://docs.astral.sh/uv/) desde la raíz del proyecto:

```bash
uv sync                      # crea .venv con las dependencias de pyproject.toml
source .venv/bin/activate
jupyter lab                  # abrir el notebook y ejecutar todas las celdas
```

o sin interfaz:

```bash
cd workshop/WorkShop_1
jupyter nbconvert --to notebook --execute --inplace DNN_Clasificacion_de_Viviendas.ipynb
```

Los notebooks leen los datos desde **su misma carpeta**. El de HIGGS descarga `higgs_2M.parquet` desde Hugging Face la primera vez.

Tiempos aproximados: clasificación ≈ 2 min y regresión ≈ 8 min en CPU. HIGGS en Colab con T4 tarda varios minutos por experimento; en CPU es mucho más lento.

## Actividades y dónde están las respuestas

### WorkShop 1 · Parte 1: Regresión (`DNN_Regression_Price of Houses.ipynb`)
- EDA: tipos de variables, nulos, distribuciones, correlaciones. **Respuestas** después de la tabla de correlación con el precio.
- Ejercicio de limitar el precio a 3 M USD y ejercicio de incorporar `zipcode` con *one-hot*.
- **13 preguntas en grupo**: respuestas en la celda *"Respuestas a las preguntas en grupo"*, al final, después de las celdas de apoyo. Esas celdas añaden la comparación con regresión lineal, el MAE por partición y un análisis sin `lat`/`long`.

| Modelo | MAE (USD) | R² |
|---|---|---|
| Regresión lineal sin ZIP | 121 088 | 0,707 |
| DNN sin ZIP | 104 036 | 0,791 |
| Regresión lineal con ZIP | 93 873 | 0,808 |
| **DNN con ZIP** | **69 022** | **0,901** |

### WorkShop 1 · Parte 2: Clasificación (`DNN_Clasificacion_de_Viviendas.ipynb`)
- Comparación de 5 arquitecturas (anchura, profundidad, *dropout*, L2), métricas, importancia por permutación y varias semillas.
- **5 preguntas en grupo**: respuestas en la celda *"Respuestas"*, al final.
- Modelo base en prueba: exactitud 0,890, F1 0,891, ROC-AUC 0,964. Mejor F1: red ancha 2×64 (0,898).

### WorkShop 2 · HIGGS (`Ejercicio_ML_vs_DL_HIGGS.ipynb`, ejecutado en Colab con T4)
- Experimentos 1–6 guiados. Las respuestas a las *apuestas* están antes de *"Retos para entregar"*.
- **Retos 1–3**: código añadido en las celdas *"Solución reto N"*, cada una seguida de su respuesta.
- **Reto 4** (pregunta de industria): respondido junto al reto 3.
- **Retos 5 y 6**: respuestas debajo de sus preguntas. El reto 6 incluye celdas de apoyo con la comparación FP/FN y el costo por umbral.

| Datos de entrenamiento | Gradient Boosting | Red neuronal |
|---|---|---|
| 1 000 | **66,6 %** | 57,5 % |
| 100 000 | **73,0 %** | 71,9 % |
| 300 000 | 73,7 % | **73,9 %** |
| 2 000 000 | 74,3 % | **76,5 %** |

## Cambios respecto al material original

- **Regresión:** las rutas a `kc_house_data_yr.xlsx` apuntaban a `~/Downloads/...`; ahora son relativas a la carpeta del notebook. En la sección de ZIP, la fecha (entero `AAAAMMDD`) se leía como nanosegundos desde 1970; ahora se interpreta con `format="%Y%m%d"`. Las celdas `%pip install` quedaron comentadas porque las dependencias están en `pyproject.toml`.
- **Clasificación:** la tabla de comparación de arquitecturas incluye además la pérdida de entrenamiento (sin *dropout*), la brecha entre entrenamiento y validación, y precisión, exhaustividad y AUC en prueba.
- **HIGGS:** se añadieron las celdas de solución de los retos 1–3 y celdas de apoyo para el reto 6. El código base no se modificó.
- Copia de los notebooks originales, sin modificar, en `../backup_original/`.

## Advertencias

- Los resultados dependen de la semilla, de la versión de las bibliotecas y del hardware. Las cifras de las respuestas corresponden a la ejecución guardada en cada notebook.
- Los textos de *"Lectura"* del notebook HIGGS son del material original y pueden citar cifras que difieren levemente de esta ejecución.
- En los retos 2 y 3 la mejor configuración y el peso del combo se eligieron mirando la exactitud de prueba. Lo correcto metodológicamente es elegir con validación; está señalado en las respuestas.
