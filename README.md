# Implementación de Phase Coupling Decomposition para remoción de artefactos del habla en registros intracraneales (iEEG)
Bautista Pelossi Schweizer | **Interfaces Cerebro-Computadora (BCI)** | Facultad de Ingeniería Química, Universidad Nacional del Litoral
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Repo](https://img.shields.io/badge/github-bautipelossi%2Fbci--pcd-black.svg)](https://github.com/bautipelossi/bci-pcd)

Implementación en Python de **Phase Coupling Decomposition (PCD)**, un filtrado espacial supervisado que remueve el artefacto acústico del habla en registros intracraneales (iEEG).(Peterson et al,2024) Tiene cuatro etapas: estimación de la banda del artefacto (**SAFB**), descomposición espacio-espectral (**SSD**), optimización del acoplamiento de fase (**PCO**) y reconstrucción de la señal limpia.

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Ejecución](#ejecución)
5. [Parámetros](#parámetros)
6. [Prueba rápida (smoke test)](#prueba-rápida-smoke-test)
7. [Datos](#datos)
8. [Validación](#validación)
9. [Referencias](#referencias)

---

## Arquitectura

```
Audio z (N_s,)                      iEEG X (N_c, N_s)
    │                                   │
    ▼                                   ▼
estimate_safb                       whiten
  ├─ PSD (Welch)                      ├─ señal analítica (Hilbert)
  ├─ ajuste gaussiano en F0           ├─ centrado
  └─ [fl, fh]  ───────────┐           └─ blanqueo con M
                          │                 │
                          │                 ▼  X_white (complejo)
                          └──────────►  fit_ssd  (mne.decoding.SSD)
                                        ├─ señal: [fl, fh]
                                        ├─ ruido: 4–240 Hz
                                        └─ W_ssd, λ
                                            │
                                            ▼
                                        select_n_components → k (participation ratio)
                                            │
                                            ▼  X_ssd = W_ssd[:, :k]ᵀ · X_white
                                        fit_pco
                                        └─ max MVL(w) con z → W_pco, vlen
                                            │
                                            ▼
                                        PCD.fit / PCD.apply
                                        ├─ W_pcd = M · W_ssd · blockdiag(W_pco, I)
                                        ├─ A_pcd = pinv(W_pcd)ᵀ
                                        └─ X_limpia = A_pcd[:, keep] · W_pcd[:, keep]ᵀ · X
```

**Decisiones de diseño:**

- **Convención de formas**: los datos son siempre canales × muestras `(N_c, N_s)`. Los filtros y patrones van en las **columnas** (`X = A S`, `S = Wᵀ X`).
- **Hilbert antes del blanqueo**: la misma matriz `M` blanquea la parte real (que usa SSD) y conserva la fase (que usa PCO).
- **Rango completo obligatorio**: `whiten` falla con un error claro si los datos tienen rango deficiente (por ejemplo, después de una CAR) y sugiere quitar un canal.
- **SAFB con Nelder-Mead**: la gaussiana se ajusta con Nelder-Mead arrancando en F0; con mínimos cuadrados no lineales el ajuste podía no converger o saltar al armónico.
- **SSD con MNE**: `mne.decoding.SSD` con Butterworth IIR de orden 5, sin regularización (`reg=None`), sin restricción de rango (`restr_type=None`) y ordenado por autovalor (`sort_by_spectral_ratio=False`).
- **Autovalores normalizados**: MNE devuelve el cociente señal/ruido λ ∈ (0, ∞). Se usa λ / (1 + λ) ∈ (0, 1), que es la fracción de potencia en la SAFB, para calcular el participation ratio.
- **MVL como objetivo de PCO**: `MVL(w) = |(1/N_s) Σ_t z(t) · exp(i · angle(wᵀ x(t)))|`, con `z` normalizado (media 0, varianza 1). Se maximiza con BFGS y gradiente analítico, 15 inicializaciones aleatorias por filtro y deflación (cada filtro nuevo es ortogonal a los anteriores).
- **Patrones en columnas**: `A_pcd = pinv(W_pcd)ᵀ`. La implementación de los autores guarda `pinv(W_pcd)`, con los patrones en las filas.
- **Componentes de artefacto**: por defecto se sacan con el criterio `"diff"` (codo de la curva de MVL), como en el código de los autores. Un `int` n saca las n primeras componentes.

---

## Estructura del proyecto

```
bci-pcd/
├── data/                  ← datos (no se versionan, ver Datos)
├── src/
│   ├── safb.py            ← estimación de la banda del artefacto (PSD + ajuste gaussiano)
│   ├── ssd.py             ← blanqueo, SSD (MNE) y elección de k
│   ├── pco.py             ← optimización de MVL
│   ├── pcd.py             ← clase PCD: fit / transform / apply
│   └── __init__.py
├── validation/
│   ├── metrics.py         ← métricas del paper (MSE, MSCE, PLV, χ, CS, ITPC)
│   ├── run_validation.py  ← comparación contra la referencia sobre datos reales
│   └── simulate.py        ← simulación con ground truth conocido
├── pyproject.toml
└── README.md
```

---

## Instalación

Requiere Python ≥ 3.10 y [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Dependencias: NumPy, SciPy, MNE-Python, scikit-learn (requerido por `mne.decoding`) y pymatreader (lee `.mat`, incluido v7.3).

---

## Ejecución

### Como librería

```python
from src.pcd import PCD

# X_fit: (N_c, N_s) iEEG en la ventana del habla · z: (N_s,) audio alineado
# X: (N_c, N_s') iEEG a limpiar (puede ser una ventana más larga)
pcd = PCD(sfreq).fit(X_fit, z, f0=f0)   # o PCD(sfreq, signal_band=(fl, fh)) para fijar la SAFB
X_limpia = pcd.apply(X)                  # saca las componentes de artefacto ("diff")

pcd.signal_band_     # SAFB usada
pcd.k_               # componentes SSD que pasaron a PCO
pcd.vlen_            # MVL de cada componente PCO, descendente
pcd.W_pcd_, pcd.A_pcd_
```

Cada etapa también se puede usar por separado (`estimate_safb`, `whiten`, `fit_ssd`, `select_n_components`, `fit_pco`).

### Comandos

```bash
uv run pcd-validar                 # validación contra la referencia sobre datos reales
uv run pcd-validar otra_ref.mat    # con otra referencia
uv run pcd-simular                 # simulación con ground truth (PCD, referencia y CAR)
uv run pytest                      # tests
```

Los comandos están definidos en `[project.scripts]` del `pyproject.toml`. Se corren desde la raíz del repo porque leen los datos de `data/`.

---

## Parámetros

| Función | Parámetro | Descripción | Default |
|---|---|---|---|
| `PCD` | `signal_band` | SAFB `(fl, fh)` en Hz; si es `None` se estima en `fit` con `f0` | `None` |
| `PCD` | `n_components` | Cómo elegir k (igual que `rule` de `select_n_components`) | `"PR"` |
| `PCD` | `n_restarts` | Inicializaciones aleatorias por filtro PCO | `15` |
| `PCD` | `random_state` | Semilla de PCO | `None` |
| `PCD.apply` | `n_remove` | `"diff"`, `"PR"`, `int` (n primeras), `float` (MVL > umbral), índices o `None` | `"diff"` |
| `estimate_safb` | `f0` | Frecuencia fundamental (Hz); si es un arreglo, se usa la media | — |
| `estimate_safb` | `gamma_band` | Rango donde se busca el pico del artefacto (Hz) | `(50, 250)` |
| `estimate_safb`, `fit_ssd`, `PCD` | `noise_band` | Band-pass de ruido de SSD; la SAFB debe quedar adentro | `(4, 240)` |
| `fit_ssd`, `PCD` | `filter_order` | Orden de los Butterworth (band-pass resultante: 2×) | `5` |
| `select_n_components` | `rule` | `"PR"`, `int` (k fijo) o `float` en (0, 1) (fracción acumulada de λ) | `"PR"` |

Si la banda estimada no entra en `noise_band`, `estimate_safb` usa una banda de respaldo de 40 Hz centrada en F0 (o, si tampoco entra, en 120 Hz) y emite un `UserWarning`.

---

## Prueba rápida (smoke test)

Una fuente sinusoidal de 130 Hz acoplada al "audio" y mezclada en 8 canales con ruido de banda ancha:

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.pcd import PCD

sfreq, n_samples = 1000.0, 6000
rng = np.random.default_rng(1)
source = np.sin(2 * np.pi * 130 * np.arange(n_samples) / sfreq)
noise = sosfiltfilt(butter(4, [4, 240], btype="band", fs=sfreq, output="sos"),
                    rng.standard_normal((8, n_samples)))
X = 0.5 * np.outer(rng.standard_normal(8), source) + noise
z = source + 0.1 * rng.standard_normal(n_samples)

pcd = PCD(sfreq, signal_band=(110, 150), random_state=0).fit(X, z)
print(pcd.lambda_ssd_.round(3))     # el primer λ se separa claramente del resto
print(pcd.vlen_.round(3))           # la primera componente tiene el MVL más alto
X_limpia = pcd.apply(X)
```

---

## Datos

Los datos son los del repositorio de los autores ([Brain-Modulation-Lab/PCD](https://github.com/Brain-Modulation-Lab/PCD)) y no se versionan. Hay que copiarlos a `data/`:

| Archivo | Contenido |
|---|---|
| `Data.mat` | Un trial real: 126 canales ECoG + 16 DBS, audio, `X_tofit` (ventana del habla) y `X_toclean` (ventana más larga) |
| `DataTrials.mat` | 20 trials reales de dos canales, para el ITPC |
| `Audio.mat`, `Audio_epochs.txt` | Audios de habla del dataset de Bush et al. (2022) y el inicio y fin del habla de cada uno, para la simulación |

Las referencias se generaron corriendo el código original en MATLAB y van en `data/reference/`:

| Archivo | Contenido | Lo usa |
|---|---|---|
| `matlab_reference.mat` | Salida sobre `Data.mat`: SAFB, M, W_ssd, λ, W_pco, MVL, W_pcd, A_pcd, X_clean | `pcd-validar` |
| `matlab_safb_trials.mat` | SAFB de los 20 trials de `DataTrials.mat` | `pcd-validar` |
| `matlab_itpc_pcd.mat` | ITPC antes y después de PCD en `DataTrials.mat` | `pcd-validar` |
| `matlab_simulation.mat` | W_pcd, A_pcd y componentes removidas en cada trial simulado | `pcd-simular` |

`pcd-simular` también escribe `data/simulation/sim_fit.mat`, con la ventana del habla, el audio y la F0 de cada trial simulado, que es lo que lee el código original para generar `matlab_simulation.mat`.

---

## Validación

### Datos reales (`pcd-validar`)

Compara cada etapa contra la implementación de Peterson et al (2024) en MATLAB sobre `Data.mat`. Python estima su propia SAFB y su propio k, sin fijar nada de la referencia:

| Etapa | Resultado |
|---|---|
| SAFB | Misma banda, [106, 122] Hz (Fc 114.30 Hz en ambas; FWHM 12.34 vs 12.29 Hz). En `DataTrials.mat`, 19 de 20 trials con la misma banda |
| k (participation ratio) | 16 en ambas |
| SSD | Subespacio de los 16 filtros: cos mediano 0.9992. λ: diferencia mediana 1 % (máx. 3.3 %). Los filtros 1–7 coinciden (\|cos\| ≥ 0.976) |
| PCO | Mismo objetivo: la función de Python recalcula el MVL de los filtros de MATLAB (0.161, 0.158, …). Python llega a otro máximo local, más alto (0.177, 0.171) |
| Reconstrucción con los filtros de MATLAB | W_pcd 3·10⁻¹⁶, A_pcd 4·10⁻¹⁴, X_clean 3·10⁻¹⁴ (error relativo) |
| Punta a punta | Ambas sacan 2 componentes; la señal removida se correlaciona 0.59 (mediana por canal) y su norma es 0.83 de la de MATLAB |
| ITPC en `DataTrials.mat` | Antes de limpiar, idéntico (error 6·10⁻¹⁶); mismas componentes sacadas en los 20 trials. En la ventana del habla: 3.02 / 2.58 antes; 2.61 / 1.66 (Python) y 2.74 / 1.61 (MATLAB) después |

Las diferencias en SSD vienen de cómo se arma la banda de ruido (MNE resta el filtrado de la señal; el código original usa band-pass + band-stop) y del padding de los filtros. Las de PCO y de punta a punta vienen de que el problema no es convexo y cada implementación arranca de puntos al azar.

Las funciones de `validation/metrics.py` dan lo mismo que sus versiones en MATLAB (`mscohere`, `circ_r`, `pca`, `get_coherence` y `CoH2ITPC`), con errores relativos menores a 10⁻¹³. El ITPC sigue al código de los autores, que divide por el error estándar (std/√N_t) y no por el desvío. En datos reales no hay ground truth, así que ahí solo se usa el ITPC.

### Simulación (`pcd-simular`)

Como las simulaciones del paper no están disponibles de forma pública se procedió a generar una simulación propia siguiendo el repositorio original (`Example_run_HybridSimulation.m`):

- 20 trials con 49 fuentes neuronales de ruido 1/f (2–250 Hz). En 10 de ellas se suma gamma alta (70–180 Hz) durante el habla, para que el artefacto se solape con actividad fisiológica.
- La fuente artefactual es un audio real de `Audio.mat`, distinto en cada trial.
- Todo se mezcla en 30 canales con una matriz aleatoria fija, como `generate_mixingmatrix.m`. `X_gt` es la misma mezcla sin el audio.
- La relación artefacto/señal neuronal se sortea entre −5 y 5 dB, medida en los canales en 70–180 Hz durante el habla.
- PCD (Python y referencia) estima su propia SAFB y su propio k en cada trial; CAR se usa como comparación.
- χ y CS se calculan en 70–250 Hz durante el habla, donde está el artefacto. Las métricas de la fuente se calculan durante el habla.

Mediana sobre los 20 trials:

| | Sin limpiar | CAR | PCD MATLAB | PCD Python |
|---|---|---|---|---|
| χ | 0.61 | 0.67 | 0.15 | 0.03 |
| CS (promedio de las cargas 1–3) | 0.09 | 0.14 | 0.78 | 0.94 |
| MSE normalizado de la fuente | — | — | 8.0·10⁻⁴ | 8.4·10⁻⁴ |
| MSCE en la SAFB | — | — | 0.997 | 0.997 |
| PLV en la SAFB | — | — | 0.984 | 0.982 |

Las dos implementaciones recuperan la fuente del artefacto con la misma fidelidad, del mismo orden que lo publicado (MSE 1.5·10⁻³, MSCE > 0.97), y preservan la señal neuronal mucho mejor que CAR. La diferencia en χ y CS entre ambas viene del número de componentes que saca cada una (MATLAB 1 o 3, Python entre 1 y 7). Como la simulación no es la del paper, la comparación con los números publicados es solo cualitativa.

---

## Referencias

- Peterson, V., Vissani, M., Luo, S., Rabbani, Q., Crone, N. E., Bush, A., & Richardson, R. M. (2024). A supervised data-driven spatial filter denoising method for acoustic-induced artifacts in intracranial electrophysiological recordings. *Imaging Neuroscience*, 2, 1–22. [doi:10.1162/imag_a_00301](https://doi.org/10.1162/imag_a_00301)
- Bush, A., Chrabaszcz, A., Peterson, V., Saravanan, V., Dastolfo-Hromack, C., Lipski, W. J., & Richardson, R. M. (2022). Differentiation of speech-induced artifacts from physiological high gamma activity in intracranial recordings. *NeuroImage*, 250, 118962.
