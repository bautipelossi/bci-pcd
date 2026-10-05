# PCD — Phase Coupling Decomposition para iEEG
Bautista Pelossi Schweizer

**Interfaces Cerebro-Computadora (BCI)** · Facultad de Ingeniería Química, Universidad Nacional del Litroal 

Implementación en Python de **Phase Coupling Decomposition (PCD)** (Peterson et al., 2024), un filtrado espacial supervisado que remueve el artefacto acústico del habla en registros intracraneales (iEEG). Tiene cuatro etapas: estimación de la banda del artefacto (**SAFB**), descomposición espacio-espectral (**SSD**), optimización del acoplamiento de fase (**PCO**) y reconstrucción de la señal limpia.

> **Estado:** en desarrollo. Implementados: `safb`, `ssd`. Pendientes: `pco`, `pcd`.

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
                                        PCO                                (pendiente)
                                        └─ max MVL(w) con z → W_pco, vlen
                                            │
                                            ▼
                                        PCD.fit / PCD.apply                (pendiente)
                                        ├─ W_pcd = M · W_ssd · blockdiag(W_pco, I)
                                        ├─ A_pcd = pinv(W_pcd)ᵀ
                                        └─ X_limpia = A_pcd[:, keep] · W_pcd[:, keep]ᵀ · X
```

**Decisiones de diseño:**

- **Convención de formas**: los datos son siempre canales × muestras `(N_c, N_s)`. Los filtros y patrones van en las **columnas** (`X = A S`, `S = Wᵀ X`).
- **Hilbert antes del blanqueo**: la misma matriz `M` blanquea la parte real (que usa SSD) y conserva la fase (que usa PCO).
- **Rango completo obligatorio**: `whiten` falla con un error claro si los datos tienen rango deficiente (por ejemplo, después de una CAR) y sugiere quitar un canal.
- **SSD con MNE**: `mne.decoding.SSD` con Butterworth IIR de orden 5, sin regularización (`reg=None`), sin restricción de rango (`restr_type=None`) y ordenado por autovalor (`sort_by_spectral_ratio=False`).
- **Autovalores normalizados**: MNE devuelve el cociente señal/ruido λ ∈ (0, ∞). Se usa λ / (1 + λ) ∈ (0, 1), que es la fracción de potencia en la SAFB, para calcular el participation ratio.
- **MVL como objetivo de PCO**: `MVL(w) = |(1/N_s) Σ_t z(t) · exp(i · angle(wᵀ x(t)))|`, con `z` normalizado (media 0, varianza 1) y varias inicializaciones aleatorias.

---

## Estructura del proyecto

```
bci-pcd/
├── data/                  ← datos simulados (.mat)
├── notebooks/             ← ejemplos
├── src/
│   ├── safb.py            ← estimación de la banda del artefacto (PSD + ajuste gaussiano)
│   ├── ssd.py             ← blanqueo, SSD (MNE) y elección de k
│   ├── pco.py             ← optimización de MVL                    (pendiente)
│   ├── pcd.py             ← clase PCD: fit / transform / apply     (pendiente)
│   └── __init__.py
├── tests/                 ← tests con pytest
├── validation/            ← comparación contra la implementación MATLAB
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

Por ahora el uso es como módulos de Python; la clase `PCD` está pendiente.

```python
from src.safb import estimate_safb
from src.ssd import whiten, fit_ssd, select_n_components

# X: (N_c, N_s) iEEG · z: (N_s,) audio · sfreq en Hz · f0: F0 en Hz
fl, fh = estimate_safb(z, sfreq, f0)

X_white, M = whiten(X)
W_ssd, lambda_ssd = fit_ssd(X_white.real, sfreq, signal_band=(fl, fh))
k = select_n_components(lambda_ssd, rule="PR")

X_ssd = W_ssd[:, :k].T @ X_white   # (k, N_s) complejo, entrada de PCO
```

Scripts y tests dentro del entorno:

```bash
uv run python script.py
uv run pytest
```

---

## Parámetros

| Función | Parámetro | Descripción | Default |
|---|---|---|---|
| `estimate_safb` | `f0` | Frecuencia fundamental (Hz); si es un arreglo, se usa la media | — |
| `estimate_safb` | `gamma_band` | Rango donde se busca el pico del artefacto (Hz) | `(50, 250)` |
| `estimate_safb` | `noise_band` | Band-pass de ruido de SSD; la SAFB debe quedar adentro | `(4, 240)` |
| `fit_ssd` | `signal_band` | SAFB `(fl, fh)` (Hz) | — |
| `fit_ssd` | `noise_band` | Band-pass de ruido (Hz) | `(4, 240)` |
| `fit_ssd` | `filter_order` | Orden de los Butterworth (band-pass resultante: 2×) | `5` |
| `select_n_components` | `rule` | `"PR"`, `int` (k fijo) o `float` en (0, 1) (fracción acumulada de λ) | `"PR"` |

Si la banda estimada no entra en `noise_band`, `estimate_safb` usa una banda de respaldo de 40 Hz centrada en F0 (o, si tampoco entra, en 120 Hz) y emite un `UserWarning`.

---

## Prueba rápida (smoke test)

Una fuente sinusoidal de 130 Hz mezclada en 8 canales con ruido de banda ancha:

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.ssd import whiten, fit_ssd, select_n_components

sfreq, n_samples = 1000.0, 6000
rng = np.random.default_rng(1)
source = np.sin(2 * np.pi * 130 * np.arange(n_samples) / sfreq)
noise = sosfiltfilt(butter(4, [4, 240], btype="band", fs=sfreq, output="sos"),
                    rng.standard_normal((8, n_samples)))
X = 0.5 * np.outer(rng.standard_normal(8), source) + noise

X_white, M = whiten(X)
W_ssd, lambda_ssd = fit_ssd(X_white.real, sfreq, signal_band=(110, 150))
print(lambda_ssd.round(3))          # el primer λ se separa claramente del resto
print(select_n_components(lambda_ssd))
```

---

## Datos

Los `.mat` de `data/` se generan con el toolkit de simulación del repositorio original y se leen con `pymatreader.read_mat`. Al leerlos:

- MATLAB guarda los datos en tiempo × canales; transponer a `(N_c, N_s)`.
- Los índices de MATLAB empiezan en 1; restar 1 antes de usarlos en Python.

Entradas que necesita el método:

| Variable | Forma | Descripción |
|---|---|---|
| `X` | `(N_c, N_s)` | Registro iEEG para ajustar el modelo |
| `z` | `(N_s,)` | Audio registrado, sincronizado con `X` |
| `sfreq` | escalar | Frecuencia de muestreo (Hz) |
| `f0` | escalar o arreglo | Frecuencia fundamental anotada (Hz) |

---

## Validación

Se compara contra la implementación MATLAB de los autores por **métricas**, no elemento a elemento: los filtros SSD tienen signo y escala arbitrarios, PCO depende de la inicialización y el SSD de MNE define el ruido distinto.

- Similitud coseno (en valor absoluto) entre patrones.
- MVL alcanzado por componente.
- MSE normalizado y correlación de la señal reconstruida.

---

## Referencias

- Peterson, V., Vissani, M., Luo, S., Rabbani, Q., Crone, N. E., Bush, A., & Richardson, R. M. (2024). A supervised data-driven spatial filter denoising method for acoustic-induced artifacts in intracranial electrophysiological recordings. *Imaging Neuroscience*, 2, 1–22. [doi:10.1162/imag_a_00301](https://doi.org/10.1162/imag_a_00301)
