"""Phase-Coupling Optimization (PCO).
 Tercera etapa de PCD
 Bautista Pelossi Schweizer
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


def _negative_mvl(w: np.ndarray, X: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    """-MVL de la componente ``w.T @ X`` y su gradiente respecto de ``w``.

    Parameters
    ----------
    w : np.ndarray, forma (d,)
        Filtro espacial (real).
    X : np.ndarray complejo, forma (d, N_s)
        Señal analítica en el espacio de búsqueda.
    z : np.ndarray, forma (N_s,)
        Audio normalizado (media 0, varianza 1).

    Returns
    -------
    float
        -MVL, negativo porque ``minimize`` minimiza.
    np.ndarray, forma (d,)
        Gradiente de -MVL.
    """
    component = w @ X
    phase_vector = np.exp(1j * np.angle(component))
    # vector medio c = mean(z · e^{iφ}); el MVL es su módulo
    mean_vector = np.mean(z * phase_vector)
    mvl = np.abs(mean_vector)

    # derivada de la fase respecto de cada w_i: Im(x_i · conj(wᵀx)) / |wᵀx|²
    phase_gradient = np.imag(X * component.conj()) / np.abs(component) ** 2
    # dc/dw = mean(i · z · e^{iφ} · dφ/dw) y d|c|/dw = Re(conj(c) · dc/dw) / |c|
    mean_vector_gradient = np.mean(1j * z * phase_vector * phase_gradient, axis=1)
    mvl_gradient = np.real(mean_vector.conj() * mean_vector_gradient) / mvl

    return -mvl, -mvl_gradient


def _best_filter(
    X: np.ndarray, z: np.ndarray, n_restarts: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Maximiza el MVL desde varias inicializaciones y se queda con la mejor.

    El MVL no es convexo en ``w``, así que una sola corrida puede quedar en un
    máximo local.
    """
    best_w, best_mvl = None, -np.inf
    for _ in range(n_restarts):
        w0 = rng.uniform(-1, 1, size=X.shape[0])
        result = minimize(_negative_mvl, w0, args=(X, z), jac=True, method="BFGS")
        if -result.fun > best_mvl:
            best_w, best_mvl = result.x, -result.fun
    return best_w, best_mvl


def fit_pco(
    X_ssd: np.ndarray,
    z: np.ndarray,
    n_restarts: int = 15,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Busca los filtros cuya fase está más acoplada al audio.

    Maximiza el mean vector length (MVL) entre la fase de cada componente
    ``wᵀ x(t)`` y la amplitud del audio ``z``::

        MVL(w) = | (1/N_s) Σ_t z(t) · wᵀx(t) / |wᵀx(t)| |

    Los filtros se sacan de a uno: cada filtro nuevo se busca en el complemento
    ortogonal de los anteriores, para no volver a encontrar la misma fuente.

    Parameters
    ----------
    X_ssd : np.ndarray complejo, forma (k, N_s)
        Señal analítica blanqueada proyectada en las k primeras componentes SSD.
    z : np.ndarray, forma (N_s,)
        Audio registrado, alineado con ``X_ssd``.
    n_restarts : int, default 15
        Inicializaciones aleatorias por filtro (el paper usa entre 10 y 15).
    random_state : int, Generator or None, default None
        Semilla para las inicializaciones.

    Returns
    -------
    W_pco : np.ndarray, forma (k, k)
        Filtros PCO en las columnas, ordenados por MVL descendente. Las
        componentes se obtienen con ``W_pco.T @ X_ssd``.
    vlen : np.ndarray, forma (k,)
        MVL de cada filtro, descendente.
    """
    rng = np.random.default_rng(random_state)
    # NaN a 0 antes de normalizar, si no la media y el desvío dan NaN
    z = np.nan_to_num(z)
    z = (z - z.mean()) / z.std()
    n_components = X_ssd.shape[0]

    filters, vlen = [], []
    for _ in range(n_components):
        # base ortonormal del espacio que todavía no cubren los filtros encontrados
        if filters:
            basis = null_space(np.column_stack(filters).T)
        else:
            basis = np.eye(n_components)
        w, mvl = _best_filter(basis.T @ X_ssd, z, n_restarts, rng)
        # volvemos el filtro de la base reducida al espacio SSD completo
        filters.append(basis @ w)
        vlen.append(mvl)

    # la deflación no garantiza que cada filtro tenga menos MVL que el anterior
    order = np.argsort(vlen)[::-1]
    return np.column_stack(filters)[:, order], np.array(vlen)[order]
