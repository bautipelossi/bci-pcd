"""Métricas para validar PCD contra una referencia o un ground truth.
 Bautista Pelossi Schweizer
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import coherence, hilbert
from scipy.signal.windows import hamming


def abs_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|cos| entre columnas de ``a`` y ``b`` (o entre dos vectores).

    En valor absoluto porque los filtros y patrones tienen signo arbitrario.
    """
    a, b = np.atleast_2d(a.T).T, np.atleast_2d(b.T).T
    cos = np.sum(a * b, axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    return np.abs(cos)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosenos de los ángulos principales entre los subespacios de las columnas.

    Valen 1 si los dos generan el mismo subespacio, aunque las columnas estén
    rotadas o en otro orden.
    """
    return np.cos(subspace_angles(A, B))[::-1]


def mean_vector_length(component: np.ndarray, z: np.ndarray) -> float:
    """MVL entre la fase de una componente real y el audio normalizado."""
    z = (z - z.mean()) / z.std()
    analytic = hilbert(component - component.mean())
    return float(np.abs(np.mean(z * np.exp(1j * np.angle(analytic)))))


def channel_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Correlación de Pearson canal por canal entre ``X`` e ``Y``, forma (N_c, N_s)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return np.sum(Xc * Yc, axis=1) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(Yc, axis=1))


def normalized_mse(true: np.ndarray, estimated: np.ndarray) -> float:
    """MSE entre dos señales divididas cada una por su máximo absoluto.

    La fuente estimada puede salir en contrafase, así que primero alineamos el
    signo con el de la verdadera.
    """
    if np.corrcoef(true, estimated)[0, 1] < 0:
        estimated = -estimated
    return float(np.mean((true / np.abs(true).max() - estimated / np.abs(estimated).max()) ** 2))


def msce(true: np.ndarray, estimated: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Coherencia cuadrática (MSCE) entre la fuente verdadera y la estimada.

    Returns
    -------
    freqs : np.ndarray
    msc : np.ndarray
        Coherencia en cada frecuencia, entre 0 y 1.

    Notes
    -----
    Usamos los defaults de ``mscohere``, que es con lo que se calculó en el
    paper: 8 segmentos de Hamming con 50 % de solapamiento, sin quitar la
    media de cada segmento y con nfft = max(256, potencia de 2 siguiente).
    """
    nperseg = int(len(true) // 4.5)
    nfft = max(256, 2 ** int(np.ceil(np.log2(nperseg))))
    return coherence(
        true,
        estimated,
        fs=sfreq,
        window=hamming(nperseg, sym=True),
        noverlap=nperseg // 2,
        nfft=nfft,
        detrend=False,
    )


def msce_in_band(
    true: np.ndarray, estimated: np.ndarray, sfreq: float, band: tuple[float, float]
) -> float:
    """MSCE promediada en ``band`` (el paper la reporta en la SAFB)."""
    freqs, msc = msce(true, estimated, sfreq)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return float(msc[in_band].mean())


def plv(true: np.ndarray, estimated: np.ndarray) -> float:
    """Phase-locking value entre dos señales reales.

    Es el módulo del promedio de e^{i(φ_true - φ_est)}, con las fases sacadas
    de la señal analítica. Una señal en contrafase da el mismo PLV.
    """
    phase_difference = np.angle(hilbert(true)) - np.angle(hilbert(estimated))
    return float(np.abs(np.mean(np.exp(1j * phase_difference))))


def chi_per_channel(X_clean: np.ndarray, X_gt: np.ndarray) -> np.ndarray:
    """Error temporal normalizado de cada canal (χ, ec. 19 del paper).

    Promedio en el tiempo de (X - X_gt)² dividido por la varianza de X_gt,
    así se pueden comparar canales con distinta amplitud.

    Parameters
    ----------
    X_clean, X_gt : np.ndarray, forma (N_c, N_s)

    Returns
    -------
    np.ndarray, forma (N_c,)
    """
    return np.mean((X_clean - X_gt) ** 2, axis=1) / np.var(X_gt, axis=1, ddof=1)


def trial_coherence(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Coherencia compleja entre cada canal y el audio en un trial (ec. 23).

    Parameters
    ----------
    X : np.ndarray, forma (N_c, N_s)
    z : np.ndarray, forma (N_s,)

    Returns
    -------
    np.ndarray complejo, forma (N_c,)
        Su argumento es el desfase entre el canal y el audio.

    Notes
    -----
    Aplicamos Hilbert al audio y conjugamos, como el código de los autores.
    Da lo mismo que tomar la señal analítica del canal (ec. 23): Hilbert es
    antisimétrico, así que Σ x·conj(H{z}) = Σ (x + iH{x})·z. Solo cambia la
    normalización en un factor √2, que se cancela en el ITPC.
    """
    X = X - X.mean(axis=1, keepdims=True)
    z = z - z.mean()
    z_analytic = hilbert(z)
    return (X @ z_analytic.conj()) / (np.linalg.norm(X, axis=1) * np.linalg.norm(z))


def itpc(coherences: np.ndarray) -> np.ndarray:
    """Coherencia de fase entre trials (ITPC) de cada canal.

    Parameters
    ----------
    coherences : np.ndarray complejo, forma (N_t, N_c)
        :func:`trial_coherence` de cada trial.

    Returns
    -------
    np.ndarray, forma (N_c,)

    Notes
    -----
    Dividimos el módulo del promedio por el error estándar (std / √N_t) de
    la parte real e imaginaria, como el código de los autores, que es con lo
    que se calibró el umbral de 3.08. La ec. 24 del paper divide por el
    desvío, que da un valor √N_t veces menor.
    """
    n_trials = coherences.shape[0]
    mean = coherences.mean(axis=0)
    standard_error_sq = (
        np.var(coherences.real, axis=0, ddof=1) + np.var(coherences.imag, axis=0, ddof=1)
    ) / n_trials
    return np.abs(mean) / np.sqrt(standard_error_sq)


def pca_loading_similarity(X_true: np.ndarray, X_est: np.ndarray, n_components: int = 3) -> np.ndarray:
    """|cos| entre los loadings de PCA de dos registros, componente a componente.

    Parameters
    ----------
    X_true, X_est : np.ndarray, forma (N_c, N_s)
    n_components : int, default 3
        Cuántos loadings comparar (el paper usa los 3 primeros).

    Returns
    -------
    np.ndarray, forma (n_components,)
    """
    def loadings(X):
        # los canales son las variables: los loadings son los autovectores de cov(X)
        _, _, Vt = np.linalg.svd((X - X.mean(axis=1, keepdims=True)).T, full_matrices=False)
        return Vt[:n_components].T

    return abs_cosine(loadings(X_true), loadings(X_est))
