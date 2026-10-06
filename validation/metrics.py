"""Métricas para validar PCD contra una referencia o un ground truth.
 Bautista Pelossi Schweizer
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import coherence, hilbert


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


def msce_in_band(
    true: np.ndarray, estimated: np.ndarray, sfreq: float, band: tuple[float, float]
) -> float:
    """Coherencia cuadrática media entre dos señales, promediada en ``band``.

    Usamos los parámetros por defecto de ``mscohere``: ventana de Hamming, 8
    segmentos con 50 % de solapamiento.
    """
    nperseg = int(len(true) // 4.5)
    freqs, msc = coherence(true, estimated, fs=sfreq, window="hamming", nperseg=nperseg)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return float(msc[in_band].mean())


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
