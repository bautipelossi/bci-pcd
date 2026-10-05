"""Blanqueo y Spatio-Spectral Decomposition (SSD), primer paso de PCD.

References
----------
Nikulin, V. V., Nolte, G., & Curio, G. (2011). A novel method for reliable and fast
extraction of neuronal EEG/MEG oscillations on the basis of spatio-spectral
decomposition. NeuroImage, 55(4), 1528-1535.

Peterson, V., et al. (2024). A supervised data-driven spatial filter denoising method
for acoustic-induced artifacts in intracranial electrophysiological recordings.
Imaging Neuroscience, 2, 1-22. doi:10.1162/imag_a_00301
"""

from __future__ import annotations

import mne
import numpy as np
from mne.decoding import SSD
from scipy.signal import hilbert


def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula la señal analítica, la centra y la blanquea.

    Parameters
    ----------
    X : np.ndarray, forma (N_c, N_s)
        Señal real, canales × muestras.

    Returns
    -------
    X_white : np.ndarray complejo, forma (N_c, N_s)
        Señal analítica centrada y blanqueada: ``M.T @ (X_analytic - media)``.
        Su parte real tiene covarianza identidad.
    M : np.ndarray, forma (N_c, N_c)
        Matriz de blanqueo, con los filtros en las columnas: autovectores de
        ``cov(X)`` ordenados por autovalor descendente y escalados por
        autovalor^(-1/2).

    Raises
    ------
    ValueError
        Si ``cov(X)`` tiene rango deficiente (por ejemplo, después de una
        referencia promedio). En ese caso no existe una matriz de blanqueo
        invertible y SSD no podría devolver N_c filtros.

    Notes
    -----
    La transformada de Hilbert va antes del blanqueo para que la misma ``M``
    sirva para la parte real (que usa SSD) y para la fase (que usa PCO). Como la
    parte real de la señal analítica es la señal original, ``M`` se estima a
    partir de ``X`` centrada.

    Para el rango se usa la misma tolerancia que ``np.linalg.matrix_rank``
    (autovalor máximo · N_c · eps).
    """
    X_analytic = hilbert(X, axis=-1)
    X_analytic -= X_analytic.mean(axis=-1, keepdims=True)

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X_analytic.real))
    # eigh los devuelve en orden ascendente.
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    tolerance = eigenvalues[0] * X.shape[0] * np.finfo(float).eps
    rank = int(np.sum(eigenvalues > tolerance))
    if rank < X.shape[0]:
        raise ValueError(
            f"Los datos tienen rango {rank} con {X.shape[0]} canales, así que no "
            "se pueden blanquear. Suele pasar después de una referencia común "
            "(CAR); quitá un canal por cada dimensión perdida."
        )

    M = eigenvectors / np.sqrt(eigenvalues)
    return M.T @ X_analytic, M


def fit_ssd(
    X: np.ndarray,
    sfreq: float,
    signal_band: tuple[float, float],
    noise_band: tuple[float, float] = (4.0, 240.0),
    filter_order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula los filtros SSD que maximizan la potencia en la SAFB.

    Parameters
    ----------
    X : np.ndarray, forma (N_c, N_s)
        Señal real blanqueada (``whiten(X)[0].real``).
    sfreq : float
        Frecuencia de muestreo en Hz.
    signal_band : tuple of float
        SAFB ``(fl, fh)`` en Hz.
    noise_band : tuple of float, default (4.0, 240.0)
        Band-pass de ruido en Hz.
    filter_order : int, default 5
        Orden de los Butterworth. Al ser band-pass, el filtro resultante es de
        orden 2·``filter_order`` y se aplica con fase cero (ida y vuelta).

    Returns
    -------
    W_ssd : np.ndarray, forma (N_c, N_c)
        Filtros SSD en las columnas, ordenados por autovalor descendente. Las
        componentes se obtienen con ``W_ssd.T @ X``.
    lambda_ssd : np.ndarray, forma (N_c,)
        Autovalores normalizados en (0, 1), descendentes.

    Notes
    -----
    - **Ruido.** ``mne.decoding.SSD`` define el ruido como
      ``filter(noise_band) - filter(signal_band)``, es decir, los flancos
      alrededor de la SAFB dentro de ``noise_band``.
    - **Autovalores.** MNE resuelve ``eigh(C_s, C_n)``, que da el cociente
      señal/ruido λ ∈ (0, ∞). Se devuelve λ / (1 + λ), que es la fracción de
      potencia en la banda de señal, ``w' C_s w / w' (C_s + C_n) w``. Al quedar
      acotado, el participation ratio no lo dominan unas pocas componentes con
      λ muy grande.
    - **Sin regularización ni restricción.** ``reg=None`` usa la covarianza
      empírica y ``restr_type=None`` evita proyectar a un subespacio de menor
      rango. El rango completo ya lo garantiza :func:`whiten`.
    - **Orden.** ``sort_by_spectral_ratio=False`` mantiene el orden por
      autovalor, en lugar de reordenar por el cociente espectral de la PSD.
    """
    iir_params = {"order": filter_order, "ftype": "butter"}
    ssd = SSD(
        info=mne.create_info(X.shape[0], sfreq, ch_types="eeg"),
        filt_params_signal={
            "l_freq": signal_band[0],
            "h_freq": signal_band[1],
            "method": "iir",
            "iir_params": dict(iir_params),
        },
        filt_params_noise={
            "l_freq": noise_band[0],
            "h_freq": noise_band[1],
            "method": "iir",
            "iir_params": dict(iir_params),
        },
        reg=None,
        rank="full",
        restr_type=None,
        sort_by_spectral_ratio=False,
    )
    with mne.use_log_level("warning"):
        ssd.fit(X)

    snr = ssd.evals_
    return ssd.filters_.T, snr / (1 + snr)


def participation_ratio(eigenvalues: np.ndarray) -> int:
    """Número efectivo de componentes según el participation ratio.

    Calcula ``round(sum(λ)² / sum(λ²))``: vale N_c si todos los autovalores son
    iguales y 1 si uno solo domina.

    Parameters
    ----------
    eigenvalues : np.ndarray, forma (N_c,)
        Autovalores SSD (``lambda_ssd``).

    Returns
    -------
    int
        k redondeado al entero más cercano.
    """
    return round(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))


def select_n_components(lambda_ssd: np.ndarray, rule: int | float | str = "PR") -> int:
    """Elige k, la cantidad de componentes SSD que pasan a PCO.

    Parameters
    ----------
    lambda_ssd : np.ndarray, forma (N_c,)
        Autovalores SSD descendentes.
    rule : int, float or "PR", default "PR"
        - ``"PR"``: participation ratio (:func:`participation_ratio`).
        - ``int``: k fijo.
        - ``float`` en (0, 1): k cuya fracción acumulada de autovalores
          ``cumsum(λ) / sum(λ)`` queda más cerca de ``rule``.

    Returns
    -------
    int
        k, entre 1 y N_c.
    """
    if rule == "PR":
        return participation_ratio(lambda_ssd)
    if isinstance(rule, float):
        cumulative = np.cumsum(lambda_ssd) / lambda_ssd.sum()
        # +1 porque k cuenta componentes y argmin devuelve un índice.
        return int(np.argmin(np.abs(cumulative - rule))) + 1
    return rule
