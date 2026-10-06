"""Estimación de la banda de frecuencia del artefacto acústico (SAFB).
Primera etapa de PCD
Bautista Pelossi Schweizer

La SAFB (speech artifact frequency band) es la banda [fl, fh] alrededor de la
frecuencia fundamental F0 del habla donde se concentra el artefacto. Se estima
ajustando una gaussiana al pico del espectro del audio ``z`` cercano a F0.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.signal import welch

# Ancho (Hz) de las bandas de respaldo
FALLBACK_BANDWIDTH = 40.0
# Centro (Hz) de la última banda de respaldo, típico de F0 en voz adulta.
FALLBACK_F0 = 120.0


def estimate_audio_psd(z: ArrayLike, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Estima la PSD del audio con Welch, en segmentos de 200 ms.

    Parameters
    ----------
    z : array_like, forma (N_s,)
        Señal de audio registrada.
    sfreq : float
        Frecuencia de muestreo en Hz.

    Returns
    -------
    freqs : np.ndarray, forma (N_f,)
        Frecuencias en Hz.
    psd : np.ndarray, forma (N_f,)
        PSD unilateral (densidad).

    Notes
    -----
    El ``nfft`` (5 veces la potencia de 2 siguiente al largo del segmento)
    rellena con ceros para tener una grilla de frecuencias fina (~0.78 Hz a
    1 kHz) y así ajustar mejor la gaussiana.
    """
    n_per_segment = round(0.2 * sfreq)
    nfft = 5 * 2 ** math.ceil(math.log2(n_per_segment))
    return welch(np.ravel(z), fs=sfreq, nperseg=n_per_segment, nfft=nfft)


def _gaussian(freqs: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Gaussiana de altura 1 parametrizada por su FWHM (ancho a media altura)."""
    return np.exp(-4 * np.log(2) * ((freqs - center) / fwhm) ** 2)


def _fit_error(params: np.ndarray, freqs: np.ndarray, psd: np.ndarray) -> float:
    """Error del ajuste para un centro y un ancho dados.

    La altura no la busca el optimizador: para cada (centro, ancho) la mejor
    altura sale cerrada por mínimos cuadrados, así Nelder-Mead tiene un
    parámetro menos.
    """
    shape = _gaussian(freqs, *params)
    height = shape @ psd / (shape @ shape) if shape.any() else 0.0
    error = np.linalg.norm(height * shape - psd)
    # una altura negativa ajustaría un valle, no un pico
    return error + 1e6 if height < 0 else error


def fit_gaussian_peak(
    freqs: ArrayLike,
    psd: ArrayLike,
    center0: float,
    fwhm0: float = 40.0,
) -> tuple[float, float]:
    """Ajusta una gaussiana al pico espectral del artefacto.

    Parameters
    ----------
    freqs : array_like, forma (N_f,)
        Frecuencias (Hz) del tramo de espectro a ajustar.
    psd : array_like, forma (N_f,)
        PSD en esas frecuencias.
    center0 : float
        Centro inicial en Hz (F0).
    fwhm0 : float, default 40.0
        FWHM inicial en Hz.

    Returns
    -------
    center : float
        Centro de la gaussiana ajustada (Hz).
    fwhm : float
        FWHM de la gaussiana ajustada (Hz).

    Notes
    -----
    Usamos Nelder-Mead arrancando en (``center0``, ``fwhm0``) porque se queda
    en el pico más cercano a F0. Con un paso inicial más grande, o con
    mínimos cuadrados no lineales, el ajuste puede saltar al armónico (2·F0) o
    no converger.
    """
    result = minimize(
        _fit_error,
        x0=[center0, fwhm0],
        args=(np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 100_000, "maxfev": 10**12},
    )
    center, fwhm = np.abs(result.x)
    # la gaussiana depende de fwhm², así que el signo no importa
    return float(center), float(fwhm)


def _band_around(center: float, bandwidth: float) -> tuple[int, int]:
    """Banda entera [fl, fh] de ancho ``bandwidth`` centrada en ``center``."""
    half = math.ceil(bandwidth / 2)
    return round(center - half), round(center + half)


def _is_valid_band(fl: int, fh: int, noise_band: tuple[float, float]) -> bool:
    """Verifica que la SAFB, con 1 Hz de margen, quede dentro de la banda de ruido.

    SSD necesita flancos de ruido a ambos lados de la banda de señal.
    """
    return noise_band[0] < fl - 1 and fh + 1 < noise_band[1]


def estimate_safb(
    z: ArrayLike,
    sfreq: float,
    f0: float | ArrayLike,
    gamma_band: tuple[float, float] = (50.0, 250.0),
    noise_band: tuple[float, float] = (4.0, 240.0),
) -> tuple[int, int]:
    """Estima la SAFB [fl, fh] a partir del audio y la F0.

    Parameters
    ----------
    z : array_like, forma (N_s,)
        Señal de audio registrada.
    sfreq : float
        Frecuencia de muestreo en Hz.
    f0 : float or array_like
        Frecuencia fundamental en Hz. Si es un arreglo (por ejemplo, una
        anotación por sílaba), se usa su media.
    gamma_band : tuple of float, default (50.0, 250.0)
        Rango (abierto) donde se busca el pico.
    noise_band : tuple of float, default (4.0, 240.0)
        Band-pass de ruido de SSD; la SAFB tiene que quedar adentro.

    Returns
    -------
    fl, fh : int
        Límites inferior y superior de la SAFB en Hz.

    Warns
    -----
    UserWarning
        Si se usa alguna de las bandas de respaldo.

    Notes
    -----
    Ancho de banda. Se usa::

        bandwidth = sqrt(2·ln 2) · ceil(fwhm)
        fl, fh    = round(center ∓ ceil(bandwidth / 2))

    como en la implementación de referencia de los autores. El factor
    sqrt(2·ln 2) ≈ 1.177 da una banda ~18 % más ancha que el FWHM que describe
    el paper.

    Selección del pico. Solo se ajusta sobre frecuencias dentro de
    ``gamma_band`` y mayores que 0.5·F0, para no incluir la actividad de baja
    frecuencia.

    Respaldos. Si la banda estimada no queda dentro de ``noise_band``, se
    usa una banda de 40 Hz centrada en F0; si tampoco queda, una de 40 Hz
    centrada en 120 Hz.
    """
    f0_mean = float(np.mean(f0))
    freqs, psd = estimate_audio_psd(z, sfreq)

    in_peak = (freqs > gamma_band[0]) & (freqs < gamma_band[1]) & (freqs > 0.5 * f0_mean)
    center, fwhm = fit_gaussian_peak(freqs[in_peak], psd[in_peak], center0=f0_mean)

    fl, fh = _band_around(center, np.sqrt(2 * np.log(2)) * math.ceil(fwhm))
    if _is_valid_band(fl, fh, noise_band):
        return fl, fh

    fallback_center = f0_mean
    fl, fh = _band_around(fallback_center, FALLBACK_BANDWIDTH)
    if not _is_valid_band(fl, fh, noise_band):
        fallback_center = FALLBACK_F0
        fl, fh = _band_around(fallback_center, FALLBACK_BANDWIDTH)

    warnings.warn(
        f"La banda estimada (centro={center:.1f} Hz, FWHM={fwhm:.1f} Hz) no entra "
        f"en la banda de ruido {noise_band}; se usa la banda de respaldo "
        f"[{fl}, {fh}] Hz centrada en {fallback_center:.1f} Hz.",
        stacklevel=2,
    )
    return fl, fh
