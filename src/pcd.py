"""Phase Coupling Decomposition (PCD).
 Cuarta etapa de PCD: matrices de separación y mezcla, y reconstrucción
 Bautista Pelossi Schweizer
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from .pco import fit_pco
from .safb import estimate_safb
from .ssd import fit_ssd, select_n_components, whiten


def select_artifact_components(
    vlen: np.ndarray, n_remove: int | float | str | np.ndarray | None
) -> np.ndarray:
    """Elige qué componentes PCO se toman como artefacto.

    Parameters
    ----------
    vlen : np.ndarray, forma (k,)
        MVL de cada componente PCO, descendente.
    n_remove : int, float, "PR", "diff", array or None
        - ``None``: no se saca nada.
        - ``int`` n: las n primeras componentes.
        - ``float``: las que tienen MVL mayor que ese umbral.
        - ``"PR"``: las ``round(sum(vlen) / max(vlen))`` primeras.
        - ``"diff"``: corta en el codo de la curva de MVL (ver Notes).
        - array: índices explícitos, en base 0.

    Returns
    -------
    np.ndarray
        Índices (base 0) de las componentes a sacar.

    Notes
    -----
    ``"diff"`` toma los saltos entre MVL consecutivos, contando un salto
    inicial desde 1, y corta en el primer salto menor que 100 veces el salto
    más chico. Es el criterio que usa la implementación de los autores para el
    codo de la curva de MVL.
    """
    if n_remove is None:
        return np.array([], dtype=int)
    if isinstance(n_remove, str):
        if n_remove == "PR":
            return np.arange(round(vlen.sum() / vlen.max()))
        if n_remove == "diff":
            steps = np.abs(np.diff(np.concatenate([[1.0], vlen])))
            small = np.flatnonzero(steps < 100 * steps.min())
            # si no hay ningún salto chico no hay codo y no sacamos nada
            return np.arange(small[0] + 1 if small.size else 0)
        raise ValueError(f"n_remove='{n_remove}' no existe; usar 'PR' o 'diff'.")
    if isinstance(n_remove, float):
        return np.flatnonzero(vlen > n_remove)
    if isinstance(n_remove, int):
        return np.arange(n_remove)
    return np.asarray(n_remove, dtype=int)


class PCD:
    """Phase Coupling Decomposition para sacar el artefacto del habla.

    Parameters
    ----------
    sfreq : float
        Frecuencia de muestreo en Hz.
    signal_band : tuple of float or None, default None
        SAFB ``(fl, fh)`` en Hz. Si es ``None`` se estima en :meth:`fit` a
        partir del audio y de F0.
    noise_band : tuple of float, default (4.0, 240.0)
        Band-pass de ruido de SSD, en Hz.
    filter_order : int, default 5
        Orden de los Butterworth de SSD.
    n_components : int, float or "PR", default "PR"
        Cómo elegir k, la cantidad de componentes SSD que pasan a PCO
        (ver :func:`src.ssd.select_n_components`).
    n_restarts : int, default 15
        Inicializaciones aleatorias por filtro PCO.
    random_state : int, Generator or None, default None
        Semilla de PCO.

    Attributes
    ----------
    signal_band_ : tuple of float
        SAFB usada.
    M_ : np.ndarray, forma (N_c, N_c)
        Matriz de blanqueo.
    W_ssd_ : np.ndarray, forma (N_c, N_c)
        Filtros SSD (en el espacio blanqueado), en columnas.
    lambda_ssd_ : np.ndarray, forma (N_c,)
        Autovalores SSD.
    k_ : int
        Componentes SSD que pasaron a PCO.
    W_pco_ : np.ndarray, forma (k, k)
        Filtros PCO, en columnas.
    vlen_ : np.ndarray, forma (k,)
        MVL de cada componente PCO, descendente.
    W_pcd_ : np.ndarray, forma (N_c, N_c)
        Matriz de separación: las fuentes son ``W_pcd_.T @ X``.
    A_pcd_ : np.ndarray, forma (N_c, N_c)
        Matriz de mezcla, con los patrones espaciales en las columnas:
        ``X = A_pcd_ @ S``.

    Notes
    -----
    Las primeras k columnas de ``W_pcd_`` son las componentes PCO ordenadas
    por MVL; las restantes N_c - k son las componentes SSD que no pasaron a
    PCO y nunca se sacan.

    ``A_pcd_ = pinv(W_pcd_).T``, así que cada patrón es una columna. La
    implementación de los autores guarda ``pinv(W_pcd)`` con los patrones en
    las filas.

    References
    ----------
    Peterson, V., et al. (2024). A supervised data-driven spatial filter
    denoising method for speech artifacts in intracranial electrophysiological
    recordings. Imaging Neuroscience, 2, 1-22. doi:10.1162/imag_a_00301
    """

    def __init__(
        self,
        sfreq: float,
        signal_band: tuple[float, float] | None = None,
        noise_band: tuple[float, float] = (4.0, 240.0),
        filter_order: int = 5,
        n_components: int | float | str = "PR",
        n_restarts: int = 15,
        random_state: int | np.random.Generator | None = None,
    ):
        self.sfreq = sfreq
        self.signal_band = signal_band
        self.noise_band = noise_band
        self.filter_order = filter_order
        self.n_components = n_components
        self.n_restarts = n_restarts
        self.random_state = random_state

    def fit(self, X: np.ndarray, z: np.ndarray, f0: float | np.ndarray | None = None) -> PCD:
        """Aprende las matrices de separación y mezcla.

        Parameters
        ----------
        X : np.ndarray, forma (N_c, N_s)
            iEEG en la ventana del habla producida.
        z : np.ndarray, forma (N_s,)
            Audio alineado con ``X``.
        f0 : float, array or None, default None
            F0 en Hz. Solo hace falta si ``signal_band`` es ``None``.

        Returns
        -------
        PCD
            El mismo objeto, ajustado.
        """
        if self.signal_band is not None:
            self.signal_band_ = tuple(self.signal_band)
        elif f0 is None:
            raise ValueError("Sin signal_band hace falta f0 para estimar la SAFB.")
        else:
            self.signal_band_ = estimate_safb(z, self.sfreq, f0, noise_band=self.noise_band)

        X_white, self.M_ = whiten(X)
        self.W_ssd_, self.lambda_ssd_ = fit_ssd(
            X_white.real, self.sfreq, self.signal_band_, self.noise_band, self.filter_order
        )
        self.k_ = select_n_components(self.lambda_ssd_, self.n_components)

        # PCO trabaja sobre la señal analítica proyectada en las k primeras SSD
        X_ssd = self.W_ssd_[:, : self.k_].T @ X_white
        self.W_pco_, self.vlen_ = fit_pco(X_ssd, z, self.n_restarts, self.random_state)

        # encadenamos blanqueo, SSD y PCO; las SSD que no pasaron a PCO quedan igual
        n_rest = X.shape[0] - self.k_
        self.W_pcd_ = self.M_ @ self.W_ssd_ @ block_diag(self.W_pco_, np.eye(n_rest))
        self.A_pcd_ = np.linalg.pinv(self.W_pcd_).T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fuentes PCD, ``W_pcd_.T @ X``.

        Parameters
        ----------
        X : np.ndarray, forma (N_c, N_s)

        Returns
        -------
        np.ndarray, forma (N_c, N_s)
            Las k primeras filas son las componentes PCO, de mayor a menor MVL.
        """
        return self.W_pcd_.T @ np.nan_to_num(X)

    def apply(
        self, X: np.ndarray, n_remove: int | float | str | np.ndarray | None = "diff"
    ) -> np.ndarray:
        """Saca las componentes de artefacto y reconstruye la señal.

        Parameters
        ----------
        X : np.ndarray, forma (N_c, N_s)
            iEEG a limpiar. Puede ser una ventana más larga que la de :meth:`fit`.
        n_remove : int, float, "PR", "diff", array or None, default "diff"
            Qué componentes sacar (ver :func:`select_artifact_components`).

        Returns
        -------
        np.ndarray, forma (N_c, N_s)
            ``A_pcd_[:, keep] @ W_pcd_[:, keep].T @ X``, con ``keep`` todas las
            componentes menos las de artefacto.

        Notes
        -----
        Con un ``int`` se sacan las n primeras componentes. La implementación
        de los autores interpreta un número entero como el índice de una sola
        componente.
        """
        artifact = select_artifact_components(self.vlen_, n_remove)
        keep = np.setdiff1d(np.arange(X.shape[0]), artifact)
        return self.A_pcd_[:, keep] @ self.W_pcd_[:, keep].T @ X
