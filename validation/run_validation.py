"""Compara la implementación en Python contra la implementación de referencia en Data.mat.
 Bautista Pelossi Schweizer

Uso, desde la raíz del repo:
    uv run pcd-validar [ruta_a_la_referencia.mat]

La referencia (data/reference/matlab_reference.mat) sale de correr el código
original sobre Data.mat, y data/reference/matlab_safb_trials.mat de estimar la
SAFB de los 20 trials de DataTrials.mat. La sección 0 estima la SAFB y k por
Python cuenta y los compara con la referencia; el resto usa esos mismos
valores, sin fijar nada de la referencia. PCO es no
convexo y arranca de puntos al azar, así que ahí esperamos máximos distintos;
la sección 4 aísla ese azar usando los filtros PCO de la referencia. La 5
replica el ejemplo de ITPC de los autores (antes y después de PCD en los 20
trials) y lo compara con data/reference/matlab_itpc_pcd.mat; es la única
métrica que se usa acá, porque en datos reales no hay ground truth.
"""

import sys
import warnings

import numpy as np
from pymatreader import read_mat
from scipy.linalg import block_diag

from src.pcd import PCD, select_artifact_components
from src.safb import estimate_audio_psd, estimate_safb, fit_gaussian_peak
from validation.metrics import (
    abs_cosine,
    channel_correlation,
    itpc,
    mean_vector_length,
    subspace_similarity,
    trial_coherence,
)

DEFAULT_REFERENCE = "data/reference/matlab_reference.mat"
SAFB_TRIALS_REFERENCE = "data/reference/matlab_safb_trials.mat"
ITPC_REFERENCE = "data/reference/matlab_itpc_pcd.mat"
# umbral de contaminación del ITPC (Bush et al., 2022)
ITPC_THRESHOLD = 3.08


def summary(values: np.ndarray) -> str:
    return f"mediana {np.median(values):.4f} | mín {np.min(values):.4f} | máx {np.max(values):.4f}"


def relative_error(estimated: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimated - reference) / np.linalg.norm(reference))


def gaussian_fit(z: np.ndarray, sfreq: float, f0: float) -> tuple[float, float]:
    """Centro y FWHM de la gaussiana, con el mismo recorte que usa estimate_safb."""
    freqs, psd = estimate_audio_psd(z, sfreq)
    in_peak = (freqs > 50) & (freqs < 250) & (freqs > 0.5 * f0)
    return fit_gaussian_peak(freqs[in_peak], psd[in_peak], center0=f0)


def safb_python(z: np.ndarray, sfreq: float, f0: float) -> tuple[tuple[int, int], bool]:
    """SAFB estimada y si se usó una banda de respaldo."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        band = estimate_safb(z, sfreq, f0)
    return band, len(caught) > 0


def main() -> None:
    reference = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REFERENCE
    data = read_mat("data/Data.mat")["Data"]
    ref = read_mat(reference)["ref_data"]
    X_fit, z, X_toclean = data["X_tofit"], data["z"], data["X_toclean"]
    sfreq = float(data["sf"])
    band = (int(ref["fl"]), int(ref["fh"]))
    k = int(ref["k"])
    removed_ref = np.setdiff1d(np.arange(1, k + 1), ref["idx_keep"]) - 1

    print(f"Referencia: {reference}")

    # --- SAFB y k, estimados en implementación propia ---------------------------------
    print("\n0) SAFB y k estimados en Python, sin fijar nada")
    f0 = float(np.nanmean(data["F0"]))
    band_ours, fallback = safb_python(z, sfreq, f0)
    center, fwhm = gaussian_fit(z, sfreq, f0)
    print(
        f"  Data.mat: SAFB Python {band_ours} Hz | referencia {band} Hz | respaldo: {'sí' if fallback else 'no'}"
    )
    print(f"  gaussiana Python: centro {center:.3f} Hz, FWHM {fwhm:.3f} Hz")
    print(
        f"  gaussiana ref.   : centro {ref['gauss_center']:.3f} Hz, FWHM {ref['gauss_width']:.3f} Hz"
    )

    trials = read_mat("data/DataTrials.mat")["DataTrials"]
    ref_trials = read_mat(SAFB_TRIALS_REFERENCE)["safb_trials"]
    n_equal = 0
    print("  DataTrials.mat (trial: Python | referencia):")
    for i, trial in enumerate(trials):
        band_i, fallback_i = safb_python(
            trial["z"], float(trial["sf"]), float(np.nanmean(trial["F0"]))
        )
        band_ref_i = (int(ref_trials["fl"][i]), int(ref_trials["fh"][i]))
        n_equal += band_i == band_ref_i
        notes = []
        if fallback_i:
            notes.append("respaldo en Python")
        if ref_trials["used_fallback"][i]:
            notes.append("respaldo en la referencia")
        mark = "" if band_i == band_ref_i else "  <- distinta"
        print(f"    {i + 1:2d}: {band_i} | {band_ref_i}{mark} {', '.join(notes)}")
    print(f"  SAFB iguales: {n_equal} de {len(trials)} trials")

    # de acá en adelante todo usa Python propia SAFB y k
    pcd = PCD(sfreq, random_state=0).fit(X_fit, z, f0=f0)
    print(f"  k por participation ratio: Python {pcd.k_} | referencia {k}")
    if pcd.signal_band_ != band or pcd.k_ != k:
        # las comparaciones filtro a filtro de abajo suponen las mismas entradas a SSD
        print(
            "  OJO: la SAFB o k no coinciden con la referencia; las secciones 1-3 comparan cosas distintas"
        )
    k = min(k, pcd.k_)

    # --- SSD -------------------------------------------------------------------
    # comparamos los filtros en el espacio de los sensores (M @ W_ssd): así no
    # importa el signo que cada implementación le dio a los autovectores de M
    print("\n1) SSD")
    ssd_ours, ssd_ref = pcd.M_ @ pcd.W_ssd_[:, :k], ref["M"] @ ref["W_ssd"][:, :k]
    cos_ssd = abs_cosine(ssd_ours, ssd_ref)
    print(f"  |cos| entre filtros 1..{k}: {summary(cos_ssd)}")
    print(f"  |cos| filtro por filtro: {np.round(cos_ssd, 3)}")
    # PCO busca en todo el subespacio de los k filtros, así que eso es lo que importa
    print(
        f"  subespacio de los {k} filtros, cos de los ángulos principales: {summary(subspace_similarity(ssd_ours, ssd_ref))}"
    )
    lambda_ref = ref["lambda_ssd"][:k]
    print(f"  lambda Python   : {np.round(pcd.lambda_ssd_[:5], 4)}")
    print(f"  lambda referencia: {np.round(lambda_ref[:5], 4)}")
    print(
        f"  diferencia relativa de lambda (1..{k}): {summary(np.abs(pcd.lambda_ssd_[:k] - lambda_ref) / lambda_ref)}"
    )

    # --- PCO -------------------------------------------------------------------
    print("\n2) PCO")
    print(f"  MVL Python   : {np.round(pcd.vlen_[:6], 3)}")
    print(f"  MVL referencia: {np.round(ref['vlen'][:6], 3)}")
    # si Python función da el MVL que reporta la referencia para sus propios filtros,
    # el objetivo es el mismo y lo que cambia es a qué máximo llega cada optimizador
    mvl_check = [mean_vector_length(ref["W_pcd"][:, j] @ X_fit, z) for j in range(6)]
    print(
        f"  MVL de los filtros de la referencia, recalculado acá: {np.round(mvl_check, 3)}"
    )
    pco_ours, pco_ref = pcd.W_pcd_[:, :k], ref["W_pcd"][:, :k]
    print(
        f"  |cos| filtro PCD 1 (el del artefacto): {abs_cosine(pco_ours[:, 0], pco_ref[:, 0])[0]:.4f}"
    )
    # dos componentes con MVL parecido pueden salir en otro orden, así que
    # también buscamos, para cada filtro de la referencia, el Python más parecido
    gram = np.abs(
        (pco_ref / np.linalg.norm(pco_ref, axis=0)).T
        @ (pco_ours / np.linalg.norm(pco_ours, axis=0))
    )
    for i in range(3):
        j = int(np.argmax(gram[i]))
        print(
            f"  filtro {i + 1} de la referencia: el más parecido es Python {j + 1} con |cos| {gram[i, j]:.4f}"
        )
    # lo que se saca de la señal es el subespacio de los patrones de artefacto
    n_removed = len(removed_ref)
    artifact_similarity = subspace_similarity(
        pcd.A_pcd_[:, :n_removed], ref["A_pcd"][:n_removed].T
    )
    print(
        f"  subespacio de los {n_removed} patrones de artefacto, cos de los ángulos: {np.round(artifact_similarity, 4)}"
    )

    # --- Reconstrucción -------------------------------------------------------------
    print(
        "\n3) Reconstrucción sobre X_toclean, de punta a punta (SAFB, k y 'diff' propios)"
    )
    X_ref = ref["X_clean"]
    X_ours = pcd.apply(X_toclean)
    print(
        f"  componentes sacadas: Python {select_artifact_components(pcd.vlen_, 'diff').tolist()} | referencia {removed_ref.tolist()}"
    )
    print(
        f"  corr canal a canal de la señal limpia: {summary(channel_correlation(X_ours, X_ref))}"
    )
    # la señal limpia es casi toda la original, así que su correlación sale alta
    # igual; lo que realmente compara las implementaciones es lo que cada una sacó
    removed_ours, removed_signal_ref = X_toclean - X_ours, X_toclean - X_ref
    print(
        f"  corr canal a canal de lo removido: {summary(channel_correlation(removed_ours, removed_signal_ref))}"
    )
    print(
        f"  ||limpia Python - limpia ref|| / ||removido ref||: {np.linalg.norm(X_ours - X_ref) / np.linalg.norm(removed_signal_ref):.4f}"
    )
    print(
        f"  energía removida Python / de la referencia: {np.linalg.norm(removed_ours) / np.linalg.norm(removed_signal_ref):.4f}"
    )

    # --- Reconstrucción con los filtros de la referencia ------------------------------
    # le damos a Python armado las matrices de la referencia (M, W_ssd, W_pco):
    # así sacamos el azar de PCO y probamos solo W_pcd, A_pcd y la reconstrucción
    print("\n4) Reconstrucción con M, W_ssd y W_pco de la referencia")
    same = PCD(sfreq)
    same.vlen_ = ref["vlen"]
    n_rest = X_fit.shape[0] - k
    same.W_pcd_ = (
        ref["M"]
        @ ref["W_ssd"]
        @ block_diag(np.atleast_2d(ref["W_pco"]), np.eye(n_rest))
    )
    same.A_pcd_ = np.linalg.pinv(same.W_pcd_).T
    X_same = same.apply(X_toclean, removed_ref)
    print(f"  W_pcd: error relativo {relative_error(same.W_pcd_, ref['W_pcd']):.2e}")
    print(
        f"  A_pcd (patrones en columnas vs filas): error relativo {relative_error(same.A_pcd_, ref['A_pcd'].T):.2e}"
    )
    print(
        f"  fuentes X_pcd: error relativo {relative_error(same.transform(X_fit)[:k], ref['X_pcd']):.2e}"
    )
    print(f"  X_clean: error relativo {relative_error(X_same, X_ref):.2e}")
    print(
        f"  lo removido: error relativo {relative_error(X_toclean - X_same, removed_signal_ref):.2e}"
    )

    # --- ITPC ------------------------------------------------------------------------
    # en datos reales no hay X_gt ni fuente verdadera, así que la única métrica
    # que tiene sentido es el ITPC; el resto se valida con la simulación
    # Como en el ejemplo de los autores: PCD por trial con su propia SAFB, se
    # ajusta en Xtofit y se aplica sobre X_toclean. El ejemplo calcula la
    # coherencia con las primeras len(z) muestras de X_toclean, que no es la
    # ventana del habla; por eso también la calculamos en Xtofit, que sí está
    # alineada con z.
    print("\n5) ITPC antes y después de PCD sobre DataTrials.mat")
    itpc_ref = read_mat(ITPC_REFERENCE)["itpc_pcd"]
    coherence = {
        key: [] for key in ("before", "after", "before_speech", "after_speech")
    }
    n_removed = []
    for trial in trials:
        X_trial, z_trial, n_z = trial["X_toclean"], trial["z"], len(trial["z"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcd_trial = PCD(float(trial["sf"]), random_state=0).fit(
                trial["Xtofit"], z_trial, f0=float(np.nanmean(trial["F0"]))
            )
        n_removed.append(len(select_artifact_components(pcd_trial.vlen_, "diff")))
        coherence["before"].append(trial_coherence(X_trial[:, :n_z], z_trial))
        coherence["after"].append(
            trial_coherence(pcd_trial.apply(X_trial)[:, :n_z], z_trial)
        )
        coherence["before_speech"].append(trial_coherence(trial["Xtofit"], z_trial))
        coherence["after_speech"].append(
            trial_coherence(pcd_trial.apply(trial["Xtofit"]), z_trial)
        )
    print(f"  componentes sacadas por trial: Python     {n_removed}")
    print(
        f"                                 referencia {itpc_ref['n_removed'].astype(int).tolist()}"
    )
    for label, window in (
        ("como el ejemplo (principio de X_toclean)", ""),
        ("ventana del habla (Xtofit)", "_speech"),
    ):
        before, after = (
            itpc(np.array(coherence["before" + window])),
            itpc(np.array(coherence["after" + window])),
        )
        before_ref, after_ref = (
            itpc_ref["ITPC_before" + window],
            itpc_ref["ITPC_after" + window],
        )
        print(f"  {label}:")
        print(
            f"    antes  : Python {np.round(before, 3)} | referencia {np.round(before_ref, 3)} | error relativo {relative_error(before, before_ref):.1e}"
        )
        print(
            f"    después: Python {np.round(after, 3)} | referencia {np.round(after_ref, 3)}"
        )
    print(f"  (umbral de contaminación: ITPC >= {ITPC_THRESHOLD})")


if __name__ == "__main__":
    main()
