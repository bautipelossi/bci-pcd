"""Simulación con ground truth conocido para evaluar PCD, la implementación de
referencia y CAR.
 Bautista Pelossi Schweizer

Uso, desde la raíz del repo:
    uv run pcd-simular

Seguimos el diseño del ejemplo de simulación de los autores
(Example_run_HybridSimulation.m y, en utilities/sim-toolkit/,
create_mixedsources.m, generate_mixingmatrix.m y get_audio.m): fuentes
neuronales más el audio real de un trial, mezclados con una matriz aleatoria
conocida. X_gt es la misma mezcla sin el audio. La diferencia es que las
fuentes neuronales no salen de la red de neuronas LIF sino de ruido 1/f, con
un aumento de gamma alta durante el habla en algunas fuentes, para que el
artefacto se solape con actividad fisiológica en la misma banda.

Datos: los audios salen de data/Audio.mat, y el inicio y el final del habla de
cada uno, de data/Audio_epochs.txt. Los dos vienen del repositorio de los
autores. Usamos los 20 primeros audios, uno por trial.

El script guarda en data/simulation/sim_fit.mat lo que necesita la
implementación de referencia para ajustar PCD (ventana del habla, audio y F0
de cada trial). Si existe data/reference/matlab_simulation.mat, también evalúa
esas matrices sobre los mismos datos.
"""

import time
import warnings
from pathlib import Path

import numpy as np
from pymatreader import read_mat
from scipy.io import savemat
from scipy.signal import butter, sosfiltfilt, welch

from src.pcd import PCD, select_artifact_components
from validation.metrics import chi_per_channel, msce_in_band, normalized_mse, pca_loading_similarity, plv

SFREQ = 1000.0
N_TRIALS = 20
# el demo usa 99 fuentes y 60 canales; con la mitad PCO corre mucho más rápido
N_SOURCES = 49
N_CHANNELS = 30
# relación artefacto / gamma neuronal en 70-180 Hz, en dB, como en el demo
SNR_LEVELS = np.arange(-5, 6)
# ventana de cada trial alrededor del inicio del habla (s), como get_audio.m
EPOCH = (-4.5, 4.5)
# fuentes con aumento de gamma alta fisiológica durante el habla
N_GAMMA_SOURCES = 10
SEED = 42
# banda donde se evalúa la preservación de la señal, además de banda ancha
HIGH_GAMMA = (70.0, 250.0)

FIT_FILE = Path("data/simulation/sim_fit.mat")
MATLAB_RESULTS = Path("data/reference/matlab_simulation.mat")


def bandpass(x: np.ndarray, band: tuple[float, float], order: int = 4) -> np.ndarray:
    sos = butter(order, band, btype="band", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def band_power(x: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    freqs, psd = welch(x, fs=SFREQ, nperseg=min(256, x.shape[-1]), axis=-1)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return psd[..., in_band].sum(axis=-1)


def pink_noise(rng: np.random.Generator, n_sources: int, n_samples: int, exponent: float = 1.5) -> np.ndarray:
    """Ruido con espectro 1/f^exponent, filtrado en 2-250 Hz como un iEEG."""
    spectrum = np.fft.rfft(rng.standard_normal((n_sources, n_samples)), axis=1)
    freqs = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n=n_samples, axis=1)
    noise = bandpass(noise, (2, 250))
    return noise / noise.std(axis=1, keepdims=True)


def neural_sources(rng: np.random.Generator, n_samples: int, speech: slice) -> np.ndarray:
    """Fuentes 1/f; las primeras N_GAMMA_SOURCES suben su gamma alta en el habla."""
    sources = pink_noise(rng, N_SOURCES, n_samples)
    envelope = np.zeros(n_samples)
    envelope[speech] = 1.0
    # subida y bajada suaves de 100 ms
    envelope = np.convolve(envelope, np.hanning(100) / np.hanning(100).sum(), mode="same")
    gamma = bandpass(rng.standard_normal((N_GAMMA_SOURCES, n_samples)), (70, 180))
    gamma /= gamma.std(axis=1, keepdims=True)
    sources[:N_GAMMA_SOURCES] += 2 * envelope * gamma
    return sources


def mixing_matrix(rng: np.random.Generator) -> np.ndarray:
    """Matriz (fuentes + audio) × canales, como generate_mixingmatrix.m."""
    return -0.5 + rng.random((N_SOURCES + 1, N_CHANNELS)) + rng.standard_normal((N_SOURCES + 1, N_CHANNELS))


def audio_epochs() -> list[dict]:
    """Audio de cada trial en la ventana EPOCH alrededor del inicio del habla."""
    audio = read_mat("data/Audio.mat")
    table = np.genfromtxt("data/Audio_epochs.txt", delimiter=",", names=True)
    epochs = []
    for i in range(N_TRIALS):
        t, z = audio["t"][i], audio["z"][i]
        onset = int(round((table["t0"][i] - t[0]) * SFREQ))
        start, stop = onset + int(EPOCH[0] * SFREQ), onset + int(EPOCH[1] * SFREQ)
        # el habla arranca en t0, que dentro de la ventana es la muestra onset - start
        n_speech = int(round(table["prod_offset_t0"][i] * SFREQ))
        speech = slice(onset - start, onset - start + n_speech)
        epochs.append({"z": z[start:stop], "speech": speech})
    return epochs


def estimate_f0(z: np.ndarray) -> float:
    """F0 aproximada: el pico del espectro del audio en 70-240 Hz.

    Audio.mat no trae la anotación de F0, que en los datos reales sí está.
    """
    freqs, psd = welch(z, fs=SFREQ, nperseg=200, nfft=1024)
    in_range = (freqs >= 70) & (freqs <= 240)
    return float(freqs[in_range][np.argmax(psd[in_range])])


def simulate(rng: np.random.Generator) -> list[dict]:
    A = mixing_matrix(rng)
    A_gt = A.copy()
    A_gt[-1] = 0  # sin el audio
    snr = rng.choice(SNR_LEVELS, N_TRIALS)
    trials = []
    for epoch, snr_db in zip(audio_epochs(), snr):
        z, speech = epoch["z"], epoch["speech"]
        D = neural_sources(rng, len(z), speech)
        # Escalamos la actividad neuronal para que, en los canales, el artefacto
        # quede snr_db por encima de ella en 70-180 Hz durante el habla. El demo
        # la compara con una sola fuente, pero cada canal suma todas, así que
        # ahí el artefacto quedaba ~17 dB por debajo de lo nominal.
        pow_artifact = band_power(np.outer(A[-1], z[speech]), (70, 180)).mean()
        pow_neural = band_power(A[:-1].T @ D[:, speech], (70, 180)).mean()
        scale = pow_artifact / pow_neural * 10 ** (-snr_db / 10)
        sources = np.vstack([np.sqrt(scale) * D, z])
        trials.append({
            "X": A.T @ sources,
            "X_gt": A_gt.T @ sources,
            "z": z,
            "speech": speech,
            "f0": estimate_f0(z[speech]),
            "snr": int(snr_db),
        })
    return trials


def save_fit_data(trials: list[dict]) -> None:
    """Lo que necesita la implementación de referencia para ajustar PCD."""
    FIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def cells(arrays):
        # arreglo de objetos, que savemat guarda como cell de MATLAB
        out = np.empty(len(arrays), dtype=object)
        out[:] = arrays
        return out

    savemat(FIT_FILE, {"sim": {
        "X_fit": cells([t["X"][:, t["speech"]] for t in trials]),
        "z_fit": cells([t["z"][t["speech"]] for t in trials]),
        "f0": np.array([t["f0"] for t in trials]),
        "sf": SFREQ,
    }})


def evaluate(X_clean: np.ndarray, source: np.ndarray | None, trial: dict, band: tuple[int, int]) -> dict:
    """Métricas de una limpieza contra el ground truth del trial.

    χ y CS se calculan en banda ancha sobre toda la ventana, y en gamma alta
    durante el habla. En banda ancha casi toda la energía es la actividad 1/f
    de baja frecuencia, y fuera del habla no hay artefacto, así que ahí la
    limpieza apenas se nota; en gamma alta durante el habla es donde está el
    artefacto y la actividad que importa para decodificar habla.
    """
    X_gt, speech = trial["X_gt"], trial["speech"]
    # filtramos la ventana completa y después recortamos, para evitar bordes
    X_clean_gamma = bandpass(X_clean, HIGH_GAMMA)[:, speech]
    X_gt_gamma = bandpass(X_gt, HIGH_GAMMA)[:, speech]
    result = {
        "chi": float(np.median(chi_per_channel(X_clean, X_gt))),
        "cs": pca_loading_similarity(X_gt, X_clean),
        "chi_gamma": float(np.median(chi_per_channel(X_clean_gamma, X_gt_gamma))),
        "cs_gamma": pca_loading_similarity(X_gt_gamma, X_clean_gamma),
    }
    # promedio de CS1-CS3 en el trial; en el resumen se toma la mediana sobre trials
    result["cs_mean"] = float(result["cs"].mean())
    result["cs_mean_gamma"] = float(result["cs_gamma"].mean())
    if source is not None:
        # la fuente estimada se compara con el audio en la ventana del habla
        z, estimated = trial["z"][trial["speech"]], source[trial["speech"]]
        result["msce"] = msce_in_band(z, estimated, SFREQ, band)
        # cada señal dividida por su máximo absoluto, como describe el paper
        result["mse"] = normalized_mse(z, estimated)
        result["plv"] = plv(bandpass(z, band), bandpass(estimated, band))
    return result


def matlab_cleaning(trial: dict, ref: dict, i: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Limpieza con las matrices de la referencia (patrones en filas)."""
    W, A = ref["W_pcd"][i], ref["A_pcd"][i].T
    keep = np.atleast_1d(ref["idx_keep"][i]).astype(int) - 1
    X_clean = A[:, keep] @ W[:, keep].T @ trial["X"]
    return X_clean, W[:, 0] @ trial["X"], (int(ref["fl"][i]), int(ref["fh"][i]))


def summary(results: list[dict], key: str, index: int | None = None) -> str:
    values = np.array([r[key] if index is None else r[key][index] for r in results])
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    # el MSE normalizado es del orden de 1e-3 y con 3 decimales no se lee
    fmt = ".1e" if abs(med) < 0.01 else ".3f"
    return f"{med:{fmt}} [{q1:{fmt}}, {q3:{fmt}}]"


def main() -> None:
    rng = np.random.default_rng(SEED)
    trials = simulate(rng)
    save_fit_data(trials)
    print(f"{N_TRIALS} trials | {N_SOURCES} fuentes + audio -> {N_CHANNELS} canales | SNR {SNR_LEVELS[0]} a {SNR_LEVELS[-1]} dB")
    print(f"datos para la referencia: {FIT_FILE}")

    results = {"Sin limpiar": [], "CAR": [], "PCD Python": []}
    for i, trial in enumerate(trials):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcd = PCD(SFREQ, random_state=0).fit(trial["X"][:, trial["speech"]], trial["z"][trial["speech"]], f0=trial["f0"])
        X_pcd = pcd.apply(trial["X"])
        band = pcd.signal_band_
        n_removed = len(select_artifact_components(pcd.vlen_, "diff"))
        print(f"  trial {i + 1:2d}: SNR {trial['snr']:+d} dB | SAFB {band} | k = {pcd.k_} | saca {n_removed} | {time.perf_counter() - start:.0f} s", flush=True)
        results["Sin limpiar"].append(evaluate(trial["X"], None, trial, band))
        results["CAR"].append(evaluate(trial["X"] - trial["X"].mean(axis=0), None, trial, band))
        results["PCD Python"].append(evaluate(X_pcd, pcd.transform(trial["X"])[0], trial, band))

    if MATLAB_RESULTS.exists():
        ref = read_mat(MATLAB_RESULTS)["sim_ref"]
        results["PCD referencia"] = []
        for i, trial in enumerate(trials):
            X_clean, source, band = matlab_cleaning(trial, ref, i)
            results["PCD referencia"].append(evaluate(X_clean, source, trial, band))
    else:
        print(f"no está {MATLAB_RESULTS}: se evalúa solo Python")

    print("\nmediana [Q1, Q3] sobre los trials")
    for title, suffix in (
        ("banda ancha, ventana completa", ""),
        (f"gamma alta {HIGH_GAMMA[0]:.0f}-{HIGH_GAMMA[1]:.0f} Hz, durante el habla", "_gamma"),
    ):
        print(f"\n{title}")
        print(f"{'':16s} {'chi':>22s} {'CS 1':>22s} {'CS 2':>22s} {'CS 3':>22s} {'CS prom.':>22s}")
        for name, res in results.items():
            print(
                f"{name:16s} {summary(res, 'chi' + suffix):>22s} "
                + " ".join(f"{summary(res, 'cs' + suffix, j):>22s}" for j in range(3))
                + f" {summary(res, 'cs_mean' + suffix):>22s}"
            )
    print(f"\n{'':16s} {'MSE normalizado':>22s} {'MSCE en la SAFB':>22s} {'PLV en la SAFB':>22s}")
    for name, res in results.items():
        if "msce" in res[0]:
            print(f"{name:16s} {summary(res, 'mse'):>22s} {summary(res, 'msce'):>22s} {summary(res, 'plv'):>22s}")


if __name__ == "__main__":
    main()
