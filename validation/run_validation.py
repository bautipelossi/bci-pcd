"""Compara nuestra implementación contra la implementación de referencia en Data.mat.
 Bautista Pelossi Schweizer

Uso, desde la raíz del repo:
    uv run python validation/run_validation.py [ruta_a_la_referencia.mat]

La referencia (data/reference/matlab_reference.mat) sale de correr el código
original sobre Data.mat. Fijamos la SAFB y k de la referencia para comparar
cada etapa con las mismas entradas. PCO es no convexo y arranca de puntos al
azar, así que ahí esperamos máximos distintos; la sección 4 aísla ese azar
usando los filtros PCO de la referencia.
"""

import sys

import numpy as np
from metrics import abs_cosine, channel_correlation, mean_vector_length, subspace_similarity
from pymatreader import read_mat
from scipy.linalg import block_diag

from src.pcd import PCD, select_artifact_components

REFERENCE = sys.argv[1] if len(sys.argv) > 1 else "data/reference/matlab_reference.mat"


def summary(values: np.ndarray) -> str:
    return f"mediana {np.median(values):.4f} | mín {np.min(values):.4f} | máx {np.max(values):.4f}"


def relative_error(estimated: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimated - reference) / np.linalg.norm(reference))


data = read_mat("data/Data.mat")["Data"]
ref = read_mat(REFERENCE)["ref_data"]
X_fit, z, X_toclean = data["X_tofit"], data["z"], data["X_toclean"]
sfreq = float(data["sf"])
band = (int(ref["fl"]), int(ref["fh"]))
k = int(ref["k"])
removed_ref = np.setdiff1d(np.arange(1, k + 1), ref["idx_keep"]) - 1

print(f"Referencia: {REFERENCE}")
print(f"Data.mat: {X_fit.shape[0]} canales | SAFB {band} Hz | k = {k} (fijos, de la referencia)")
pcd = PCD(sfreq, signal_band=band, n_components=k, random_state=0).fit(X_fit, z)

# --- SSD -------------------------------------------------------------------
# comparamos los filtros en el espacio de los sensores (M @ W_ssd): así no
# importa el signo que cada implementación le dio a los autovectores de M
print("\n1) SSD")
ssd_ours, ssd_ref = pcd.M_ @ pcd.W_ssd_[:, :k], ref["M"] @ ref["W_ssd"][:, :k]
cos_ssd = abs_cosine(ssd_ours, ssd_ref)
print(f"  |cos| entre filtros 1..{k}: {summary(cos_ssd)}")
print(f"  |cos| filtro por filtro: {np.round(cos_ssd, 3)}")
# PCO busca en todo el subespacio de los k filtros, así que eso es lo que importa
print(f"  subespacio de los {k} filtros, cos de los ángulos principales: {summary(subspace_similarity(ssd_ours, ssd_ref))}")
lambda_ref = ref["lambda_ssd"][:k]
print(f"  lambda nuestro   : {np.round(pcd.lambda_ssd_[:5], 4)}")
print(f"  lambda referencia: {np.round(lambda_ref[:5], 4)}")
print(f"  diferencia relativa de lambda (1..{k}): {summary(np.abs(pcd.lambda_ssd_[:k] - lambda_ref) / lambda_ref)}")

# --- PCO -------------------------------------------------------------------
print("\n2) PCO")
print(f"  MVL nuestro   : {np.round(pcd.vlen_[:6], 3)}")
print(f"  MVL referencia: {np.round(ref['vlen'][:6], 3)}")
# si nuestra función da el MVL que reporta la referencia para sus propios filtros,
# el objetivo es el mismo y lo que cambia es a qué máximo llega cada optimizador
mvl_check = [mean_vector_length(ref["W_pcd"][:, j] @ X_fit, z) for j in range(6)]
print(f"  MVL de los filtros de la referencia, recalculado acá: {np.round(mvl_check, 3)}")
pco_ours, pco_ref = pcd.W_pcd_[:, :k], ref["W_pcd"][:, :k]
print(f"  |cos| filtro PCD 1 (el del artefacto): {abs_cosine(pco_ours[:, 0], pco_ref[:, 0])[0]:.4f}")
# dos componentes con MVL parecido pueden salir en otro orden, así que
# también buscamos, para cada filtro de la referencia, el nuestro más parecido
gram = np.abs(
    (pco_ref / np.linalg.norm(pco_ref, axis=0)).T @ (pco_ours / np.linalg.norm(pco_ours, axis=0))
)
for i in range(3):
    j = int(np.argmax(gram[i]))
    print(f"  filtro {i + 1} de la referencia: el más parecido es nuestro {j + 1} con |cos| {gram[i, j]:.4f}")
# lo que se saca de la señal es el subespacio de los patrones de artefacto
n_removed = len(removed_ref)
artifact_similarity = subspace_similarity(pcd.A_pcd_[:, :n_removed], ref["A_pcd"][:n_removed].T)
print(f"  subespacio de los {n_removed} patrones de artefacto, cos de los ángulos: {np.round(artifact_similarity, 4)}")

# --- Reconstrucción -------------------------------------------------------------
print("\n3) Reconstrucción sobre X_toclean (nuestro pipeline completo)")
X_ref = ref["X_clean"]
X_ours = pcd.apply(X_toclean, removed_ref)
print(f"  componentes sacadas por la referencia ('diff'): {removed_ref.tolist()} -> usamos las mismas")
print(f"  con nuestro propio 'diff' sacaríamos: {select_artifact_components(pcd.vlen_, 'diff').tolist()}")
print(f"  corr canal a canal de la señal limpia: {summary(channel_correlation(X_ours, X_ref))}")
# la señal limpia es casi toda la original, así que su correlación sale alta
# igual; lo que realmente compara las implementaciones es lo que cada una sacó
removed_ours, removed_signal_ref = X_toclean - X_ours, X_toclean - X_ref
print(f"  corr canal a canal de lo removido: {summary(channel_correlation(removed_ours, removed_signal_ref))}")
print(f"  ||limpia nuestra - limpia ref|| / ||removido ref||: {relative_error(X_ours, X_ref) * np.linalg.norm(X_ref) / np.linalg.norm(removed_signal_ref):.4f}")

# --- Reconstrucción con los filtros de la referencia ------------------------------
# le damos a nuestro armado las matrices de la referencia (M, W_ssd, W_pco):
# así sacamos el azar de PCO y probamos solo W_pcd, A_pcd y la reconstrucción
print("\n4) Reconstrucción con M, W_ssd y W_pco de la referencia")
same = PCD(sfreq)
same.vlen_ = ref["vlen"]
n_rest = X_fit.shape[0] - k
same.W_pcd_ = ref["M"] @ ref["W_ssd"] @ block_diag(np.atleast_2d(ref["W_pco"]), np.eye(n_rest))
same.A_pcd_ = np.linalg.pinv(same.W_pcd_).T
X_same = same.apply(X_toclean, removed_ref)
print(f"  W_pcd: error relativo {relative_error(same.W_pcd_, ref['W_pcd']):.2e}")
print(f"  A_pcd (patrones en columnas vs filas): error relativo {relative_error(same.A_pcd_, ref['A_pcd'].T):.2e}")
print(f"  fuentes X_pcd: error relativo {relative_error(same.transform(X_fit)[:k], ref['X_pcd']):.2e}")
print(f"  X_clean: error relativo {relative_error(X_same, X_ref):.2e}")
print(f"  lo removido: error relativo {relative_error(X_toclean - X_same, removed_signal_ref):.2e}")
