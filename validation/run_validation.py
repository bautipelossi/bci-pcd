"""Compara nuestra implementación contra la referencia de Octave en Data.mat.
 Bautista Pelossi Schweizer

Uso, desde la raíz del repo:
    uv run python validation/run_validation.py [ruta_a_la_referencia.mat]

La referencia sale de correr el código original en Octave con la SAFB fija
(data/reference/octave_reference.mat). Es aproximada: el fminunc de Octave no
es el de MATLAB, así que en PCO esperamos diferencias.
"""

import sys

import numpy as np
from metrics import abs_cosine, channel_correlation, mean_vector_length, subspace_similarity
from pymatreader import read_mat

from src.pcd import PCD, select_artifact_components

REFERENCE = sys.argv[1] if len(sys.argv) > 1 else "data/reference/octave_reference.mat"


def summary(values: np.ndarray) -> str:
    return f"mediana {np.median(values):.4f} | mín {np.min(values):.4f} | máx {np.max(values):.4f}"


data = read_mat("data/Data.mat")["Data"]
ref = read_mat(REFERENCE)["ref_data"]
X_fit, z, X_toclean = data["X_tofit"], data["z"], data["X_toclean"]
sfreq = float(data["sf"])
band = (int(ref["fl"]), int(ref["fh"]))
k = int(ref["k"])

print(f"Data.mat: {X_fit.shape[0]} canales | SAFB {band} Hz | k = {k} (fijos, de la referencia)")
pcd = PCD(sfreq, signal_band=band, n_components=k, random_state=0).fit(X_fit, z)

# --- SSD -------------------------------------------------------------------
# comparamos los filtros en el espacio de los sensores (M @ W_ssd): así no
# importa el signo que cada implementación le dio a los autovectores de M
print("\n1) SSD")
cos_ssd = abs_cosine(pcd.M_ @ pcd.W_ssd_[:, :k], ref["M"] @ ref["W_ssd"][:, :k])
print(f"  |cos| entre filtros 1..{k}: {summary(cos_ssd)}")
print(f"  |cos| filtro por filtro: {np.round(cos_ssd, 3)}")
# PCO busca en todo el subespacio de los k filtros, así que eso es lo que importa
similarity = subspace_similarity(pcd.M_ @ pcd.W_ssd_[:, :k], ref["M"] @ ref["W_ssd"][:, :k])
print(f"  subespacio de los {k} filtros, cos de los ángulos principales: {summary(similarity)}")
lambda_ref = ref["lambda_ssd"][:k]
print(f"  lambda nuestro: {np.round(pcd.lambda_ssd_[:5], 4)}")
print(f"  lambda Octave : {np.round(lambda_ref[:5], 4)}")
print(f"  diferencia relativa de lambda (1..{k}): {summary(np.abs(pcd.lambda_ssd_[:k] - lambda_ref) / lambda_ref)}")

# --- PCO -------------------------------------------------------------------
print("\n2) PCO")
print(f"  MVL nuestro: {np.round(pcd.vlen_[:6], 3)}")
print(f"  MVL Octave : {np.round(ref['vlen'][:6], 3)}")
# si nuestra función da el MVL que reporta Octave para sus propios filtros, el
# objetivo es el mismo y lo que cambia es solo a qué máximo llega cada optimizador
mvl_check = [mean_vector_length(ref["W_pcd"][:, j] @ X_fit, z) for j in range(6)]
print(f"  MVL de los filtros de Octave recalculado acá: {np.round(mvl_check, 3)}")
pco_ours, pco_ref = pcd.W_pcd_[:, :k], ref["W_pcd"][:, :k]
print(f"  |cos| filtro PCD 1 (el del artefacto): {abs_cosine(pco_ours[:, 0], pco_ref[:, 0])[0]:.4f}")
# dos componentes con MVL parecido pueden salir en otro orden, así que
# también buscamos, para cada filtro de Octave, el nuestro más parecido
gram = np.abs(
    (pco_ref / np.linalg.norm(pco_ref, axis=0)).T @ (pco_ours / np.linalg.norm(pco_ours, axis=0))
)
for i in range(3):
    j = int(np.argmax(gram[i]))
    print(f"  filtro Octave {i + 1}: el más parecido es nuestro {j + 1} con |cos| {gram[i, j]:.4f}")
# lo que se saca de la señal es el subespacio de los patrones de artefacto
n_removed = len(np.setdiff1d(np.arange(1, k + 1), ref["idx_keep"]))
artifact_similarity = subspace_similarity(pcd.A_pcd_[:, :n_removed], ref["A_pcd"][:n_removed].T)
print(f"  subespacio de los {n_removed} patrones de artefacto, cos de los ángulos: {np.round(artifact_similarity, 4)}")

# --- Reconstrucción -------------------------------------------------------------
print("\n3) Reconstrucción sobre X_toclean")
removed_ref = np.setdiff1d(np.arange(1, k + 1), ref["idx_keep"]) - 1
X_ref = ref["X_clean"]
X_ours = pcd.apply(X_toclean, removed_ref)
print(f"  componentes sacadas (Octave, 'diff'): {removed_ref.tolist()} -> usamos las mismas")
print(f"  corr canal a canal de la señal limpia: {summary(channel_correlation(X_ours, X_ref))}")
# la señal limpia es casi toda la original, así que su correlación sale alta
# igual; lo que realmente compara las implementaciones es lo que cada una sacó
removed_signal_ours, removed_signal_ref = X_toclean - X_ours, X_toclean - X_ref
print(f"  corr canal a canal de lo removido: {summary(channel_correlation(removed_signal_ours, removed_signal_ref))}")
relative_error = np.linalg.norm(X_ours - X_ref) / np.linalg.norm(removed_signal_ref)
print(f"  ||limpia nuestra - limpia Octave|| / ||removido Octave||: {relative_error:.4f}")
print(f"  con nuestro propio 'diff' sacaríamos: {select_artifact_components(pcd.vlen_, 'diff').tolist()}")
