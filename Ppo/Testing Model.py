# %% [markdown]
# # PPO Testing Notebook

# %% [markdown]
# Notebook ini menyiapkan subset data serangan, menerapkan pipeline preprocessing, dan menjalankan model PPO yang sudah dilatih.
#
#
# Langkah cepat:
#
# 1. Pastikan artefak preprocessing (Preprocessing/artifacts/preprocessing_pipeline.joblib) tersedia.
#
# 2. Notebook otomatis akan mencari checkpoint *_full_model.pt dari hasil retrain penuh di ppo_outputs. Jika ingin model lain, isi variabel SELECTED_MODEL dengan path yang diinginkan.
#
# 3. Jalankan sel satu per satu untuk mengambil 5 sampel serangan dan melihat prediksi model.

# %%
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

try:
    import joblib
except ModuleNotFoundError as exc:
    raise RuntimeError("Install joblib terlebih dahulu: pip install joblib") from exc


# %%
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# UBAH INI kalau mau testing file lain (misal bnat_kdd_like.csv / testing_bnat.csv)
DATA_PATH = BASE_DIR.parent / "Dataset" / "testing 1.csv"

PIPELINE_PATH = BASE_DIR.parent / "Preprocessing" / "artifacts" / "preprocessing_pipeline.joblib"

MODEL_DIR = BASE_DIR / "ppo_outputs"
FULL_MODEL_CANDIDATES = sorted(MODEL_DIR.glob("*_full_retrain/*_full_model.pt"))
DEFAULT_MODEL_PATH = FULL_MODEL_CANDIDATES[0] if FULL_MODEL_CANDIDATES else None

# kalau mau model tertentu:
SELECTED_MODEL = None

FILTER_ATTACK_ONLY = False
N_SAMPLES = None
RANDOM_STATE = 42
SHUFFLE_BEFORE_SAMPLE = False  # True jika ingin ambil sampel acak

# Kalau kamu yakin mapping labelnya: 1=DoS, 0=Normal, biarkan.
# Kalau belum yakin, set None agar script menampilkan dua interpretasi.
DOS_CLASS_ID = 1  # set None untuk tampilkan dua interpretasi

if SELECTED_MODEL:
    MODEL_PATH = Path(SELECTED_MODEL)
elif DEFAULT_MODEL_PATH is not None:
    MODEL_PATH = DEFAULT_MODEL_PATH
else:
    MODEL_PATH = MODEL_DIR / "path_to_model.pt"

print(f"Dataset path : {DATA_PATH}")
print(f"Pipeline path: {PIPELINE_PATH}")
print("Model candidates (full retrain):")
for idx, c in enumerate(FULL_MODEL_CANDIDATES, 1):
    print(f"  {idx}. {c}")
print(f"Model path yang digunakan: {MODEL_PATH}")


# %%
df = pd.read_csv(DATA_PATH)

# deteksi kolom label (lebih robust)
label_candidates = ["Label_encoded", "label_encoded", "Label", "label", "Class", "class", "target", "Target"]
label_col = next((c for c in label_candidates if c in df.columns), None)

if label_col is None:
    print("Dataset tidak memiliki kolom label; prediksi akan dilaporkan tanpa label referensi.")
else:
    print(f"Label column terdeteksi: {label_col}")

# opsional: filter serangan saja
if FILTER_ATTACK_ONLY and label_col is not None:
    s = df[label_col]
    if pd.api.types.is_numeric_dtype(s):
        # asumsi 0 = benign/normal, 1 = attack
        attack_mask = s != 0
    else:
        # string label
        attack_mask = s.astype(str).str.lower().isin(["dos", "attack", "malicious", "1", "true"]) | (
            s.astype(str).str.lower() != "benign"
        )
    df = df[attack_mask].copy()
    if df.empty:
        raise ValueError("Tidak ditemukan baris serangan pada dataset setelah filter.")

# sampling
if SHUFFLE_BEFORE_SAMPLE:
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

sample_size = N_SAMPLES if N_SAMPLES is not None else len(df)
sample_size = min(sample_size, len(df))
sample_df = df.iloc[:sample_size].reset_index(drop=True)

print(f"Mengambil {len(sample_df)} baris untuk pengujian.")
if label_col is not None:
    print(sample_df[[label_col]].assign(sample_index=sample_df.index).head(20))
else:
    print(sample_df.assign(sample_index=sample_df.index)[["sample_index"]].head(20))


# %%
def load_preprocessor_and_features():
    artifact = joblib.load(PIPELINE_PATH)
    preprocessor = None
    feature_cols = None

    if isinstance(artifact, dict):
        preprocessor = artifact.get("pipeline") or artifact.get("preprocessor") or artifact.get("transformer")
        feature_cols = artifact.get("feature_cols") or artifact.get("selected_features") or artifact.get("feature_names")
    else:
        preprocessor = artifact
        feature_cols = getattr(preprocessor, "feature_names_in_", None)

    if preprocessor is None:
        raise ValueError("Preprocessor tidak ditemukan di artefak joblib.")
    if feature_cols is not None:
        feature_cols = list(feature_cols)

    return preprocessor, feature_cols


def align_features(df_features: pd.DataFrame, feature_cols: list[str] | None):
    """Paksa kolom sesuai fitur training. Missing diisi default. Extra dibuang."""
    if feature_cols is None:
        return df_features

    aligned = df_features.reindex(columns=feature_cols)

    # isi default untuk missing / NaN
    for c in aligned.columns:
        if c in ["protocol_type", "service", "flag"]:
            aligned[c] = aligned[c].fillna("unknown").astype(str)
        else:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(0.0)

    return aligned


def transform_samples(df_sample: pd.DataFrame, label_column: str | None):
    preprocessor, feature_cols = load_preprocessor_and_features()

    df_features = df_sample.drop(columns=[label_column], errors="ignore") if label_column else df_sample.copy()
    df_features = align_features(df_features, feature_cols)

    transformed = preprocessor.transform(df_features)
    transformed = np.asarray(transformed, dtype=np.float32)
    return transformed, feature_cols


X_sample, feature_reference = transform_samples(sample_df, label_col)
print("Shape fitur sampel:", X_sample.shape)


# %%
if not MODEL_PATH.exists():
    raise FileNotFoundError("Perbarui MODEL_PATH agar menunjuk ke checkpoint yang valid.")

module_path = BASE_DIR / "Model PPO.py"
spec = importlib.util.spec_from_file_location("ppo_impl", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Tidak dapat memuat modul PPO dari {module_path}")

import sys
ppo_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ppo_impl
spec.loader.exec_module(ppo_impl)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
config = ppo_impl.PPOConfig(**checkpoint["config"])

tester = ppo_impl.PPOClassifier(state_dim=X_sample.shape[1], config=config, class_weights=None)
tester.model.load_state_dict(checkpoint["state_dict"])
tester.model.eval()
print("Model PPO siap untuk inference.")


# %%
probs = tester.predict_proba(X_sample)
probs = np.asarray(probs)

# sanity check
print("probs[0:5] =\n", probs[:5])
print("sum probs row[0:5] =", probs[:5].sum(axis=1))

pred_ids = probs.argmax(axis=1)
print("unique pred_ids:", np.unique(pred_ids, return_counts=True))

result_df = sample_df.copy()
result_df["predicted_class"] = pred_ids

if probs.ndim == 2 and probs.shape[1] >= 2:
    result_df["p0"] = probs[:, 0]
    result_df["p1"] = probs[:, 1]

    if DOS_CLASS_ID is None:
        # tampilkan dua interpretasi agar tidak salah mapping
        result_df["status_if_DoS_is_1"] = np.where(pred_ids == 1, "Serangan DoS", "Normal")
        result_df["status_if_DoS_is_0"] = np.where(pred_ids == 0, "Serangan DoS", "Normal")
    else:
        # interpretasi tunggal
        p_attack = probs[:, DOS_CLASS_ID]
        result_df["p_attack"] = p_attack
        result_df["status"] = np.where(pred_ids == DOS_CLASS_ID, "Serangan DoS", "Normal") + \
                              " (p_attack=" + result_df["p_attack"].round(4).astype(str) + ")"
else:
    # fallback jika model hanya keluarkan 1 kolom
    result_df["status"] = result_df["predicted_class"].map({1: "Serangan DoS", 0: "Normal"}).fillna("Unknown")

# tampilkan ringkas
cols_to_show = []
if label_col is not None and label_col in result_df.columns:
    cols_to_show.append(label_col)
cols_to_show += ["predicted_class"]
for c in ["p0", "p1", "p_attack", "status", "status_if_DoS_is_1", "status_if_DoS_is_0"]:
    if c in result_df.columns:
        cols_to_show.append(c)

try:
    from IPython.display import display
    display(result_df[cols_to_show].style.hide(axis="index"))
except Exception:
    print(result_df[cols_to_show].to_string(index=False))
