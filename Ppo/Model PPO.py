# %% [markdown]
"""
# PPO untuk Klasifikasi Normal vs DoS

Mengotomasi seluruh pipeline: memanfaatkan artefak preprocessing (`prepared_split.joblib`),
menjalankan kombinasi hyperparameter + evaluasi K-Fold, dan melakukan retrain penuh
berdasarkan eksperimen terbaik.
"""

# %%
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


# %%
# Konstanta utama: lokasi artefak preprocessing dan folder output PPO


def resolve_project_root(marker: str = "Preprocessing") -> Path:
    """Temukan root proyek dengan mencari folder marker (mis. Preprocessing)."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    # Tambahkan parent-parent agar bisa menemukan root dari berbagai lokasi
    for base in list(candidates):
        parent = base.parent
        if parent not in candidates:
            candidates.append(parent)
        grandparent = parent.parent
        if grandparent not in candidates:
            candidates.append(grandparent)

    seen = set()
    for base in candidates:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        if (base / marker).exists():
            return base
    raise FileNotFoundError(f"Tidak ditemukan folder '{marker}' mulai dari {Path.cwd().resolve()}")


PROJECT_ROOT = resolve_project_root()
ARTIFACT_PATH = PROJECT_ROOT / "Preprocessing" / "artifacts" / "prepared_split.joblib"
OUTPUT_DIR = PROJECT_ROOT / "Ppo" / "ppo_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(f"Artefak preprocessing tidak ditemukan di {ARTIFACT_PATH.resolve()}")


# %%
# Memuat data train/test dari joblib
artifacts = joblib.load(ARTIFACT_PATH)


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "values"):
        arr = arr.values
    return np.asarray(arr, dtype=np.float32)


X_train = _to_numpy(artifacts["X_train_prepared"])
X_test = _to_numpy(artifacts["X_test_prepared"])
y_train = np.asarray(artifacts["y_train"], dtype=np.int64)
y_test = np.asarray(artifacts["y_test"], dtype=np.int64)
feature_names = artifacts.get("feature_names")

print(f"Shape train: {X_train.shape}, test: {X_test.shape}")
print(f"Distribusi label train: {np.bincount(y_train)} | test: {np.bincount(y_test)}")

CLASS_NAMES = {0: "Normal", 1: "DoS"}
classes = np.array(sorted(np.unique(y_train)))
class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
CLASS_WEIGHT_DICT = {int(cls): float(weight) for cls, weight in zip(classes, class_weight_values)}
max_label_index = int(max(CLASS_WEIGHT_DICT.keys()))
weight_array = np.ones(max_label_index + 1, dtype=np.float32)
for cls, weight in CLASS_WEIGHT_DICT.items():
    weight_array[int(cls)] = weight
CLASS_WEIGHT_TENSOR = torch.tensor(weight_array, dtype=torch.float32, device=DEVICE)

print("Class weight per label:")
for cls, weight in CLASS_WEIGHT_DICT.items():
    label_name = CLASS_NAMES.get(int(cls), str(cls))
    print(f"  {label_name} ({cls}): {weight:.4f}")


# %%
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% [markdown]

## Konfigurasi PPO & Spesifikasi Eksperimen



# %%
@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.08
    entropy_coef: float = 0.03
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    episodes: int = 200
    batch_size: int = 256
    update_interval: int = 5
    hidden_size: int = 256
    reward_correct: float = 0.2
    reward_wrong: float = -0.2
    supervised_coef: float = 0.3


@dataclass
class ExperimentSpec:
    name: str
    config_overrides: Dict[str, object]
    n_splits: int = 5
    notes: Optional[str] = None


def _format_value_for_name(key: str, value: object) -> str:
    if isinstance(value, float):
        return f"{key}{value:.0e}".replace("+", "")
    if isinstance(value, (int, np.integer)):
        return f"{key}{int(value)}"
    if isinstance(value, bool):
        return f"{key}{int(value)}"
    return f"{key}{str(value).replace(' ', '')}"


def generate_experiments_from_grid(
    param_grid: Dict[str, List[object]],
    fold_options: List[int],
    notes_template: Optional[str] = None,
) -> List[ExperimentSpec]:
    if not param_grid:
        return []
    keys = sorted(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    folds = fold_options or [5]
    experiments: List[ExperimentSpec] = []
    for values in itertools.product(*value_lists):
        overrides = dict(zip(keys, values))
        name_core = "_".join(_format_value_for_name(k, overrides[k]) for k in keys)
        for n_splits in folds:
            exp_name = f"{name_core}_fold{n_splits}"
            note = (
                notes_template.format(**overrides, n_splits=n_splits)
                if notes_template
                else None
            )
            experiments.append(
                ExperimentSpec(
                    name=exp_name,
                    config_overrides=overrides,
                    n_splits=n_splits,
                    notes=note,
                )
            )
    return experiments


def apply_config_overrides(config: PPOConfig, overrides: Dict[str, object]) -> PPOConfig:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"PPOConfig tidak memiliki atribut '{key}'")
        setattr(config, key, value)
    return config


def cleanup_experiment_dir(path: str | None, root_dir: Path = OUTPUT_DIR):
    """Hapus folder percobaan jika bukan folder utama yang harus dipertahankan."""
    if not path:
        return
    target = Path(path).resolve()
    root_dir = root_dir.resolve()
    if target == root_dir:
        return
    if target.exists():
        print(f"[INFO] Menghapus folder percobaan: {target}")
        shutil.rmtree(target, ignore_errors=True)


# %% [markdown]

## Arsitektur PPO (Actor-Critic) & Komponen Pendukung



# %%
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int, action_dim: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values

# %%
class PPOMemory:
    def __init__(self):
        self.clear()

    def store(self, states, actions, rewards, log_probs, values, labels):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.labels.append(labels)

    def get_batch(self):
        return (
            torch.cat(self.states),
            torch.cat(self.actions),
            torch.cat(self.rewards),
            torch.cat(self.log_probs),
            torch.cat(self.values),
            torch.cat(self.labels),
        )

    def clear(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

# %%
class PPOClassifier:
    def __init__(self, state_dim: int, config: PPOConfig, class_weights: Optional[torch.Tensor] = None):
        self.config = config
        self.model = ActorCritic(state_dim, config.hidden_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.memory = PPOMemory()
        self.class_weights: Optional[torch.Tensor] = None
        if class_weights is not None:
            self.class_weights = class_weights.detach().clone().to(DEVICE)
        self.last_val_history: Dict[str, List[float]] = {"accuracy": [], "f1": [], "episodes": []}

    def _sample_minibatch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(len(X), self.config.batch_size, replace=True)
        return X[idx], y[idx]

    def _gather_experience(self, X: np.ndarray, y: np.ndarray):
        X_batch, y_batch = self._sample_minibatch(X, y)
        states = torch.tensor(X_batch, dtype=torch.float32, device=DEVICE)
        labels = torch.tensor(y_batch, dtype=torch.long, device=DEVICE)
        logits, values = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        rewards = torch.where(
            actions == labels,
            torch.full_like(actions, self.config.reward_correct, dtype=torch.float32),
            torch.full_like(actions, self.config.reward_wrong, dtype=torch.float32),
        )
        self.memory.store(
            states.detach(),
            actions.detach(),
            rewards.detach(),
            log_probs.detach(),
            values.detach(),
            labels.detach(),
        )
        metrics = {
            "batch_acc": float((actions == labels).float().mean().item()),
            "batch_reward": float(rewards.mean().item()),
        }
        return metrics

    def _ppo_update(self):
        states, actions, rewards, old_log_probs, values, labels = self.memory.get_batch()
        returns = rewards
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies = [], [], []
        num_samples = states.size(0)
        for _ in range(self.config.num_epochs):
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                logits, new_values = self.model(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)
                entropy = dist.entropy().mean()
                if self.class_weights is not None:
                    ce_loss = F.cross_entropy(logits, labels[batch_idx], weight=self.class_weights)
                else:
                    ce_loss = F.cross_entropy(logits, labels[batch_idx])
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.supervised_coef * ce_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.memory.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        episodes: int,
        eval_data: Optional[np.ndarray] = None,
        eval_labels: Optional[np.ndarray] = None,
        eval_interval: int = 2,
    ) -> Dict[str, List[float]]:
        history = {"accuracy": [], "reward": [], "policy_loss": [], "value_loss": [], "entropy": []}
        val_history: Dict[str, List[float]] = {"accuracy": [], "f1": [], "episodes": []}
        best_f1 = 0.0
        patience_counter = 0
        patience = 12
        
        for episode in range(1, episodes + 1):
            batch_metrics = self._gather_experience(X, y)
            update_stats = self._ppo_update()
            history["accuracy"].append(batch_metrics["batch_acc"])
            history["reward"].append(batch_metrics["batch_reward"])
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])
            # Early stopping sederhana berdasarkan F1 batch
            if batch_metrics["batch_acc"] > best_f1:
                best_f1 = batch_metrics["batch_acc"]
                patience_counter = 0
            else:
                patience_counter += 1
            if eval_data is not None and eval_labels is not None:
                should_eval = False
                if eval_interval > 0 and episode % eval_interval == 0:
                    should_eval = True
                if episode == 1 and not val_history["episodes"]:
                    should_eval = True
                if should_eval:
                    eval_probs = self.predict_proba(eval_data)
                    eval_pred = np.argmax(eval_probs, axis=1)
                    val_history["accuracy"].append(float(accuracy_score(eval_labels, eval_pred)))
                    val_history["f1"].append(float(f1_score(eval_labels, eval_pred, zero_division=0)))
                    val_history["episodes"].append(episode)
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at episode {episode} (patience reached).")
                break
            if episode % 1 == 0:
                print(
                    f"[Episode {episode:04d}] acc={batch_metrics['batch_acc']:.3f} "
                    f"reward={batch_metrics['batch_reward']:.3f} "
                    f"loss={update_stats['policy_loss']:.3f}/{update_stats['value_loss']:.3f}"
                )
        if self.memory.states:
            final_stats = self._ppo_update()
            history["policy_loss"][-1] = final_stats["policy_loss"]
            history["value_loss"][-1] = final_stats["value_loss"]
            history["entropy"][-1] = final_stats["entropy"]
        if (
            eval_data is not None
            and eval_labels is not None
            and (not val_history["episodes"] or val_history["episodes"][-1] != episode)
        ):
            eval_probs = self.predict_proba(eval_data)
            eval_pred = np.argmax(eval_probs, axis=1)
            val_history["accuracy"].append(float(accuracy_score(eval_labels, eval_pred)))
            val_history["f1"].append(float(f1_score(eval_labels, eval_pred, zero_division=0)))
            val_history["episodes"].append(episode)
        self.last_val_history = val_history
        return history

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=DEVICE)
                logits, _ = self.model(batch)
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": asdict(self.config)}, path)


# %% [markdown]

## Fungsi Evaluasi & Visualisasi



# %%
sns.set_palette("husl")
if "seaborn-whitegrid" in plt.style.available:
    plt.style.use("seaborn-whitegrid")
else:
    plt.style.use("default")


def plot_training_history(history: Dict[str, List[float]], title: str, out_dir: Path, file_stem: Optional[str] = None):
    episodes = list(range(1, len(history["accuracy"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Training History - {title}", fontsize=14, fontweight="bold")

    axes[0, 0].plot(episodes, history["accuracy"], color="royalblue", linewidth=1.2)
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Accuracy")

    axes[0, 1].plot(episodes, history["reward"], color="forestgreen", linewidth=1.2)
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")

    axes[1, 0].plot(episodes, history["policy_loss"], label="Policy Loss", color="crimson", linewidth=1.0)
    axes[1, 0].plot(episodes, history["value_loss"], label="Value Loss", color="darkorange", linewidth=1.0)
    axes[1, 0].set_title("Training Losses")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, history["entropy"], color="purple", linewidth=1.0)
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Entropy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = slugify(file_stem if file_stem else f"{title}_training_history")
    fig_path = out_dir / f"{stem}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_validation_metrics(
    val_history: Dict[str, List[float]],
    title: str,
    out_dir: Path,
    file_stem: Optional[str] = None,
) -> Optional[Path]:
    accuracy_values = val_history.get("accuracy", [])
    if not accuracy_values:
        return None
    episodes = val_history.get("episodes") or list(range(1, len(accuracy_values) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Validation Metrics - {title}", fontsize=14, fontweight="bold")
    axes[0].plot(
        episodes,
        accuracy_values,
        marker="o",
        color="royalblue",
        linewidth=1.8,
        markersize=5,
    )
    axes[0].set_xlabel("Evaluation Point")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Validation Accuracy")
    axes[0].grid(True, alpha=0.3)

    f1_values = val_history.get("f1", [])
    if f1_values:
        axes[1].plot(
            episodes,
            f1_values,
            marker="s",
            color="seagreen",
            linewidth=1.8,
            markersize=5,
        )
        axes[1].set_xlabel("Evaluation Point")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Validation F1 Score")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    stem = slugify(file_stem if file_stem else f"{title}_validation_metrics")
    fig_path = out_dir / f"{stem}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_dir: Path,
    file_stem: Optional[str] = None,
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5), facecolor="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "DoS"], yticklabels=["Normal", "DoS"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    stem = slugify(file_stem if file_stem else f"{title}_cm")
    fig_path = out_dir / f"{stem}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return cm, fig_path


def plot_roc_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str,
    out_dir: Path,
    file_stem: Optional[str] = None,
):
    roc_auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    stem = slugify(file_stem if file_stem else title)

    plt.figure(figsize=(6, 5), facecolor="white")
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {title}\nAUC={roc_auc:.4f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    roc_path = out_dir / f"{stem}_roc.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5), facecolor="white")
    plt.plot(rec, prec, label=f"AP={ap:.4f}")
    plt.title(f"Precision-Recall - {title}\nAP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    pr_path = out_dir / f"{stem}_pr.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return {"roc_auc": roc_auc, "average_precision": ap, "roc_path": roc_path, "pr_path": pr_path}


def slugify(name: str) -> str:
    lowered = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", lowered)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "plot"


def evaluate_split(
    model: PPOClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    out_dir: Path,
    export_tag: Optional[str] = None,
):
    probs = model.predict_proba(X)
    y_scores = probs[:, 1]
    y_pred = np.argmax(probs, axis=1)
    report_dict = classification_report(y, y_pred, target_names=["Normal", "DoS"], output_dict=True, zero_division=0)
    report_text = classification_report(y, y_pred, target_names=["Normal", "DoS"], digits=4, zero_division=0)
    tag = export_tag or split_name
    cm, cm_path = plot_confusion(
        y,
        y_pred,
        f"{split_name} Confusion Matrix",
        out_dir,
        file_stem=f"{tag}_confusion_matrix",
    )
    curve_info = plot_roc_pr(y, y_scores, split_name, out_dir, file_stem=tag)
    print(f"\n[{split_name}] Classification Report:\n{report_text}")
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": curve_info["roc_auc"],
        "average_precision": curve_info["average_precision"],
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_path": str(cm_path),
        "roc_path": str(curve_info["roc_path"]),
        "pr_path": str(curve_info["pr_path"]),
    }
    return metrics


def save_classification_report(report: Dict[str, Any], base_name: str, output_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Persist classification report to JSON/CSV similar to binary pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_name}_classification_report.json"
    csv_path = output_dir / f"{base_name}_classification_report.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    rows = []
    for label, scores in report.items():
        if isinstance(scores, dict):
            row = {"label": label}
            row.update(scores)
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        csv_result: Optional[Path] = csv_path
    else:
        csv_result = None
    return json_path, csv_result


def plot_kfold_summary(
    fold_results: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Buat ringkasan visual (ROC-AUC / AP / F1) per fold."""
    metrics = ["roc_auc", "ap", "f1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("K-Fold Cross-Validation Summary", fontsize=14, fontweight="bold")
    for idx, metric in enumerate(metrics):
        values = [r.get(metric, 0.0) for r in fold_results]
        folds = list(range(1, len(values) + 1))
        axes[idx].bar(folds, values, color="steelblue", alpha=0.7, edgecolor="black")
        if values:
            axes[idx].axhline(
                y=float(np.mean(values)),
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {np.mean(values):.4f}",
            )
        axes[idx].set_xlabel("Fold")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f"{metric.upper()} per Fold")
        axes[idx].set_xticks(folds)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "kfold_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


# %%
def run_kfold_experiment(
    X: np.ndarray,
    y: np.ndarray,
    config: PPOConfig,
    n_splits: int = 5,
    run_name: str = "kfold",
    output_root: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_dir = output_root / run_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -float("inf")
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*70}\n[ K-FOLD ] Fold {fold}/{n_splits}\n{'='*70}")
        trainer = PPOClassifier(state_dim=X.shape[1], config=config, class_weights=CLASS_WEIGHT_TENSOR)
        history = trainer.train(
            X[train_idx],
            y[train_idx],
            config.episodes,
            eval_data=X[val_idx],
            eval_labels=y[val_idx],
        )
        val_history = trainer.last_val_history
        plot_training_history(
            history,
            f"{run_name}_fold{fold}",
            fold_dir,
            file_stem=f"fold_{fold}_training_history",
        )
        plot_validation_metrics(
            val_history,
            f"{run_name} Fold {fold}",
            fold_dir,
            file_stem=f"fold_{fold}_validation_metrics",
        )

        metrics = evaluate_split(
            trainer,
            X[val_idx],
            y[val_idx],
            f"{run_name} Fold {fold}",
            fold_dir,
            export_tag=f"fold_{fold}",
        )
        fold_metrics.append(metrics)

        if metrics["roc_auc"] > best_metric:
            best_metric = metrics["roc_auc"]
            best_state = trainer.model.state_dict()

    plot_kfold_summary(
        [
            {"roc_auc": m["roc_auc"], "ap": m["average_precision"], "f1": m["f1"]}
            for m in fold_metrics
        ],
        output_dir=fold_dir,
    )

    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    mean_roc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
    mean_ap = float(np.mean([m["average_precision"] for m in fold_metrics]))

    summary = {
        "fold_metrics": fold_metrics,
        "mean_metrics": {
            "accuracy": mean_acc,
            "f1": mean_f1,
            "roc_auc": mean_roc,
            "average_precision": mean_ap,
        },
    }

    with open(fold_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_state is not None:
        torch.save(
            {"state_dict": best_state, "config": asdict(config)},
            fold_dir / f"{run_name}_best_model.pt",
        )

    return summary


# %%
def retrain_full_dataset(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test_eval: np.ndarray,
    y_test_eval: np.ndarray,
    config: PPOConfig,
    run_label: str,
    output_dir: Path,
) -> Dict[str, object]:
    """Latih ulang model terbaik menggunakan seluruh dataset agar siap deploy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = PPOClassifier(state_dim=X_train_full.shape[1], config=config, class_weights=CLASS_WEIGHT_TENSOR)
    history = trainer.train(
        X_train_full,
        y_train_full,
        config.episodes,
        eval_data=X_test_eval,
        eval_labels=y_test_eval,
    )
    val_history = trainer.last_val_history
    history_path = plot_training_history(
        history,
        f"{run_label}_full",
        output_dir,
        file_stem="retrain_training_history",
    )
    validation_plot = plot_validation_metrics(
        val_history,
        f"{run_label} Retrain",
        output_dir,
        file_stem="retrain_validation_metrics",
    )
    test_metrics = evaluate_split(
        trainer,
        X_test_eval,
        y_test_eval,
        f"{run_label} Retrain Test",
        output_dir,
        export_tag="retrain_test",
    )
    report_json, report_csv = save_classification_report(
        test_metrics["classification_report"],
        f"{run_label}_retrain_test",
        output_dir,
    )
    model_path = output_dir / f"{run_label}_full_model.pt"
    trainer.save(model_path)
    return {
        "history_plot": str(history_path),
        "model_path": str(model_path),
        "test_metrics": test_metrics,
        "validation_plot": str(validation_plot) if validation_plot else None,
        "classification_report_paths": {
            "json": str(report_json),
            "csv": str(report_csv) if report_csv else None,
        },
    }


# %%
# %% [markdown]

## Konfigurasi Grid & Parameter Kontrol


# %%
HYPERPARAMETER_GRID: Dict[str, List[object]] = {
    "learning_rate": [1e-4, 3e-4],
    "clip_epsilon": [0.08, 0.12],
    "entropy_coef": [0.005, 0.01],
    "batch_size": [128, 256],
    "value_coef": [0.5, 0.7],
}
FOLD_OPTIONS: List[int] = [2, 5]
GRID_NOTES_TEMPLATE: Optional[str] = "lr={learning_rate}, clip={clip_epsilon}, entropy={entropy_coef}, fold={n_splits}"
AUTO_RETRAIN_METRIC: Optional[str] = "f1"


# %% [markdown]

## Eksekusi Utama (Grid → Ringkasan → Retrain)

# %%
def main() -> None:
    """
    Jalankan keseluruhan eksperimen (grid hyperparameter + evaluasi tambahan).
    Fungsi ini menjaga alur utama tetap rapi dan mudah dipanggil.
    """
    set_seed(RANDOM_STATE)
    X_train_data = X_train
    y_train_data = y_train
    X_test_data = X_test
    y_test_data = y_test

    experiments = generate_experiments_from_grid(
        HYPERPARAMETER_GRID,
        FOLD_OPTIONS,
        GRID_NOTES_TEMPLATE,
    )
    if not experiments:
        experiments = [
            ExperimentSpec(
                name="default_cfg",
                config_overrides={},
                n_splits=FOLD_OPTIONS[0] if FOLD_OPTIONS else 5,
            )
        ]

    summary_payload = []

    total_experiments = len(experiments)

    for idx, spec in enumerate(experiments, 1):
        print("\n" + "-" * 70)
        print(f"[INFO] Experiment {idx}/{total_experiments}: {spec.name}")
        if spec.notes:
            print(f"[INFO] Catatan: {spec.notes}")
        config = apply_config_overrides(PPOConfig(), dict(spec.config_overrides))
        print(f"[INFO] Konfigurasi: {config}")
        print(
            f"[INFO] Dataset shape: train={X_train.shape}, test={X_test.shape}, "
            f"features used={X_train.shape[1]}"
        )
        print("-" * 70)
        kfold_run_name = f"{spec.name}_kfold"
        kfold_summary = run_kfold_experiment(
            X_train_data,
            y_train_data,
            config=config,
            n_splits=spec.n_splits,
            run_name=kfold_run_name,
            output_root=OUTPUT_DIR,
        )

        run_dir = OUTPUT_DIR / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer = PPOClassifier(state_dim=X_train_data.shape[1], config=config, class_weights=CLASS_WEIGHT_TENSOR)
        history = trainer.train(
            X_train_data,
            y_train_data,
            config.episodes,
            eval_data=X_test_data,
            eval_labels=y_test_data,
        )
        val_history = trainer.last_val_history
        history_path = plot_training_history(
            history,
            spec.name,
            run_dir,
            file_stem="training_history",
        )
        validation_path = plot_validation_metrics(
            val_history,
            f"{spec.name} Validation",
            run_dir,
            file_stem="validation_metrics",
        )
        test_metrics_full = evaluate_split(
            trainer,
            X_test_data,
            y_test_data,
            f"{spec.name} Test",
            run_dir,
            export_tag="test",
        )
        model_path = run_dir / "ppo_model.pt"
        trainer.save(model_path)

        save_classification_report(test_metrics_full["classification_report"], f"{spec.name}_test", run_dir)

        summary_payload.append(
            {
                "name": spec.name,
                "n_splits": spec.n_splits,
                "notes": spec.notes,
                "config": asdict(config),
                "kfold_summary": kfold_summary,
                "test_metrics": {k: v for k, v in test_metrics_full.items() if k not in {"classification_report"}},
                "artifacts": {
                    "run_dir": str(run_dir.resolve()),
                    "model_path": str(model_path.resolve()),
                    "history_plot": str(history_path.resolve()),
                    "validation_plot": str(validation_path.resolve()) if validation_path else None,
                    "kfold_dir": str((OUTPUT_DIR / kfold_run_name).resolve()),
                },
            }
        )

    timestamp = int(time.time())
    summary_path = OUTPUT_DIR / f"ppo_experiments_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    summary_csv_path = OUTPUT_DIR / f"ppo_experiments_summary_{timestamp}.csv"
    rows = []
    for entry in summary_payload:
        kfold_means = entry["kfold_summary"]["mean_metrics"]
        test_metrics = entry["test_metrics"]
        rows.append(
            {
                "name": entry["name"],
                "notes": entry["notes"],
                "n_splits": entry["n_splits"],
                "config": json.dumps(entry["config"]),
                "kfold_accuracy": kfold_means.get("accuracy"),
                "kfold_f1": kfold_means.get("f1"),
                "kfold_roc_auc": kfold_means.get("roc_auc"),
                "kfold_average_precision": kfold_means.get("average_precision"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_f1": test_metrics.get("f1"),
                "test_roc_auc": test_metrics.get("roc_auc"),
                "test_average_precision": test_metrics.get("average_precision"),
            }
        )
    pd.DataFrame(rows).to_csv(summary_csv_path, index=False)

    metric_field_map = {
        "f1": ("kfold_summary", "mean_metrics", "f1"),
        "roc_auc": ("kfold_summary", "mean_metrics", "roc_auc"),
        "ap": ("kfold_summary", "mean_metrics", "average_precision"),
    }

    print(f"\nRingkasan eksperimen tersimpan di: {summary_path.resolve()}")
    for entry in summary_payload:
        kfold_means = entry["kfold_summary"]["mean_metrics"]
        test_metrics = entry["test_metrics"]
        print(
            f"- {entry['name']}: kfold_acc={kfold_means['accuracy']:.4f}, "
            f"kfold_f1={kfold_means['f1']:.4f}, "
            f"kfold_ROC-AUC={kfold_means['roc_auc']:.4f}; "
            f"test_acc={test_metrics['accuracy']:.4f}, test_f1={test_metrics['f1']:.4f}"
        )

    if AUTO_RETRAIN_METRIC:
        path_tuple = metric_field_map.get(AUTO_RETRAIN_METRIC.lower())
        if path_tuple:
            def extract_metric(entry):
                data = entry
                for key in path_tuple:
                    data = data[key]
                return data

            best_entry = max(summary_payload, key=extract_metric)
            best_value = extract_metric(best_entry)
            print(
                f"\n[INFO] Eksperimen terbaik menurut {AUTO_RETRAIN_METRIC.upper()}: "
                f"{best_entry['name']} ({AUTO_RETRAIN_METRIC.upper()}={best_value:.4f})"
            )
        else:
            print(
                f"[INFO] AUTO_RETRAIN_METRIC='{AUTO_RETRAIN_METRIC}' tidak dikenali, "
                "menggunakan F1 sebagai default."
            )
            best_entry = max(summary_payload, key=lambda entry: entry["test_metrics"]["f1"])
            best_value = best_entry["test_metrics"]["f1"]
            print(
                f"[INFO] Eksperimen terbaik berdasarkan F1: "
                f"{best_entry['name']} (F1={best_value:.4f})"
            )
    else:
        best_entry = max(summary_payload, key=lambda entry: entry["test_metrics"]["f1"])
        best_value = best_entry["test_metrics"]["f1"]
        print(
            f"[INFO] AUTO_RETRAIN_METRIC tidak diset; memilih F1 tertinggi "
            f"({best_entry['name']} dengan F1={best_value:.4f})."
        )

    # Hapus folder percobaan lain agar hanya eksperimen terbaik yang tersisa
    for entry in summary_payload:
        if entry is best_entry:
            continue
        cleanup_experiment_dir(entry["artifacts"].get("run_dir"), OUTPUT_DIR)
        cleanup_experiment_dir(entry["artifacts"].get("kfold_dir"), OUTPUT_DIR)

    print("\n[INFO] Melakukan retrain penuh menggunakan konfigurasi terbaik...")
    retrain_dir = OUTPUT_DIR / f"{best_entry['name']}_full_retrain"
    retrain_result = retrain_full_dataset(
        X_train_data,
        y_train_data,
        X_test_data,
        y_test_data,
        config=apply_config_overrides(PPOConfig(), dict(best_entry["config"])),
        run_label=best_entry["name"],
        output_dir=retrain_dir,
    )
    retrain_metrics: Dict[str, float] = retrain_result["test_metrics"]  # type: ignore[assignment]
    print(
        f"[INFO] Retrain selesai. Model tersimpan di {retrain_result['model_path']}, "
        f"akurasi full data={retrain_metrics['accuracy']:.4f}, "
        f"F1={retrain_metrics['f1']:.4f}"
    )


# %%
if __name__ == "__main__":
    main()
