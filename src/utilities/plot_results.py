import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for SLURM/headless
import matplotlib.pyplot as plt


# result.csv column indices (both mAP and acc modes share the same layout,
# only col 0 and col 7 differ in meaning)
_COL_PRIMARY   = 0   # acc (acc mode) or mAP (mAP mode)
_COL_AUC       = 1
_COL_AVG_PREC  = 2
_COL_AVG_REC   = 3
_COL_DPRIME    = 4
_COL_TRAIN_LOSS = 5
_COL_VAL_LOSS  = 6
_COL_CUM_PRIMARY = 7  # cum_acc or cum_mAP
_COL_CUM_AUC   = 8
_COL_LR        = 9


def plot_training_results(exp_dir, main_metrics="acc"):
    """Plot training curves and eval summary from result.csv / eval_result.csv.

    Saves PNG figures to exp_dir/figures/.

    Args:
        exp_dir: path to the experiment directory (same as --exp-dir in run.py)
        main_metrics: "acc" or "mAP" — controls axis labels
    """
    fig_dir = os.path.join(exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    result_path = os.path.join(exp_dir, "result.csv")
    eval_path   = os.path.join(exp_dir, "eval_result.csv")

    # ------------------------------------------------------------------ #
    # Training curves from result.csv
    # ------------------------------------------------------------------ #
    if os.path.exists(result_path):
        result = np.loadtxt(result_path, delimiter=",")
        if result.ndim == 1:
            result = result[np.newaxis, :]  # single-epoch edge case

        # Only plot rows that have been filled (non-zero primary metric)
        filled = np.any(result != 0, axis=1)
        result = result[filled]
        epochs = np.arange(1, len(result) + 1)
        metric_label = "Accuracy" if main_metrics == "acc" else "mAP"

        # --- Loss ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_TRAIN_LOSS], label="Train Loss")
        ax.plot(epochs, result[:, _COL_VAL_LOSS],   label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "loss.png"), dpi=150)
        plt.close(fig)

        # --- Primary metric ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_PRIMARY],     label=metric_label)
        ax.plot(epochs, result[:, _COL_CUM_PRIMARY], label=f"Cumulative {metric_label}", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{main_metrics}.png"), dpi=150)
        plt.close(fig)

        # --- AUC ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_AUC],     label="AUC")
        ax.plot(epochs, result[:, _COL_CUM_AUC], label="Cumulative AUC", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("AUC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "auc.png"), dpi=150)
        plt.close(fig)

        # --- Precision / Recall ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_AVG_PREC], label="Avg Precision")
        ax.plot(epochs, result[:, _COL_AVG_REC],  label="Avg Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Precision & Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "precision_recall.png"), dpi=150)
        plt.close(fig)

        # --- d-prime ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_DPRIME], label="d-prime", color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("d-prime")
        ax.set_title("d-prime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "dprime.png"), dpi=150)
        plt.close(fig)

        # --- Learning rate ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, result[:, _COL_LR], label="Learning Rate", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "lr.png"), dpi=150)
        plt.close(fig)

        print(f"[plot_results] Training curves saved to {fig_dir}")

    else:
        print(f"[plot_results] result.csv not found at {result_path}, skipping training curves.")

    # ------------------------------------------------------------------ #
    # Eval summary from eval_result.csv
    # ------------------------------------------------------------------ #
    if os.path.exists(eval_path):
        eval_result = np.loadtxt(eval_path)
        # layout: [val_acc, val_mAUC, eval_acc, eval_mAUC]
        labels = ["Val Acc", "Val mAUC", "Eval Acc", "Eval mAUC"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, eval_result, color=["steelblue", "steelblue", "darkorange", "darkorange"])
        for bar, val in zip(bars, eval_result):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Final Evaluation Results")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "eval_summary.png"), dpi=150)
        plt.close(fig)
        print(f"[plot_results] Eval summary saved to {fig_dir}")

    else:
        print(f"[plot_results] eval_result.csv not found at {eval_path}, skipping eval summary.")


# ------------------------------------------------------------------ #
# Self-test with dummy data matching the real CSV format
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile

    print("Running self-test with dummy data...")
    n_epochs = 30
    rng = np.random.default_rng(42)

    # Simulate realistic training curves
    t = np.linspace(0, 1, n_epochs)
    train_loss    = 2.5 * np.exp(-3 * t) + 0.05 * rng.random(n_epochs)
    val_loss      = 2.8 * np.exp(-2.5 * t) + 0.08 * rng.random(n_epochs)
    acc           = 0.95 * (1 - np.exp(-4 * t)) + 0.01 * rng.random(n_epochs)
    auc           = 0.97 * (1 - np.exp(-4 * t)) + 0.008 * rng.random(n_epochs)
    avg_precision = acc * 0.98 + 0.005 * rng.random(n_epochs)
    avg_recall    = acc * 0.96 + 0.005 * rng.random(n_epochs)
    dprime        = 0.5 + 2.5 * t + 0.05 * rng.random(n_epochs)
    cum_acc       = np.array([acc[:i+1].mean() for i in range(n_epochs)])
    cum_auc       = np.array([auc[:i+1].mean() for i in range(n_epochs)])
    lr            = np.array([2.5e-4 * (0.85 ** max(0, i - 5)) for i in range(n_epochs)])

    # result.csv: 10 columns, n_epochs rows
    result = np.stack(
        [acc, auc, avg_precision, avg_recall, dprime,
         train_loss, val_loss, cum_acc, cum_auc, lr],
        axis=1,
    )
    # eval_result.csv: 4 values
    eval_result = np.array([acc[-1], auc[-1], acc[-1] * 0.99, auc[-1] * 0.98])

    with tempfile.TemporaryDirectory() as tmp:
        np.savetxt(os.path.join(tmp, "result.csv"),      result,      delimiter=",")
        np.savetxt(os.path.join(tmp, "eval_result.csv"), eval_result)

        plot_training_results(tmp, main_metrics="acc")

        expected = ["loss.png", "acc.png", "auc.png", "precision_recall.png",
                    "dprime.png", "lr.png", "eval_summary.png"]
        all_ok = True
        for fname in expected:
            path = os.path.join(tmp, "figures", fname)
            exists = os.path.exists(path)
            size   = os.path.getsize(path) if exists else 0
            status = "PASS" if (exists and size > 0) else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  [{status}] {fname} ({size} bytes)")

    print("Self-test", "PASSED" if all_ok else "FAILED")
