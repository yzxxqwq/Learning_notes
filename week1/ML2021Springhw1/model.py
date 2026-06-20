"""
ML2021 Spring HW1: COVID-19 Daily Cases Prediction
Author: Zixiang Yang
Date: 2026-06-20
Description: This is a simple implementation of a shallow DNN for COVID-19 daily cases prediction.
Best dev RMSE: 0.9039
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_STATES = 40
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(DATA_DIR, "figures")


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_csv(path):
    with open(path, "r", newline="") as fp:
        rows = list(csv.reader(fp))
    return np.array(rows[1:])[:, 1:].astype(np.float64)


def get_feature_indices(mode="medium"):
    """
    medium: 40 州 one-hot + 第 1、2 天的 tested_positive
    strong: 40 州 + 3 天 cli/ili/hh_cmnty_cli/nohh_cmnty_cli + 第 1、2 天 tested_positive
    all: 全部 93 维特征（不含标签）
    """
    day1_tp = N_STATES + 17
    day2_tp = N_STATES + 18 + 17

    if mode == "medium":
        return list(range(N_STATES)) + [day1_tp, day2_tp]

    if mode == "strong":
        cli_ili = []
        for day in range(3):
            base = N_STATES + day * 18
            cli_ili.extend([base, base + 1, base + 2, base + 3])
        return list(range(N_STATES)) + cli_ili + [day1_tp, day2_tp]

    if mode == "all":
        return list(range(93))

    raise ValueError(f"Unknown feature mode: {mode}")


def prepare_datasets(
    train_path,
    test_path,
    feature_mode="strong",
    dev_ratio=10,
):
    raw_train = load_csv(train_path)
    raw_test = load_csv(test_path)

    feat_idx = get_feature_indices(feature_mode)
    n_state = N_STATES

    x_all = raw_train[:, feat_idx]
    y_all = raw_train[:, -1]
    x_test = raw_test[:, feat_idx]

    dev_mask = np.arange(len(x_all)) % dev_ratio == 0
    train_mask = ~dev_mask

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_dev, y_dev = x_all[dev_mask], y_all[dev_mask]

    x_train, x_dev, x_test, mean, std = normalize_features(
        x_train, x_dev, x_test, n_state
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_dev": x_dev,
        "y_dev": y_dev,
        "x_test": x_test,
        "feat_idx": feat_idx,
        "norm_mean": mean,
        "norm_std": std,
    }


def normalize_features(x_train, x_dev, x_test, n_state):
    mean = x_train[:, n_state:].mean(axis=0)
    std = x_train[:, n_state:].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    def _norm(x):
        out = x.copy()
        out[:, n_state:] = (out[:, n_state:] - mean) / std
        return out

    return _norm(x_train), _norm(x_dev), _norm(x_test), mean, std


def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


class MLP:
    """两层全连接网络：Linear -> ReLU -> Linear。"""
    def __init__(self, input_dim, hidden_dim, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1)),
            "b2": np.zeros(1),
        }
        self.cache = {}

    def forward(self, x):
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        self.cache = {"x": x, "z1": z1, "a1": a1}
        return z2.squeeze(-1)

    def backward(self, y, l2_lambda):
        n = y.shape[0]
        x, z1, a1 = self.cache["x"], self.cache["z1"], self.cache["a1"]
        y_pred = (a1 @ self.params["W2"] + self.params["b2"]).squeeze(-1)

        dz2 = (2.0 / n) * (y_pred - y)[:, None]
        dW2 = a1.T @ dz2 + 2 * l2_lambda * self.params["W2"]
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1 + 2 * l2_lambda * self.params["W1"]
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict(self, x):
        a1 = np.maximum(0.0, x @ self.params["W1"] + self.params["b1"])
        return (a1 @ self.params["W2"] + self.params["b2"]).squeeze(-1)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = {k: v.copy() for k, v in params.items()}


def iterate_minibatches(x, y, batch_size, rng):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]


def train(
    x_train,
    y_train,
    x_dev,
    y_dev,
    hidden_dim=128,
    lr=1e-3,
    batch_size=135,
    epochs=300,
    momentum=0.9,
    l2_lambda=5e-4,
    patience=30,
    seed=42,
):
    model = MLP(x_train.shape[1], hidden_dim, seed=seed)
    velocity = {k: np.zeros_like(v) for k, v in model.get_params().items()}

    best_state = None
    best_dev_rmse = float("inf")
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    history = {"epoch": [], "train_rmse": [], "dev_rmse": []}

    for epoch in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            model.forward(xb)
            grads = model.backward(yb, l2_lambda)
            params = model.get_params()
            for key in params:
                velocity[key] = momentum * velocity[key] - lr * grads[key]
                params[key] += velocity[key]
            model.set_params(params)

        train_pred = model.predict(x_train)
        dev_pred = model.predict(x_dev)
        train_rmse = rmse(train_pred, y_train)
        dev_rmse = rmse(dev_pred, y_dev)

        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["dev_rmse"].append(dev_rmse)

        if dev_rmse < best_dev_rmse:
            best_dev_rmse = dev_rmse
            best_state = {k: v.copy() for k, v in model.get_params().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | train RMSE: {train_rmse:.4f} | dev RMSE: {dev_rmse:.4f}"
            )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.set_params(best_state)
    print(f"Best dev RMSE: {best_dev_rmse:.4f}")
    return model, best_dev_rmse, history


def save_submission(path, predictions):
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, pred in enumerate(predictions):
            writer.writerow([i, pred])


def plot_training_curve(history, save_path):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    epochs = history["epoch"]
    ax.plot(epochs, history["train_rmse"], label="Train RMSE", color="#2563eb", linewidth=2)
    ax.plot(epochs, history["dev_rmse"], label="Dev RMSE", color="#dc2626", linewidth=2)

    best_idx = int(np.argmin(history["dev_rmse"]))
    best_epoch = epochs[best_idx]
    best_rmse = history["dev_rmse"][best_idx]
    ax.scatter([best_epoch], [best_rmse], color="#16a34a", s=80, zorder=5)
    ax.annotate(
        f"Best: epoch {best_epoch}, RMSE={best_rmse:.4f}",
        xy=(best_epoch, best_rmse),
        xytext=(best_epoch + max(epochs) * 0.05, best_rmse + 0.3),
        arrowprops={"arrowstyle": "->", "color": "#16a34a"},
        fontsize=10,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("COVID-19 预测模型训练曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_prediction_analysis(y_true, y_pred, title, save_path):
    setup_plot_style()
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)
    r = rmse(y_pred, y_true)
    mae = float(np.mean(abs_errors))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#2563eb", edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#64748b", linewidth=1.5, label="理想预测 y=x")
    ax.set_xlabel("真实值 tested_positive (%)")
    ax.set_ylabel("预测值 tested_positive (%)")
    ax.set_title(f"{title}\nRMSE={r:.4f}, MAE={mae:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.45, s=18, color="#dc2626", edgecolors="none")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.5)
    ax.set_xlabel("预测值 tested_positive (%)")
    ax.set_ylabel("残差 (预测 - 真实)")
    ax.set_title("残差图")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.hist(residuals, bins=30, color="#7c3aed", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.5)
    ax.set_xlabel("残差 (预测 - 真实)")
    ax.set_ylabel("样本数")
    ax.set_title("残差分布")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"预测分析 - {title}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_prediction_summary(test_pred, save_path):
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(test_pred, bins=35, color="#0891b2", alpha=0.85, edgecolor="white")
    ax.axvline(test_pred.mean(), color="#dc2626", linestyle="--", linewidth=1.5, label=f"均值={test_pred.mean():.2f}")
    ax.axvline(np.median(test_pred), color="#16a34a", linestyle="--", linewidth=1.5, label=f"中位数={np.median(test_pred):.2f}")
    ax.set_xlabel("预测 tested_positive (%)")
    ax.set_ylabel("样本数")
    ax.set_title(f"测试集预测分布 (n={len(test_pred)})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    sorted_pred = np.sort(test_pred)
    ax.plot(sorted_pred, color="#2563eb", linewidth=2)
    ax.fill_between(range(len(sorted_pred)), sorted_pred, alpha=0.15, color="#2563eb")
    ax.set_xlabel("样本序号（按预测值排序）")
    ax.set_ylabel("预测 tested_positive (%)")
    ax.set_title("测试集预测值排序曲线")
    ax.grid(True, alpha=0.3)

    fig.suptitle("测试集预测分析（无真实标签）", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(history, y_train, train_pred, y_dev, dev_pred, test_pred, save_path):
    setup_plot_style()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history["epoch"], history["train_rmse"], label="Train RMSE", color="#2563eb", linewidth=2)
    ax1.plot(history["epoch"], history["dev_rmse"], label="Dev RMSE", color="#dc2626", linewidth=2)
    best_idx = int(np.argmin(history["dev_rmse"]))
    ax1.scatter(
        [history["epoch"][best_idx]],
        [history["dev_rmse"][best_idx]],
        color="#16a34a",
        s=70,
        zorder=5,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE")
    ax1.set_title("训练曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for ax, y_true, y_hat, name, color in [
        (fig.add_subplot(gs[1, 0]), y_train, train_pred, "Train", "#2563eb"),
        (fig.add_subplot(gs[1, 1]), y_dev, dev_pred, "Dev", "#dc2626"),
    ]:
        ax.scatter(y_true, y_hat, alpha=0.4, s=14, color=color, edgecolors="none")
        lo = min(y_true.min(), y_hat.min())
        hi = max(y_true.max(), y_hat.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#64748b", linewidth=1.2)
        ax.set_xlabel("真实值 (%)")
        ax.set_ylabel("预测值 (%)")
        ax.set_title(f"{name} 集: RMSE={rmse(y_hat, y_true):.4f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"COVID-19 预测可视化汇总 | 测试集预测均值={test_pred.mean():.2f}%, "
        f"范围=[{test_pred.min():.2f}, {test_pred.max():.2f}]",
        fontsize=14,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    train_path = os.path.join(DATA_DIR, "covid.train.csv")
    test_path = os.path.join(DATA_DIR, "covid.test.csv")
    submit_path = os.path.join(DATA_DIR, "submission.csv")

    print("Loading and preprocessing data...")
    data = prepare_datasets(train_path, test_path, feature_mode="strong")
    print(f"Train: {data['x_train'].shape}, Dev: {data['x_dev'].shape}, Test: {data['x_test'].shape}")
    print(f"Feature dim: {data['x_train'].shape[1]}")

    print("\nTraining model...")
    model, best_dev_rmse, history = train(
        data["x_train"],
        data["y_train"],
        data["x_dev"],
        data["y_dev"],
    )

    train_pred = model.predict(data["x_train"])
    dev_pred = model.predict(data["x_dev"])
    print(f"\nFinal train RMSE: {rmse(train_pred, data['y_train']):.4f}")
    print(f"Final dev RMSE:   {rmse(dev_pred, data['y_dev']):.4f}")

    test_pred = model.predict(data["x_test"])
    save_submission(submit_path, test_pred)
    print(f"\nPredictions saved to: {submit_path}")
    print(f"Test samples: {len(test_pred)}")

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_training_curve(history, os.path.join(FIG_DIR, "training_curve.png"))
    plot_prediction_analysis(
        data["y_train"],
        train_pred,
        "Train 集",
        os.path.join(FIG_DIR, "prediction_train.png"),
    )
    plot_prediction_analysis(
        data["y_dev"],
        dev_pred,
        "Dev 集",
        os.path.join(FIG_DIR, "prediction_dev.png"),
    )
    plot_test_prediction_summary(
        test_pred,
        os.path.join(FIG_DIR, "prediction_test.png"),
    )
    plot_summary_dashboard(
        history,
        data["y_train"],
        train_pred,
        data["y_dev"],
        dev_pred,
        test_pred,
        os.path.join(FIG_DIR, "summary_dashboard.png"),
    )
    print(f"\n可视化图表已保存至: {FIG_DIR}")
    print("  - training_curve.png      训练/验证 RMSE 曲线")
    print("  - prediction_train.png    训练集预测分析")
    print("  - prediction_dev.png      验证集预测分析")
    print("  - prediction_test.png     测试集预测分布")
    print("  - summary_dashboard.png   汇总仪表盘")


if __name__ == "__main__":
    main()
