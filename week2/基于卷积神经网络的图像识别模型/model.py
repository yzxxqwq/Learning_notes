"""
CIFAR-10 卷积神经网络图像分类
Author: Zixiang Yang
Environment: pytorch_gpu (PyTorch)
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class TrainConfig:
    activation: str = "relu"
    kernel_size: int = 3
    stride: int = 1
    pool_type: str = "max"
    output_type: str = "softmax"
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def conv2d_output_size(size, kernel, stride=1, padding=0):
    return (size + 2 * padding - kernel) // stride + 1


def pool2d_output_size(size, kernel=2, stride=2):
    return (size - kernel) // stride + 1


def build_shape_trace(kernel_size, stride, pool_type):
    """生成默认 CNN 各层 shape 说明。"""
    h, w = 32, 32
    pad = kernel_size // 2
    lines = [f"Input: (B, 3, {h}, {w})"]

    h = conv2d_output_size(h, kernel_size, stride, pad)
    w = conv2d_output_size(w, kernel_size, stride, pad)
    lines.append(
        f"Conv1 ({kernel_size}x{kernel_size}, s={stride}, p={pad}): (B, 32, {h}, {w})"
    )
    lines.append(f"Activation")

    h = pool2d_output_size(h)
    w = pool2d_output_size(w)
    lines.append(f"{pool_type.capitalize()}Pool2d(2x2, s=2): (B, 32, {h}, {w})")

    h2 = conv2d_output_size(h, kernel_size, stride, pad)
    w2 = conv2d_output_size(w, kernel_size, stride, pad)
    lines.append(
        f"Conv2 ({kernel_size}x{kernel_size}, s={stride}, p={pad}): (B, 64, {h2}, {w2})"
    )
    lines.append(f"Activation")

    h2 = pool2d_output_size(h2)
    w2 = pool2d_output_size(w2)
    lines.append(f"{pool_type.capitalize()}Pool2d(2x2, s=2): (B, 64, {h2}, {w2})")

    flat = 64 * h2 * w2
    lines.append(f"Flatten: (B, {flat})")
    lines.append(f"Linear: (B, 10)")
    lines.append(f"Output ({'Softmax+CE' if True else 'Sigmoid'}): (B, 10)")
    return lines, flat, (h2, w2)


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def get_pool(name):
    name = name.lower()
    if name == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if name == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    raise ValueError(f"Unsupported pool type: {name}")


class CIFAR10CNN(nn.Module):
    """可配置的两层卷积网络，用于 CIFAR-10 消融实验。"""

    def __init__(
        self,
        activation="relu",
        kernel_size=3,
        stride=1,
        pool_type="max",
        num_classes=10,
    ):
        super().__init__()
        self.config = {
            "activation": activation,
            "kernel_size": kernel_size,
            "stride": stride,
            "pool_type": pool_type,
        }
        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, stride=stride, padding=padding),
            get_activation(activation),
            get_pool(pool_type),
            nn.Conv2d(32, 64, kernel_size, stride=stride, padding=padding),
            get_activation(activation),
            get_pool(pool_type),
        )

        h = w = 32
        for _ in range(2):
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)
            h = pool2d_output_size(h)
            w = pool2d_output_size(w)

        self.flatten_dim = 64 * h * w
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_dataloaders(batch_size=128, num_workers=0):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def get_loss_fn(output_type):
    output_type = output_type.lower()
    if output_type == "softmax":
        return nn.CrossEntropyLoss()
    if output_type == "sigmoid":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported output type: {output_type}")


def one_hot(labels, num_classes=10):
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(
        1, labels.view(-1, 1), 1.0
    )


def run_epoch(model, loader, criterion, optimizer, device, output_type, train=True):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(images)
        if output_type == "softmax":
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
        else:
            targets = one_hot(labels)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(config: TrainConfig, device, log_tb=False, verbose=True):
    set_seed(config.seed)
    train_loader, test_loader = get_dataloaders(config.batch_size)
    model = CIFAR10CNN(
        activation=config.activation,
        kernel_size=config.kernel_size,
        stride=config.stride,
        pool_type=config.pool_type,
    ).to(device)

    criterion = get_loss_fn(config.output_type)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    writer = None
    if log_tb:
        exp_name = (
            f"{config.activation}_k{config.kernel_size}_s{config.stride}_"
            f"{config.pool_type}_{config.output_type}"
        )
        writer = SummaryWriter(os.path.join(LOG_DIR, exp_name))

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, config.output_type, train=True
        )
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, config.output_type, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        best_acc = max(best_acc, test_acc)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/test", test_acc, epoch)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == config.epochs):
            print(
                f"Epoch {epoch:02d}/{config.epochs} | "
                f"train acc={train_acc:.4f} | test acc={test_acc:.4f}"
            )

    if writer:
        writer.close()

    elapsed = time.time() - start
    return {
        "config": asdict(config),
        "history": history,
        "best_test_acc": best_acc,
        "final_test_acc": history["test_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "elapsed_sec": elapsed,
        "flatten_dim": model.flatten_dim,
        "model": model,
    }


def experiment_suite(epochs=15):
    baseline = TrainConfig()
    return [
        ("baseline_relu", TrainConfig(epochs=epochs)),
        ("activation_sigmoid", TrainConfig(activation="sigmoid", epochs=epochs)),
        ("kernel_5", TrainConfig(kernel_size=5, epochs=epochs)),
        ("stride_2", TrainConfig(stride=2, epochs=epochs)),
        ("pool_avg", TrainConfig(pool_type="avg", epochs=epochs)),
        ("output_sigmoid", TrainConfig(output_type="sigmoid", epochs=epochs)),
    ]


def run_all_experiments(epochs=15, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    results = {}
    print(f"Device: {device}")
    print(f"Running {len(experiment_suite(epochs))} experiments, epochs={epochs}\n")

    for name, cfg in experiment_suite(epochs):
        print(f"===== {name} =====")
        out = train_model(cfg, device, log_tb=False, verbose=True)
        out.pop("model", None)
        results[name] = out
        print(
            f"Done: best test acc={results[name]['best_test_acc']:.4f}, "
            f"time={results[name]['elapsed_sec']:.1f}s\n"
        )

    result_path = os.path.join(RESULT_DIR, "experiment_results.json")
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    plot_experiment_results(results)
    save_results_csv(results)
    print(f"Results saved to {result_path}")
    return results


def plot_experiment_results(results):
    setup_plot_style()
    names = list(results.keys())
    labels = [
        "Baseline\n(ReLU)",
        "Sigmoid\n激活",
        "Kernel\n5x5",
        "Stride\n2",
        "Avg\nPool",
        "Sigmoid\n输出",
    ]
    accs = [results[n]["best_test_acc"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#0891b2", "#16a34a", "#ea580c"]
    bars = ax.bar(labels, accs, color=colors, edgecolor="white")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CIFAR-10 消融实验：测试集最佳准确率对比")
    ax.set_ylim(0, max(accs) + 8)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    baseline_acc = accs[0]
    ax.axhline(baseline_acc, color="#64748b", linestyle="--", linewidth=1.2, label="Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name, label in zip(axes.flat, names, labels):
        hist = results[name]["history"]
        epochs = range(1, len(hist["test_acc"]) + 1)
        ax.plot(epochs, [x * 100 for x in hist["train_acc"]], label="Train", color="#2563eb")
        ax.plot(epochs, [x * 100 for x in hist["test_acc"]], label="Test", color="#dc2626")
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("各实验训练/测试准确率曲线", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
    plot_grouped_analysis(results)


def plot_grouped_analysis(results):
    """按实验主题分组绘制对比图。"""
    setup_plot_style()
    baseline = results["baseline_relu"]["best_test_acc"] * 100

    groups = [
        ("激活函数对比", ["baseline_relu", "activation_sigmoid"], ["ReLU", "Sigmoid"], ["#2563eb", "#dc2626"]),
        (
            "卷积/池化参数对比",
            ["baseline_relu", "kernel_5", "stride_2", "pool_avg"],
            ["Baseline\n3x3,s1,max", "Kernel 5", "Stride 2", "Avg Pool"],
            ["#2563eb", "#7c3aed", "#0891b2", "#16a34a"],
        ),
        (
            "输出层对比",
            ["baseline_relu", "output_sigmoid"],
            ["Softmax+CE", "Sigmoid+BCE"],
            ["#2563eb", "#ea580c"],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (title, keys, labels, colors) in zip(axes, groups):
        accs = [results[k]["best_test_acc"] * 100 for k in keys]
        bars = ax.bar(labels, accs, color=colors, edgecolor="white")
        ax.axhline(baseline, color="#64748b", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(0, max(accs) + 10)
        ax.grid(axis="y", alpha=0.3)
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{acc:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("CIFAR-10 分组消融实验分析", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "grouped_analysis.png"), dpi=150)
    plt.close(fig)


def regenerate_plots():
    path = os.path.join(RESULT_DIR, "experiment_results.json")
    with open(path, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    plot_experiment_results(results)
    print(f"Plots regenerated in {FIG_DIR}")


def save_results_csv(results):
    path = os.path.join(RESULT_DIR, "experiment_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "experiment",
                "activation",
                "kernel_size",
                "stride",
                "pool_type",
                "output_type",
                "best_test_acc",
                "final_test_acc",
                "flatten_dim",
                "elapsed_sec",
            ]
        )
        for name, res in results.items():
            cfg = res["config"]
            writer.writerow(
                [
                    name,
                    cfg["activation"],
                    cfg["kernel_size"],
                    cfg["stride"],
                    cfg["pool_type"],
                    cfg["output_type"],
                    f"{res['best_test_acc']:.4f}",
                    f"{res['final_test_acc']:.4f}",
                    res["flatten_dim"],
                    f"{res['elapsed_sec']:.1f}",
                ]
            )


def train_baseline_and_save(epochs=20, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(epochs=epochs)
    result = train_model(cfg, device, log_tb=True, verbose=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULT_DIR, "best_model.pt")
    torch.save(
        {
            "model_state": result["model"].state_dict(),
            "config": asdict(cfg),
            "result": {
                "best_test_acc": result["best_test_acc"],
                "final_test_acc": result["final_test_acc"],
            },
        },
        ckpt_path,
    )
    result.pop("model", None)
    print(f"Baseline model checkpoint saved to {ckpt_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Experiments")
    parser.add_argument(
        "--mode",
        choices=["experiments", "baseline", "plot"],
        default="experiments",
        help="experiments: 消融实验; baseline: 训练 baseline; plot: 从 JSON 重绘图表",
    )
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.mode == "baseline":
        train_baseline_and_save(epochs=args.epochs, device=device)
    elif args.mode == "plot":
        regenerate_plots()
    else:
        run_all_experiments(epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
