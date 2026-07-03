"""
烂番茄电影评论情感分析 — 自注意力 / BiLSTM 对比实验
Author: Zixiang Yang
Environment: pytorch_gpu (PyTorch)
Dataset: https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data
"""

import argparse
import csv
import io
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置数据/图像/结果输出的目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

TRAIN_URL = (
    "https://raw.githubusercontent.com/pratikmjoshi/"
    "sentiment-analysis-moviereviews/master/train.tsv"
)

# 情感类别名与类别数
SENTIMENT_NAMES = [
    "negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "positive",
]
NUM_CLASSES = 5
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# 训练配置参数定义
@dataclass
class TrainConfig:
    model_type: str = "self_attn"
    vocab_size: int = 10000
    embed_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    max_len: int = 50
    epochs: int = 6
    batch_size: int = 128
    lr: float = 2e-3
    max_samples: int = 50000
    seed: int = 42

# 设置matplotlib中文与字体正常
def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# 全局随机种子，保证实验可重复
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 基本英文分词
def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()

# 下载训练数据集（仅首次需要）
def download_train_tsv():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "train.tsv")
    if os.path.exists(path):
        return path
    print(f"Downloading train.tsv from GitHub mirror...")
    urllib.request.urlretrieve(TRAIN_URL, path)
    print(f"Saved to {path}")
    return path

# 加载数据/标签
def load_phrases(path):
    phrases, labels = [], []
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if not tokenize(row["Phrase"]):  # 跳过空文本
                continue
            phrases.append(row["Phrase"])
            labels.append(int(row["Sentiment"]))
    return phrases, labels

# 词典类，负责构建和映射（tok<->id）
class Vocabulary:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id2token = [PAD_TOKEN, UNK_TOKEN]

    # 基于训练集构建词表
    def build(self, texts):
        freq = {}
        for text in texts:
            for tok in tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        limit = self.max_size - len(self.token2id)
        for tok, _ in sorted_tokens[:limit]:
            if tok not in self.token2id:
                self.token2id[tok] = len(self.id2token)
                self.id2token.append(tok)

    # 文本转id、自动填充到max_len
    def encode(self, text, max_len):
        ids = [self.token2id.get(tok, 1) for tok in tokenize(text)]
        if not ids:
            ids = [1]
        ids = ids[:max_len]
        length = len(ids)
        ids = ids + [0] * (max_len - len(ids))
        return ids, length

    # OOV比例（未登录词率）
    def oov_rate(self, texts):
        total, oov = 0, 0
        for text in texts:
            for tok in tokenize(text):
                total += 1
                if tok not in self.token2id:
                    oov += 1
        return oov / max(total, 1)

    def __len__(self):
        return len(self.id2token)

# 自定义数据集，与torch DataLoader配合
class SentimentDataset(Dataset):
    def __init__(self, phrases, labels, vocab, max_len):
        self.phrases = phrases
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        ids, length = self.vocab.encode(self.phrases[idx], self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

# DataLoader批量拼接用
def collate_fn(batch):
    ids, lengths, labels = zip(*batch)
    return (
        torch.stack(ids),
        torch.stack(lengths),
        torch.stack(labels),
    )

# 划分训练/验证数据
def split_data(phrases, labels, dev_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(phrases))
    split = int(len(phrases) * (1 - dev_ratio))
    train_idx, dev_idx = idx[:split], idx[split:]
    train_p = [phrases[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]
    dev_p = [phrases[i] for i in dev_idx]
    dev_y = [labels[i] for i in dev_idx]
    return train_p, train_y, dev_p, dev_y

# 构建DataLoader、词表等，返回训练/验证集
def make_loaders(config: TrainConfig, vocab=None):
    path = download_train_tsv()
    phrases, labels = load_phrases(path)

    # 限定样本数以便快速实验
    if config.max_samples > 0 and config.max_samples < len(phrases):
        rng = np.random.default_rng(config.seed)
        idx = rng.choice(len(phrases), config.max_samples, replace=False)
        phrases = [phrases[i] for i in idx]
        labels = [labels[i] for i in idx]

    train_p, train_y, dev_p, dev_y = split_data(phrases, labels, seed=config.seed)

    # 仅根据训练集构建词表
    if vocab is None:
        vocab = Vocabulary(max_size=config.vocab_size)
        vocab.build(train_p)

    train_ds = SentimentDataset(train_p, train_y, vocab, config.max_len)
    dev_ds = SentimentDataset(dev_p, dev_y, vocab, config.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    oov_rate = vocab.oov_rate(dev_p)
    return train_loader, dev_loader, vocab, oov_rate

# 对变长序列在batch中做掩码平均池化
def masked_mean(x, lengths):
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).float()
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom

# 双向LSTM情感分类器
class BiLSTMClassifier(nn.Module):
    """串行模型：双向 LSTM 编码 + 最后时刻池化。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=0):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 双向LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        # 分类器
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        # pack处理变长输入，提高LSTM效率
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        # 拼接正反两个方向最后输出
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.dropout(self.fc(h_cat))

# 带自注意力的LSTM情感分类器
class SelfAttentiveClassifier(nn.Module):
    """
    并行自注意力模型（Lin et al. 2017 风格）：
    Embedding → BiLSTM → Self-Attention 加权句向量 → FC
    注：LSTM 为串行，注意力池化为并行加权；对比实验中与纯 BiLSTM 对照。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # BiLSTM编码每个时刻
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # 注意力权重W, U
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        h, _ = self.lstm(emb)
        u = torch.tanh(self.attn_w(h))
        scores = self.attn_u(u).squeeze(-1)
        # mask填充的pad部分，注意力为-inf，不参与归一化
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        # 注意力加权池化序列信息
        sent = (weights * h).sum(dim=1)
        return self.dropout(self.fc(sent))

# 纯多头自注意力情感分类器
class ParallelSelfAttentionClassifier(nn.Module):
    """纯并行模型：Embedding → Multi-Head Self-Attention → Masked Mean Pool。"""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        num_heads=4,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_heads = num_heads

    def forward(self, x, lengths):
        emb = self.embedding(x)
        h = self.input_proj(emb)
        # 生成padding mask，使填充不参与自注意力分数
        key_padding_mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[
            :, None
        ]
        attn_out, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = self.norm(h + attn_out)
        sent = masked_mean(h, lengths)  # mask平均池化
        return self.dropout(self.fc(sent))

# 根据配置选择构建模型
def build_model(config: TrainConfig, vocab_size):
    if config.model_type == "bilstm":
        return BiLSTMClassifier(vocab_size, config.embed_dim, config.hidden_dim, NUM_CLASSES)
    if config.model_type == "self_attn":
        return SelfAttentiveClassifier(
            vocab_size, config.embed_dim, config.hidden_dim, NUM_CLASSES
        )
    if config.model_type == "multi_head":
        return ParallelSelfAttentionClassifier(
            vocab_size,
            config.embed_dim,
            config.hidden_dim,
            NUM_CLASSES,
            num_heads=config.num_heads,
        )
    raise ValueError(f"Unknown model_type: {config.model_type}")

# 单个epoch训练/评估
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for ids, lengths, labels in loader:
        ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()

        logits = model(ids, lengths)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# 训练主循环，包含日志记录、保存最佳模型等
def train_model(config: TrainConfig, device, log_tb=False, exp_name="run"):
    set_seed(config.seed)
    train_loader, dev_loader, vocab, oov_rate = make_loaders(config)

    model = build_model(config, len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    writer = None
    if log_tb:
        writer = SummaryWriter(os.path.join(LOG_DIR, exp_name))

    history = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}
    best_acc, best_state = 0.0, None
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        dev_loss, dev_acc = run_epoch(
            model, dev_loader, criterion, optimizer, device, train=False
        )

        # 日志记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)

        # 记录最佳模型
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/dev", dev_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/dev", dev_acc, epoch)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train acc={train_acc:.4f} | dev acc={dev_acc:.4f}"
        )

    # 恢复验证集上最佳结果
    if best_state:
        model.load_state_dict(best_state)

    # TensorBoard词向量可视化
    if writer:
        sample_words = vocab.id2token[: min(500, len(vocab))]
        try:
            writer.add_embedding(
                model.embedding.weight[: len(sample_words)],
                metadata=sample_words,
                tag=f"{exp_name}_embeddings",
            )
        except Exception as exc:
            print(f"TensorBoard embedding skipped: {exc}")
        writer.close()

    elapsed = time.time() - start
    return {
        "config": asdict(config),
        "history": history,
        "best_dev_acc": best_acc,
        "final_dev_acc": history["dev_acc"][-1],
        "vocab_size_actual": len(vocab),
        "oov_rate_dev": oov_rate,
        "elapsed_sec": elapsed,
        "model": model,
        "vocab": vocab,
    }

# 导出词向量，用于TensorBoard Projector可交互可视化
def export_embeddings(model, vocab, path_prefix):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    weights = model.embedding.weight.detach().cpu().numpy()
    vec_path = path_prefix + "_vectors.tsv"
    meta_path = path_prefix + "_metadata.tsv"
    with open(vec_path, "w", encoding="utf-8") as fv, open(
        meta_path, "w", encoding="utf-8"
    ) as fm:
        fm.write("word\n")
        for i, word in enumerate(vocab.id2token):
            if i >= weights.shape[0]:
                break
            fm.write(f"{word}\n")
            fv.write("\t".join(f"{x:.6f}" for x in weights[i]) + "\n")
    print(f"Embeddings exported to {vec_path} (use with TensorBoard Projector)")

# 定义实验组设置
def experiment_suite():
    base = TrainConfig()
    return [
        ("baseline_bilstm", TrainConfig(model_type="bilstm")),
        ("parallel_self_attn", TrainConfig(model_type="self_attn")),
        ("multi_head_4", TrainConfig(model_type="multi_head", num_heads=4)),
        ("single_head_1", TrainConfig(model_type="multi_head", num_heads=1)),
        ("vocab_5000", TrainConfig(model_type="bilstm", vocab_size=5000)),
        ("vocab_20000", TrainConfig(model_type="bilstm", vocab_size=20000)),
    ]

# 批量运行所有组实验，保存结果及绘图
def run_all_experiments(device, epochs=6, max_samples=50000):
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}

    print(f"Device: {device}")
    print(f"Experiments: {len(experiment_suite())}, epochs={epochs}, samples={max_samples}\n")

    for name, cfg in experiment_suite():
        cfg.epochs = epochs
        cfg.max_samples = max_samples
        print(f"===== {name} =====")
        out = train_model(cfg, device, log_tb=(name == "parallel_self_attn"), exp_name=name)
        if name == "parallel_self_attn":
            export_embeddings(
                out["model"],
                out["vocab"],
                os.path.join(RESULT_DIR, "embedding_projector"),
            )
        out.pop("model", None)
        out.pop("vocab", None)
        results[name] = out
        print(f"Done: best dev acc={out['best_dev_acc']:.4f}, OOV={out['oov_rate_dev']:.4f}\n")

    result_path = os.path.join(RESULT_DIR, "experiment_results.json")
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    plot_results(results)
    save_csv(results)
    print(f"Results saved to {result_path}")
    return results

# 可视化整体、训练曲线
def plot_results(results):
    setup_plot_style()
    names = list(results.keys())
    labels = [
        "BiLSTM\n(串行)",
        "Self-Attn\n(LSTM+Attn)",
        "Multi-Head\n(h=4)",
        "Single-Head\n(h=1)",
        "Vocab\n5000",
        "Vocab\n20000",
    ]
    accs = [results[n]["best_dev_acc"] * 100 for n in names]
    oovs = [results[n]["oov_rate_dev"] * 100 for n in names]

    # 柱状图：各实验准确率&OOV
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(labels, accs, color="#2563eb", edgecolor="white")
    ax.set_ylabel("Dev Accuracy (%)")
    ax.set_title("各实验验证集准确率")
    ax.set_ylim(0, max(accs) + 8)
    ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center",
            fontsize=9,
        )

    ax = axes[1]
    ax.bar(labels, oovs, color="#dc2626", edgecolor="white")
    ax.set_ylabel("Dev OOV Rate (%)")
    ax.set_title("验证集未登录词 (OOV) 比例")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_summary.png"), dpi=150)
    plt.close(fig)

    # 训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name, label in zip(axes.flat, names, labels):
        hist = results[name]["history"]
        epochs = range(1, len(hist["dev_acc"]) + 1)
        ax.plot(epochs, [x * 100 for x in hist["train_acc"]], label="Train", color="#2563eb")
        ax.plot(epochs, [x * 100 for x in hist["dev_acc"]], label="Dev", color="#dc2626")
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("训练曲线对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

# 实验结果保存为csv汇总
def save_csv(results):
    path = os.path.join(RESULT_DIR, "experiment_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "experiment",
                "model_type",
                "vocab_size",
                "num_heads",
                "best_dev_acc",
                "oov_rate_dev",
                "elapsed_sec",
            ]
        )
        for name, res in results.items():
            cfg = res["config"]
            writer.writerow(
                [
                    name,
                    cfg["model_type"],
                    cfg["vocab_size"],
                    cfg["num_heads"],
                    f"{res['best_dev_acc']:.4f}",
                    f"{res['oov_rate_dev']:.4f}",
                    f"{res['elapsed_sec']:.1f}",
                ]
            )

# 命令行入口：可选实验全局运行/单模型训练/仅绘图
def main():
    parser = argparse.ArgumentParser(description="Rotten Tomatoes Sentiment Analysis")
    parser.add_argument(
        "--mode",
        choices=["experiments", "train", "plot"],
        default="experiments",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--model", default="self_attn", choices=["bilstm", "self_attn", "multi_head"])
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--num-heads", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.mode == "plot":
        with open(os.path.join(RESULT_DIR, "experiment_results.json"), encoding="utf-8") as fp:
            plot_results(json.load(fp))  # 仅重画图
        print("Plots regenerated.")
    elif args.mode == "experiments":
        run_all_experiments(device, epochs=args.epochs, max_samples=args.max_samples)
    else:
        # 单模型训练
        cfg = TrainConfig(
            model_type=args.model,
            vocab_size=args.vocab_size,
            num_heads=args.num_heads,
            epochs=args.epochs,
            max_samples=args.max_samples if args.max_samples > 0 else 0,
        )
        out = train_model(cfg, device, log_tb=True, exp_name="main_train")
        export_embeddings(out["model"], out["vocab"], os.path.join(RESULT_DIR, "embedding_projector"))
        torch.save(out["model"].state_dict(), os.path.join(RESULT_DIR, "best_model.pt"))
        print(f"Best dev acc: {out['best_dev_acc']:.4f}")

# 脚本入口
if __name__ == "__main__":
    main()
