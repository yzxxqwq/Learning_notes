"""
信用卡交易欺诈检测 — GCN / GAT 图神经网络
Author: Zixiang Yang
Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection/
"""

import argparse
import csv
import gc
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 数据、图片、结果保存路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 示例数据url
SAMPLE_URL = (
    "https://raw.githubusercontent.com/Vignesh-Hariharan/"
    "fraud-detection-pipeline/main/data/sample_transactions.csv"
)

# 图缓存
_GRAPH_CACHE = {}


@dataclass
class TrainConfig:
    # 训练配置参数，包括模型类型与超参数
    model_type: str = "gcn"
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    k_neighbors: int = 8
    dropout: float = 0.3
    lr: float = 1e-2
    weight_decay: float = 5e-4
    epochs: int = 80
    patience: int = 15
    max_nodes: int = 6000
    fraud_oversample: float = 3.0
    seed: int = 42


def setup_plot_style():
    # 设置matplotlib全局风格
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    # 固定随机种子保证可复现
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def haversine(lat1, lon1, lat2, lon2):
    # haversine 公式计算地球球面距离
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def download_sample_csv():
    # 下载示例数据到本地
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "sample_transactions.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def load_transactions():
    """优先加载本地 fraudTrain.csv，否则生成模拟数据。"""
    for name in ["fraudTrain.csv", "fraudTest.csv"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return _read_csv(path), name

    print("未找到 Kaggle 数据，生成模拟交易数据（分布与真实集一致）...")
    return _generate_synthetic(60000), "synthetic"


def _read_csv(path):
    # 读取csv到dict组成的列表
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _generate_synthetic(n=60000, fraud_rate=0.005, seed=42):
    # 生成模拟交易样本
    rng = np.random.default_rng(seed)
    categories = [
        "grocery_pos", "misc_pos", "gas_transport", "shopping_net",
        "shopping_pos", "entertainment", "food_dining", "travel",
    ]
    rows = []
    for i in range(n):
        is_fraud = int(rng.random() < fraud_rate)
        if is_fraud:
            amt = float(np.exp(rng.normal(5.5, 1.2)))
            hour = int(rng.choice(list(range(0, 6)) + list(range(22, 24))))
            dist = float(rng.uniform(200, 2000))
        else:
            amt = float(np.exp(rng.normal(3.8, 0.9)))
            hour = int(rng.integers(8, 22))
            dist = float(rng.uniform(0, 50))

        cc = str(rng.integers(100000000000, 999999999999))
        rows.append({
            "cc_num": cc,
            "category": str(rng.choice(categories)),
            "amt": f"{amt:.2f}",
            "lat": f"{rng.uniform(25, 48):.4f}",
            "long": f"{rng.uniform(-125, -70):.4f}",
            "merch_lat": f"{rng.uniform(25, 48):.4f}",
            "merch_long": f"{rng.uniform(-125, -70):.4f}",
            "city_pop": str(int(rng.integers(1000, 500000))),
            "gender": str(rng.choice(["M", "F"])),
            "trans_date_trans_time": f"2020-{rng.integers(1,13):02d}-{rng.integers(1,28):02d} {hour:02d}:00:00",
            "dob": f"{rng.integers(1960, 2000)}-01-01",
            "is_fraud": str(is_fraud),
        })
    return rows


def build_features(rows):
    """特征工程，返回 X, y, cc_nums。"""
    amounts, hours, dists, ages, cats, genders, labels, cc_nums = [], [], [], [], [], [], [], []

    for row in rows:
        try:
            amt = float(row["amt"])
            lat = float(row["lat"])
            lon = float(row["long"])
            mlat = float(row["merch_lat"])
            mlon = float(row["merch_long"])
            ts = row.get("trans_date_trans_time", "2020-01-01 12:00:00")
            hour = int(ts[11:13]) if len(ts) >= 13 else 12
            dob_year = int(str(row.get("dob", "1980-01-01"))[:4])
            age = 2020 - dob_year
            dist = haversine(lat, lon, mlat, mlon)
            label = int(row["is_fraud"])
        except (ValueError, KeyError):
            continue

        # 聚合重要特征至各自列表
        amounts.append(np.log1p(amt))
        hours.append(hour)
        dists.append(dist)
        ages.append(age)
        cats.append(row.get("category", "misc"))
        genders.append(row.get("gender", "M"))
        labels.append(label)
        cc_nums.append(str(row.get("cc_num", str(len(cc_nums)))))

    # 编码类别特征
    cat_enc = LabelEncoder().fit(cats)
    gen_enc = LabelEncoder().fit(genders)

    X = np.column_stack([
        amounts,
        hours,
        dists,
        ages,
        cat_enc.transform(cats),
        gen_enc.transform(genders),
    ]).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, np.array(cc_nums)


def subsample_balanced(X, y, cc_nums, max_nodes, fraud_oversample, seed):
    # 欺诈样本过采样并控制总节点数量
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    if len(idx) <= max_nodes:
        return X, y, cc_nums

    fraud_idx = idx[y == 1]
    normal_idx = idx[y == 0]
    n_fraud = len(fraud_idx)
    n_fraud_target = min(n_fraud, int(n_fraud * fraud_oversample))
    n_normal_target = max_nodes - n_fraud_target

    sel_fraud = fraud_idx if n_fraud_target >= n_fraud else rng.choice(
        fraud_idx, n_fraud_target, replace=False
    )
    sel_normal = rng.choice(normal_idx, min(n_normal_target, len(normal_idx)), replace=False)
    sel = np.concatenate([sel_fraud, sel_normal])
    rng.shuffle(sel)
    return X[sel], y[sel], cc_nums[sel]


def build_adjacency(X, cc_nums, k=8):
    """kNN 图 + 同 cc_num 连边，返回对称归一化邻接矩阵。"""
    n = X.shape[0]
    # k近邻找边
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    rows, cols = [], []
    for i in range(n):
        for j in indices[i]:
            if i != j:
                rows.extend([i, j])
                cols.extend([j, i])

    # 相同信用卡号全连边
    cc_map = {}
    for i, cc in enumerate(cc_nums):
        cc_map.setdefault(cc, []).append(i)
    for nodes in cc_map.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                rows.extend([i, j, j, i])
                cols.extend([j, i, i, j])

    # 构造稀疏邻接矩阵
    data = np.ones(len(rows), dtype=np.float32)
    adj = torch.sparse_coo_tensor(
        torch.LongTensor([rows, cols]),
        torch.FloatTensor(data),
        (n, n),
    ).coalesce()

    # 归一化邻接矩阵（对称归一化）
    adj = adj.to_dense()
    adj = adj + torch.eye(n)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return adj_norm


def prepare_graph_data(config: TrainConfig):
    # 图数据准备（含缓存）
    cache_key = (config.max_nodes, config.k_neighbors, config.fraud_oversample, config.seed)
    if cache_key in _GRAPH_CACHE:
        cached = _GRAPH_CACHE[cache_key]
        print(
            f"Data: {cached['source']} (cached), nodes={cached['n']}, "
            f"fraud_rate={cached['fraud_rate']:.4f}, "
            f"train/val/test={cached['counts']}"
        )
        return (
            cached["features"].clone(),
            cached["labels"].clone(),
            cached["adj"].clone(),
            {k: v.clone() for k, v in cached["masks"].items()},
            cached["fraud_rate"],
        )

    rows, source = load_transactions()
    X, y, cc_nums = build_features(rows)
    X, y, cc_nums = subsample_balanced(
        X, y, cc_nums, config.max_nodes, config.fraud_oversample, config.seed
    )

    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=config.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, stratify=y[train_idx], random_state=config.seed
    )

    adj = build_adjacency(X, cc_nums, k=config.k_neighbors)
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y)
    masks = {
        "train": torch.zeros(len(y), dtype=torch.bool),
        "val": torch.zeros(len(y), dtype=torch.bool),
        "test": torch.zeros(len(y), dtype=torch.bool),
    }
    masks["train"][train_idx] = True
    masks["val"][val_idx] = True
    masks["test"][test_idx] = True

    fraud_rate = y.mean()
    print(
        f"Data: {source}, nodes={len(y)}, fraud_rate={fraud_rate:.4f}, "
        f"train/val/test={masks['train'].sum()}/{masks['val'].sum()}/{masks['test'].sum()}"
    )
    _GRAPH_CACHE[cache_key] = {
        "source": source,
        "n": len(y),
        "fraud_rate": float(fraud_rate),
        "counts": (
            int(masks["train"].sum()),
            int(masks["val"].sum()),
            int(masks["test"].sum()),
        ),
        "features": features,
        "labels": labels,
        "adj": adj,
        "masks": masks,
    }
    return features, labels, adj, masks, fraud_rate


class GCNLayer(nn.Module):
    # 图卷积层
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.mm(adj, x)
        return F.relu(self.linear(x))


class GATLayer(nn.Module):
    # 图注意力层
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        """仅在邻接边上计算注意力，避免 O(N^2) 显存。"""
        n = x.size(0)
        h = self.W(self.dropout(x)).view(n, self.num_heads, self.head_dim)
        ei, ej = adj.nonzero(as_tuple=True)

        h_src = h[ei]
        h_dst = h[ej]
        e = F.leaky_relu(
            (h_src * self.a_src).sum(-1) + (h_dst * self.a_dst).sum(-1), 0.2
        )

        out = torch.zeros(n, self.num_heads, self.head_dim, device=x.device)
        for head in range(self.num_heads):
            eh = e[:, head]
            max_e = torch.full((n,), float("-inf"), device=x.device)
            max_e.scatter_reduce_(0, ei, eh, reduce="amax", include_self=False)
            max_e = torch.where(torch.isinf(max_e), torch.zeros_like(max_e), max_e)
            exp_eh = torch.exp(eh - max_e[ei])
            denom = torch.zeros(n, device=x.device)
            denom.index_add_(0, ei, exp_eh)
            alpha = exp_eh / (denom[ei] + 1e-16)
            msg = h_dst[:, head, :] * alpha.unsqueeze(-1)
            out[:, head, :].index_add_(0, ei, msg)

        return F.elu(out.reshape(n, -1))


class MLPClassifier(nn.Module):
    # MLP分类器（无图结构）
    def __init__(self, in_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, adj=None):
        return self.net(x)


class GCNClassifier(nn.Module):
    # GCN模型
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        dim_in = in_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(dim_in, hidden_dim, dropout))
            dim_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return self.fc(x)


class GATClassifier(nn.Module):
    # GAT模型
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        dim_in = in_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.append(GATLayer(dim_in, out_dim, num_heads=num_heads, dropout=dropout))
            dim_in = out_dim
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return self.fc(x)


def build_model(config, in_dim):
    # 构造指定类型的模型
    if config.model_type == "mlp":
        return MLPClassifier(in_dim, config.hidden_dim)
    if config.model_type == "gcn":
        return GCNClassifier(in_dim, config.hidden_dim, config.num_layers, dropout=config.dropout)
    if config.model_type == "gat":
        return GATClassifier(
            in_dim, config.hidden_dim, config.num_layers, config.num_heads, dropout=config.dropout
        )
    raise ValueError(config.model_type)


def class_weights(labels, mask):
    # 按类别自动平衡权重
    y = labels[mask].cpu().numpy()
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    w = torch.FloatTensor([1.0, n_neg / n_pos])
    return w


@torch.no_grad()
def evaluate(model, features, labels, adj, mask, device):
    # 评估模型输出各类指标
    model.eval()
    logits = model(features, adj)
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_true = labels[mask].cpu().numpy()
    y_pred = logits[mask].argmax(dim=1).cpu().numpy()
    probs_m = probs[mask.cpu().numpy()]

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, probs_m)
    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ap": float(average_precision_score(y_true, probs_m)),
    }


def train_model(config: TrainConfig, device):
    # 单模型完整训练+保存最优
    set_seed(config.seed)
    features, labels, adj, masks, fraud_rate = prepare_graph_data(config)
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    for k in masks:
        masks[k] = masks[k].to(device)

    model = build_model(config, features.size(1)).to(device)
    weight = class_weights(labels, masks["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_auc, best_state, stale = 0.0, None, 0
    history = {"train_loss": [], "val_auc": [], "val_f1": []}
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        # 训练一步
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = criterion(logits[masks["train"]], labels[masks["train"]])
        loss.backward()
        optimizer.step()

        # 验证集评估
        val_metrics = evaluate(model, features, labels, adj, masks["val"], device)
        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        # 若验证AUC提升则存最优权重，否则计数提前停止
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        # 每10轮打印
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_f1={val_metrics['f1']:.4f}"
            )
        if stale >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 回载最优
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, features, labels, adj, masks["test"], device)
    elapsed = time.time() - start

    # 清理资源
    del model, optimizer, criterion, features, labels, adj, masks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 返回训练信息
    return {
        "config": asdict(config),
        "history": history,
        "best_val_auc": best_auc,
        "test_metrics": test_metrics,
        "fraud_rate": fraud_rate,
        "elapsed_sec": elapsed,
    }


def experiment_suite():
    # 定义一系列实验配置
    base = TrainConfig()
    return [
        ("baseline_mlp", TrainConfig(model_type="mlp")),
        ("gcn_2layer", TrainConfig(model_type="gcn", num_layers=2)),
        ("gat_2layer", TrainConfig(model_type="gat", num_layers=2, num_heads=4)),
        ("gcn_1layer", TrainConfig(model_type="gcn", num_layers=1)),
        ("gcn_3layer", TrainConfig(model_type="gcn", num_layers=3)),
        ("gcn_4layer", TrainConfig(model_type="gcn", num_layers=4)),
        ("gcn_hidden32", TrainConfig(model_type="gcn", hidden_dim=32)),
        ("gcn_hidden128", TrainConfig(model_type="gcn", hidden_dim=128)),
        ("gcn_k4", TrainConfig(model_type="gcn", k_neighbors=4)),
        ("gcn_k16", TrainConfig(model_type="gcn", k_neighbors=16)),
        ("gat_1head", TrainConfig(model_type="gat", num_heads=1)),
        ("gat_8head", TrainConfig(model_type="gat", num_heads=8, hidden_dim=64)),
    ]


def run_all_experiments(device):
    # 全部实验批量运行
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}
    print(f"Device: {device}, experiments={len(experiment_suite())}\n")

    for name, cfg in experiment_suite():
        print(f"===== {name} =====")
        results[name] = train_model(cfg, device)
        tm = results[name]["test_metrics"]
        print(
            f"Done: val_auc={results[name]['best_val_auc']:.4f}, "
            f"test_auc={tm['auc']:.4f}, test_f1={tm['f1']:.4f}\n"
        )

    # 结果保存为文件和图片
    path = os.path.join(RESULT_DIR, "experiment_results.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    plot_results(results)
    save_csv(results)
    print(f"Results saved to {path}")
    return results


def plot_results(results):
    # 绘制实验效果对比
    setup_plot_style()

    arch_names = ["baseline_mlp", "gcn_2layer", "gat_2layer"]
    arch_labels = ["MLP\n(无图)", "GCN\n2层", "GAT\n2层"]
    arch_aucs = [results[n]["test_metrics"]["auc"] * 100 for n in arch_names if n in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 不同架构对比
    ax = axes[0]
    bars = ax.bar(arch_labels, arch_aucs, color=["#64748b", "#2563eb", "#dc2626"])
    ax.set_ylabel("Test AUC (%)")
    ax.set_title("模型架构对比")
    ax.set_ylim(0, max(arch_aucs) + 10)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, arch_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.1f}%", ha="center")

    # 层数敏感性对比
    layer_keys = [k for k in results if k.startswith("gcn_") and "layer" in k]
    layer_map = {"gcn_1layer": 1, "gcn_2layer": 2, "gcn_3layer": 3, "gcn_4layer": 4}
    layer_items = sorted(
        [(layer_map[k], results[k]["test_metrics"]["auc"] * 100) for k in layer_keys if k in layer_map],
        key=lambda x: x[0],
    )
    if layer_items:
        ax = axes[1]
        xs, ys = zip(*layer_items)
        ax.plot(xs, ys, "o-", color="#2563eb", linewidth=2, markersize=8)
        ax.set_xlabel("GCN 层数")
        ax.set_ylabel("Test AUC (%)")
        ax.set_title("过平滑：层数 vs AUC")
        ax.set_xticks(list(xs))
        ax.grid(alpha=0.3)

    # 隐层维度和邻居数敏感性对比
    sens_keys = ["gcn_hidden32", "gcn_2layer", "gcn_hidden128", "gcn_k4", "gcn_k16"]
    sens_labels = ["h=32", "h=64", "h=128", "k=4", "k=16"]
    sens_aucs = [results[k]["test_metrics"]["auc"] * 100 for k in sens_keys if k in results]
    sens_labels = sens_labels[: len(sens_aucs)]
    ax = axes[2]
    ax.bar(sens_labels, sens_aucs, color="#16a34a", edgecolor="white")
    ax.set_ylabel("Test AUC (%)")
    ax.set_title("参数敏感性")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_summary.png"), dpi=150)
    plt.close(fig)


def save_csv(results):
    # 保存实验主要结果为csv
    path = os.path.join(RESULT_DIR, "experiment_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "experiment", "model_type", "hidden_dim", "num_layers", "num_heads",
            "k_neighbors", "best_val_auc", "test_auc", "test_f1", "test_recall", "elapsed_sec",
        ])
        for name, res in results.items():
            c = res["config"]
            tm = res["test_metrics"]
            writer.writerow([
                name, c["model_type"], c["hidden_dim"], c["num_layers"], c["num_heads"],
                c["k_neighbors"], f"{res['best_val_auc']:.4f}", f"{tm['auc']:.4f}",
                f"{tm['f1']:.4f}", f"{tm['recall']:.4f}", f"{res['elapsed_sec']:.1f}",
            ])


def main():
    # 主程序入口：支持批量实验与单模型训练
    parser = argparse.ArgumentParser(description="GNN Fraud Detection")
    parser.add_argument("--mode", choices=["experiments", "train"], default="experiments")
    parser.add_argument("--model", default="gcn", choices=["mlp", "gcn", "gat"])
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.mode == "experiments":
        run_all_experiments(device)
    else:
        cfg = TrainConfig(model_type=args.model, epochs=args.epochs)
        res = train_model(cfg, device)
        print(json.dumps(res["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
