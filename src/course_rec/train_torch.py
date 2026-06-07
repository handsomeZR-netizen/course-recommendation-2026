from __future__ import annotations

import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np

from .data import build_dataset, dataset_stats
from .io_utils import copy_config, dump_json, ensure_dir, read_config, tee_stdout
from .metrics import evaluate_full_ranking
from .torch_models import BPRMF, LightGCN, require_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    torch = require_torch()
    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = config.get("run_name") or f"{config['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(args.output_root) / run_name)
    copy_config(args.config, out_dir)

    with tee_stdout(out_dir / "run.log"):
        print(f"run_name={run_name}")
        print(f"python={platform.python_version()} platform={platform.platform()}")
        print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")

        data = build_dataset(args.data_dir)
        stats = dataset_stats(data)
        print("data_stats=", stats)
        dump_json(stats, out_dir / "data_stats.json")

        model = build_model(config, data, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 0.001)))

        epochs = int(config.get("epochs", 80))
        batch_size = int(config.get("batch_size", 4096))
        neg_per_pos = int(config.get("negatives_per_positive", 1))
        weight_decay = float(config.get("weight_decay", 1.0e-5))
        eval_every = int(config.get("eval_every", 5))
        ks = tuple(int(k) for k in config.get("ks", [10, 20]))
        positive_sets = [set(items.tolist()) for items in data.train_user_items]
        best_metric = -1.0
        best_metrics = None

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(
                torch,
                model,
                optimizer,
                data,
                positive_sets,
                batch_size,
                neg_per_pos,
                weight_decay,
                device,
            )
            print(f"epoch={epoch} loss={loss:.6f}")
            if epoch % eval_every == 0 or epoch == epochs:
                metrics = evaluate_full_ranking(
                    data=data,
                    score_fn=model.score_all_items,
                    ks=ks,
                    batch_size=int(config.get("eval_batch_size", 4096)),
                    sample_topk_users=int(config.get("sample_topk_users", 20)),
                    sample_path=out_dir / "topk_sample.csv",
                )
                print("eval=", metrics)
                score = float(metrics.get("NDCG@20", 0.0))
                if score > best_metric:
                    best_metric = score
                    best_metrics = metrics
                    dump_json(best_metrics, out_dir / "metrics.json")
                    torch.save(model.model.state_dict(), out_dir / "best_model.pt")

        if best_metrics is not None:
            print("best_metrics=")
            for key, value in best_metrics.items():
                print(f"  {key}: {value}")


def build_model(config: dict, data, device):
    model_name = config["model"]
    dim = int(config.get("embedding_dim", 64))
    if model_name == "mf_bpr":
        return BPRMF(data.n_users, data.n_items, dim, device)
    if model_name == "lightgcn":
        return LightGCN(data, dim, int(config.get("layers", 3)), device)
    raise ValueError(f"Unknown torch model: {model_name}")


def train_one_epoch(
    torch, model, optimizer, data, positive_sets, batch_size, neg_per_pos, weight_decay, device
):
    pairs = data.train_pairs
    order = np.random.permutation(len(pairs))
    total_loss = 0.0
    total_batches = 0
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        pos = pairs[batch_idx]
        users = np.repeat(pos[:, 0], neg_per_pos)
        pos_items = np.repeat(pos[:, 1], neg_per_pos)
        neg_items = sample_negatives(data.n_items, positive_sets, users)
        users_t = torch.as_tensor(users, dtype=torch.long, device=device)
        pos_t = torch.as_tensor(pos_items, dtype=torch.long, device=device)
        neg_t = torch.as_tensor(neg_items, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(users_t, pos_t, neg_t, weight_decay)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(1, total_batches)


def sample_negatives(n_items: int, positive_sets: list[set[int]], users: np.ndarray) -> np.ndarray:
    neg = np.random.randint(0, n_items, size=len(users), dtype=np.int64)
    for idx, user in enumerate(users):
        positives = positive_sets[int(user)]
        while int(neg[idx]) in positives:
            neg[idx] = np.random.randint(0, n_items)
    return neg


if __name__ == "__main__":
    main()
