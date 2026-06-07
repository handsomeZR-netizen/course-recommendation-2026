from __future__ import annotations

import json
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.data import build_dataset, dataset_stats
from course_rec.io_utils import read_config
from course_rec.kg import read_tsv_edges
from course_rec.run_experiment import build_scorer
from search_fusion_weights import build_eval_cache, evaluate_matrix


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "paper_figures"
TABLE_DIR = ROOT / "reports" / "tables"

P = {
    "p1": "#D5E8F1",
    "p2": "#ABD7DF",
    "p3": "#CAEBE7",
    "p4": "#A9D9BB",
    "p5": "#90B4CF",
    "p6": "#337BAC",
    "p7": "#4FB1B2",
    "ink": "#1F3349",
    "muted": "#6C7A86",
    "grid": "#DDE8EB",
    "warm": "#D89273",
    "rose": "#C98298",
    "gold": "#D2A449",
    "green": "#6FAE75",
    "bg": "#F7FAFB",
}


MODEL_LABELS = {
    "popularity": "Pop",
    "kg_profile": "KG profile",
    "itemknn_cosine_k100_s100": "ItemKNN",
    "rp3beta_a07_b04_k100": "RP3-old",
    "rp3beta_a09_b02_k300": "RP3",
    "ease_reg_500": "EASE",
    "fusion_ease_kg_pop": "EASE+KG",
    "fusion_extended": "Stable fusion",
    "fusion_search_best": "Fusion v1",
    "fusion_search_best_reg100": "Fusion v2",
    "fusion_best_rp3": "RKF-M",
    "fusion_validated": "RKF-M",
    "adaptive_fusion_validated": "RAKF-M",
    "adaptive_kg_gate_validated": "RAKF-M-G",
    "mf_bpr": "MF-BPR",
    "lightgcn": "LightGCN",
}

FINAL_KEY = "adaptive_kg_gate_validated"
FINAL_LABEL = "RAKF-M-G"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.4,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(P["ink"])
    ax.spines["bottom"].set_color(P["ink"])
    ax.tick_params(axis="both", width=0.8, length=3.0, colors=P["ink"], pad=2.0)
    ax.grid(axis=grid_axis, color=P["grid"], linewidth=0.7, alpha=0.78, zorder=0)


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=11.5, color=P["ink"])


def add_soft_band(ax, values: pd.Series | np.ndarray, pad: float = 0.012) -> None:
    vals = np.asarray(values, dtype=float)
    ax.axvspan(vals.min() - pad, vals.max() + pad, color=P["p1"], alpha=0.18, zorder=0)


def save_bundle(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG_DIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    make_tables(summary)
    main_result_figure(summary)
    evidence_ablation_figure(summary)
    formal_ablation_figure()
    adaptive_gate_figure()
    data_kg_figure()


def load_summary() -> pd.DataFrame:
    summary_path = ROOT / "results" / "summary_with_gpu.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()
    local_path = ROOT / "results" / "summary.csv"
    if local_path.exists():
        local = pd.read_csv(local_path)
        if len(summary):
            summary = pd.concat([summary, local], ignore_index=True)
            summary = summary.drop_duplicates(subset=["run_name"], keep="last")
        else:
            summary = local
    return summary.reset_index(drop=True)


def preferred_key(summary: pd.DataFrame, preferred: str, fallback: str) -> str:
    names = set(summary["run_name"])
    return preferred if preferred in names else fallback


def make_tables(summary: pd.DataFrame) -> None:
    fixed_key = preferred_key(summary, "fusion_validated", "fusion_best_rp3")
    final_key = FINAL_KEY
    main_order = [
        "popularity",
        "kg_profile",
        "mf_bpr",
        "lightgcn",
        "itemknn_cosine_k100_s100",
        "ease_reg_500",
        "rp3beta_a09_b02_k300",
        fixed_key,
        final_key,
    ]
    table = summary.set_index("run_name").loc[
        [name for name in main_order if name in set(summary["run_name"])]
    ].reset_index()
    table.insert(0, "Model", table["run_name"].map(MODEL_LABELS).fillna(table["run_name"]))
    cols = ["Model", "HR@10", "HR@20", "NDCG@10", "NDCG@20"]
    table[cols].to_csv(TABLE_DIR / "main_results_table.csv", index=False, encoding="utf-8-sig")

    data = build_dataset(ROOT / "课程作业数据集")
    stats = dataset_stats(data)
    pd.DataFrame([stats]).to_csv(TABLE_DIR / "dataset_stats.csv", index=False, encoding="utf-8-sig")

    rel_dir = ROOT / "课程作业数据集" / "MOOCCube (1)" / "MOOCCube" / "relations"
    kg_rows = []
    for path in sorted(rel_dir.glob("*.json")):
        if path.name in {"user-course.json", "user-video.json", "concept-paper.json"}:
            continue
        kg_rows.append({"relation": path.stem, "edges": len(read_tsv_edges(path))})
    pd.DataFrame(kg_rows).to_csv(TABLE_DIR / "kg_stats.csv", index=False, encoding="utf-8-sig")
    with open(TABLE_DIR / "best_metrics.json", "w", encoding="utf-8") as f:
        best = summary.set_index("run_name").loc[final_key].to_dict()
        best["run_name"] = final_key
        json.dump(best, f, ensure_ascii=False, indent=2)
    segment_model_comparison_table(data)


def score_all_users(scorer, eval_users: np.ndarray, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start : start + batch_size]
        scores.append(np.asarray(scorer.score(batch), dtype=np.float32))
    return np.vstack(scores)


def segment_model_comparison_table(data) -> None:
    data_dir = ROOT / "课程作业数据集"
    eval_users = data.test_user_ids
    eval_cache = build_eval_cache(data, eval_users)
    counts = data.train_matrix[eval_users].getnnz(axis=1)
    segments = [
        ("0 次历史", counts == 0),
        ("1 次历史", counts == 1),
        ("2 次历史", counts == 2),
        ("3-4 次历史", (counts >= 3) & (counts <= 4)),
        ("5 次及以上", counts >= 5),
    ]
    fixed_config = ROOT / "configs" / "fusion_validated.yaml"
    if not fixed_config.exists():
        fixed_config = ROOT / "configs" / "fusion_best_rp3.yaml"
    adaptive_config = ROOT / "configs" / "adaptive_kg_gate_validated.yaml"
    models = [
        ("EASE", ROOT / "configs" / "ease.yaml"),
        ("RP3", ROOT / "configs" / "rp3beta_best.yaml"),
        ("RKF-M", fixed_config),
        (FINAL_LABEL, adaptive_config),
    ]
    rows = []
    for model_label, config_path in models:
        config = read_config(config_path)
        scorer = build_scorer(config, data, data_dir)
        batch_size = int(config.get("batch_size", 2048))
        scores = score_all_users(scorer, eval_users, batch_size)
        for group_label, mask in segments:
            sub_cache = {key: value[mask] for key, value in eval_cache.items()}
            metrics = evaluate_matrix(data, eval_users[mask], scores[mask], sub_cache)
            rows.append(
                {
                    "user_group": group_label,
                    "model": model_label,
                    "users": int(mask.sum()),
                    "HR@20": metrics["HR@20"],
                    "NDCG@20": metrics["NDCG@20"],
                }
            )
        del scorer, scores
        gc.collect()
    long_df = pd.DataFrame(rows)
    long_df.to_csv(TABLE_DIR / "segment_model_comparison_long.csv", index=False, encoding="utf-8-sig")
    pivot = long_df.pivot(index="user_group", columns="model", values="NDCG@20").reset_index()
    pivot.insert(1, "users", [int(mask.sum()) for _, mask in segments])
    pivot = pivot[["user_group", "users", "EASE", "RP3", "RKF-M", FINAL_LABEL]]
    pivot.to_csv(TABLE_DIR / "segment_model_comparison.csv", index=False, encoding="utf-8-sig")


def main_result_figure(summary: pd.DataFrame) -> None:
    fixed_key = preferred_key(summary, "fusion_validated", "fusion_best_rp3")
    final_key = FINAL_KEY
    order = [
        "popularity",
        "kg_profile",
        "mf_bpr",
        "lightgcn",
        "itemknn_cosine_k100_s100",
        "ease_reg_500",
        "rp3beta_a09_b02_k300",
        fixed_key,
        final_key,
    ]
    df = summary.set_index("run_name").loc[[x for x in order if x in set(summary["run_name"])]].reset_index()
    labels = df["run_name"].map(MODEL_LABELS).to_numpy()
    colors = {
        "popularity": P["p5"],
        "kg_profile": P["gold"],
        "mf_bpr": P["p2"],
        "lightgcn": P["p3"],
        "itemknn_cosine_k100_s100": P["p4"],
        "ease_reg_500": P["p6"],
        "rp3beta_a09_b02_k300": P["p7"],
        "fusion_best_rp3": P["warm"],
        "fusion_validated": P["warm"],
        "adaptive_fusion_validated": P["muted"],
        "adaptive_kg_gate_validated": P["ink"],
    }
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ax, metric, label in [(axes[0], "HR@20", "A"), (axes[1], "NDCG@20", "B")]:
        plot_df = df.sort_values(metric, ascending=True).reset_index(drop=True)
        y = np.arange(len(plot_df))
        bar_colors = [colors.get(name, P["p2"]) for name in plot_df["run_name"]]
        ax.barh(y, plot_df[metric], color=bar_colors, edgecolor="black", linewidth=0.4, height=0.66, zorder=3)
        ax.scatter(plot_df[metric], y, s=24, color="white", edgecolor=P["ink"], linewidth=0.6, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["run_name"].map(MODEL_LABELS))
        ax.set_xlabel(metric)
        ax.set_xlim(max(0.0, plot_df[metric].min() - 0.055), min(0.74, plot_df[metric].max() + 0.026))
        ref = float(df.loc[df["run_name"] == "ease_reg_500", metric].iloc[0])
        ax.axvline(ref, color=P["muted"], linestyle=(0, (4, 2)), linewidth=1.0, zorder=2)
        ax.text(ref + 0.002, len(plot_df) - 0.35, "EASE", fontsize=8.2, color=P["muted"], va="top")
        for yi, value in zip(y, plot_df[metric]):
            ax.text(value + 0.003, yi, f"{value:.3f}", va="center", ha="left", fontsize=8.2, color=P["ink"])
        clean_axis(ax, grid_axis="x")
        panel_label(ax, label)

    metric_cols = ["HR@10", "HR@20", "NDCG@10", "NDCG@20"]
    heat = df.set_index(df["run_name"].map(MODEL_LABELS))[metric_cols]
    norm = (heat - heat.min()) / (heat.max() - heat.min())
    cmap = mcolors.LinearSegmentedColormap.from_list("paper_heat", [P["p1"], P["p7"], P["ink"]])
    im = axes[2].imshow(norm.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    axes[2].set_xticks(np.arange(len(metric_cols)))
    axes[2].set_xticklabels(metric_cols, rotation=25, ha="right")
    axes[2].set_yticks(np.arange(len(heat.index)))
    axes[2].set_yticklabels(heat.index)
    axes[2].tick_params(length=0)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[2].text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7.6, color="white" if norm.iloc[i, j] > 0.62 else P["ink"])
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    panel_label(axes[2], "C", x=-0.10)

    base = df.set_index("run_name").loc["ease_reg_500", metric_cols]
    final = df.set_index("run_name").loc[final_key, metric_cols]
    gains = final - base
    x = np.arange(len(metric_cols))
    axes[3].bar(x, gains, color=[P["p6"], P["p7"], P["warm"], P["ink"]], edgecolor="black", linewidth=0.45, zorder=3)
    axes[3].axhline(0, color=P["muted"], linewidth=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(metric_cols, rotation=25, ha="right")
    axes[3].set_ylabel(r"$\Delta$ vs. EASE")
    axes[3].set_ylim(0, max(gains.max() + 0.002, 0.01))
    for xi, value in zip(x, gains):
        axes[3].text(xi, value + 0.00025, f"+{value:.4f}", ha="center", va="bottom", fontsize=8.0, color=P["ink"])
    clean_axis(axes[3])
    panel_label(axes[3], "D")
    save_bundle(fig, "main_results")


def evidence_ablation_figure(summary: pd.DataFrame) -> None:
    df = summary.set_index("run_name")
    fixed_key = preferred_key(summary, "fusion_validated", "fusion_best_rp3")
    final_key = FINAL_KEY
    chain = [
        ("popularity", "Popularity"),
        ("ease_reg_500", "EASE\nlocal"),
        ("rp3beta_a09_b02_k300", "RP3\nwalk"),
        ("fusion_search_best_reg100", "EASE+RP3\ncalibrated"),
        (fixed_key, "RKF-M\nfixed"),
        (final_key, "RAKF-M-G\nfinal"),
    ]
    ablation_order = [
        ("ease_reg_500", "EASE"),
        ("rp3beta_a09_b02_k300", "RP3"),
        ("fusion_search_best_reg100", "+ calibrated\nEASE"),
        (fixed_key, "+ fixed\nmix"),
        (final_key, "+ KG gate\ngrouping"),
    ]

    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.0, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    vals = np.array([float(df.loc[k, "NDCG@20"]) for k, _ in chain])
    x = np.arange(len(chain))
    axes[0].plot(x, vals, color=P["p6"], marker="o", markersize=4.6, linewidth=1.55, zorder=4)
    axes[0].fill_between(x, vals.min() - 0.012, vals, color=P["p1"], alpha=0.62, zorder=2)
    axes[0].vlines(x, vals.min() - 0.012, vals, color="white", linewidth=0.55, zorder=3)
    for i, value in enumerate(vals):
        axes[0].text(i, value + 0.004, f"{value:.4f}", ha="center", va="bottom", fontsize=8.0, color=P["ink"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in chain], fontsize=7.9)
    axes[0].set_ylabel("NDCG@20")
    axes[0].set_ylim(vals.min() - 0.035, vals.max() + 0.028)
    clean_axis(axes[0])
    panel_label(axes[0], "A", x=-0.11)

    base_ndcg = float(df.loc["ease_reg_500", "NDCG@20"])
    base_hr = float(df.loc["ease_reg_500", "HR@20"])
    plot_specs = [
        (axes[1], [float(df.loc[key, "NDCG@20"] - base_ndcg) for key, _ in ablation_order], r"$\Delta$ NDCG@20"),
        (axes[2], [float(df.loc[key, "HR@20"] - base_hr) for key, _ in ablation_order], r"$\Delta$ HR@20"),
    ]
    for idx, (ax, values, ylabel) in enumerate(plot_specs, start=1):
        y = np.arange(len(ablation_order))
        bars = ax.barh(
            y,
            values,
            color=[P["p5"], P["p7"], P["p6"], P["warm"], P["ink"]],
            edgecolor="black",
            linewidth=0.45,
            height=0.66,
            zorder=3,
        )
        ax.axvline(0, color=P["muted"], linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([text for _, text in ablation_order], fontsize=8.0)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_xlim(min(values) - 0.001, max(values) + 0.003)
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + 0.00035,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.4f}",
                ha="left",
                va="center",
                fontsize=8.0,
                color=P["ink"],
            )
        clean_axis(ax, grid_axis="x")
        panel_label(ax, "B" if idx == 1 else "C", x=-0.13)

    save_bundle(fig, "evidence_ablation_summary")


def formal_ablation_figure() -> None:
    path = TABLE_DIR / "formal_ablation_table.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = [
        "RAKF-M-G full",
        "non-gated KG mix",
        "w/o RP3",
        "w/o EASE",
        "w/o adaptive grouping",
        "KG profile only",
    ]
    df = df.set_index("variant").loc[[x for x in order if x in set(df["variant"])]].reset_index()
    labels = df["variant"].to_numpy()
    fig = plt.figure(figsize=(7.2, 3.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    y = np.arange(len(df))
    colors = [P["ink"], P["p7"], P["warm"], P["rose"], P["p6"], P["gold"]][: len(df)]
    axes[0].barh(y, df["NDCG@20"], color=colors, edgecolor="black", linewidth=0.45, height=0.66, zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("NDCG@20")
    axes[0].set_xlim(max(0.0, float(df["NDCG@20"].min()) - 0.035), float(df["NDCG@20"].max()) + 0.015)
    for yi, value in zip(y, df["NDCG@20"]):
        axes[0].text(value + 0.002, yi, f"{value:.4f}", va="center", fontsize=8.0, color=P["ink"])
    clean_axis(axes[0], grid_axis="x")
    panel_label(axes[0], "A", x=-0.10)

    for ax, metric, label in [
        (axes[1], "Delta_NDCG@20", "B"),
        (axes[2], "Delta_HR@20", "C"),
    ]:
        values = df[metric].to_numpy(dtype=float)
        ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.45, height=0.66, zorder=3)
        ax.axvline(0, color=P["muted"], linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("Delta_", r"$\Delta$ "))
        pad = max(0.004, float(np.max(np.abs(values))) * 0.15)
        ax.set_xlim(float(values.min()) - pad, float(values.max()) + pad)
        for yi, value in zip(y, values):
            ha = "left" if value >= 0 else "right"
            offset = 0.00045 if value >= 0 else -0.00045
            ax.text(value + offset, yi, f"{value:+.4f}", va="center", ha=ha, fontsize=8.0, color=P["ink"])
        clean_axis(ax, grid_axis="x")
        panel_label(ax, label, x=-0.10)
    save_bundle(fig, "formal_ablation")


def adaptive_gate_figure() -> None:
    path = ROOT / "results" / "adaptive_kg_gate_validated_segments.csv"
    if not path.exists():
        path = ROOT / "results" / "adaptive_fusion_search.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["segment"] != "assembled"].copy()
    segment_labels = {
        "cold": "0",
        "one": "1",
        "two": "2",
        "three_four": "3-4",
        "five_plus": "5+",
    }
    component_cols = ["w_ease", "w_itemknn", "w_rp3", "w_pop", "w_kg"]
    component_labels = ["EASE", "ItemKNN", "RP3", "Pop", "KG"]
    weights = df[component_cols].to_numpy(dtype=float)
    vmax = max(abs(weights.min()), abs(weights.max()))
    cmap = mcolors.LinearSegmentedColormap.from_list("gate_div", [P["rose"], "#F8FAFB", P["p7"], P["ink"]])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)
    im = axes[0].imshow(weights, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(np.arange(len(component_labels)))
    axes[0].set_xticklabels(component_labels, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(df)))
    axes[0].set_yticklabels([segment_labels[x] for x in df["segment"]])
    axes[0].set_ylabel("User history length")
    axes[0].tick_params(length=0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            value = weights[i, j]
            axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.0, color="white" if abs(value) > 0.72 else P["ink"])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=7.2, width=0.6, length=2.5)
    panel_label(axes[0], "A", x=-0.13)

    y = np.arange(len(df))
    axes[1].plot(df["HR@20"], y, marker="o", color=P["p6"], linewidth=1.45, label="HR@20", zorder=3)
    axes[1].plot(df["NDCG@20"], y, marker="s", color=P["warm"], linewidth=1.45, label="NDCG@20", zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([segment_labels[x] for x in df["segment"]])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Segment metric")
    axes[1].set_xlim(0.20, 0.74)
    axes[1].legend(frameon=False, loc="lower right")
    for yi, hr, ndcg in zip(y, df["HR@20"], df["NDCG@20"]):
        axes[1].text(hr + 0.008, yi - 0.08, f"{hr:.3f}", fontsize=7.8, color=P["p6"], va="center")
        axes[1].text(ndcg + 0.008, yi + 0.10, f"{ndcg:.3f}", fontsize=7.8, color=P["warm"], va="center")
    clean_axis(axes[1], grid_axis="x")
    panel_label(axes[1], "B")
    save_bundle(fig, "adaptive_gate_analysis")


def data_kg_figure() -> None:
    data = build_dataset(ROOT / "课程作业数据集")
    user_counts = pd.Series([len(x) for x in data.train_user_items if len(x) > 0])
    item_pop = pd.Series(data.item_popularity)
    rel_dir = ROOT / "课程作业数据集" / "MOOCCube (1)" / "MOOCCube" / "relations"
    rels = ["course-concept", "course-video", "video-concept", "teacher-course", "school-course", "prerequisite-dependency"]
    rel_counts = [len(read_tsv_edges(rel_dir / f"{r}.json")) for r in rels]

    fig = plt.figure(figsize=(7.15, 4.95), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axes[0].hist(user_counts.clip(upper=20), bins=np.arange(1, 22), color=P["p6"], edgecolor="white", linewidth=0.35)
    axes[0].set_xlabel("User interactions (clipped)")
    axes[0].set_ylabel("Users")
    axes[0].set_yscale("log")
    panel_label(axes[0], "A", x=-0.12)

    sorted_pop = np.sort(item_pop.to_numpy())[::-1]
    ranks = np.arange(1, len(sorted_pop) + 1)
    axes[1].plot(ranks, sorted_pop, color=P["p7"], linewidth=1.5)
    axes[1].fill_between(ranks, sorted_pop, color=P["p1"], alpha=0.65)
    axes[1].set_xlabel("Course rank by popularity")
    axes[1].set_ylabel("Train interactions")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    panel_label(axes[1], "B", x=-0.12)

    y = np.arange(len(rels))
    axes[2].barh(y, rel_counts, color=P["p5"], edgecolor="black", linewidth=0.35, zorder=3)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(["C-concept", "C-video", "V-concept", "Teacher", "School", "Prereq"])
    axes[2].set_xlabel("KG edges")
    axes[2].set_xscale("log")
    panel_label(axes[2], "C", x=-0.12)
    seg_path = ROOT / "results" / "adaptive_kg_gate_validated_segments.csv"
    if not seg_path.exists():
        seg_path = ROOT / "results" / "adaptive_fusion_search.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        seg = seg[seg["segment"] != "assembled"].copy()
        seg_labels = ["0", "1", "2", "3-4", "5+"]
        x = np.arange(len(seg))
        axes[3].bar(x, seg["users"], color=P["p2"], edgecolor="black", linewidth=0.35, zorder=2)
        axes[3].set_yscale("log")
        axes[3].set_ylabel("Users")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(seg_labels)
        axes[3].set_xlabel("User history length")
        ax2 = axes[3].twinx()
        ax2.plot(x, seg["HR@20"], color=P["p6"], marker="o", linewidth=1.35, label="HR@20", zorder=4)
        ax2.plot(x, seg["NDCG@20"], color=P["warm"], marker="s", linewidth=1.35, label="NDCG@20", zorder=4)
        ax2.set_ylabel("Metric")
        ax2.set_ylim(0.20, 0.76)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(P["ink"])
        ax2.tick_params(axis="y", width=0.8, length=3.0, colors=P["ink"], pad=2.0)
    panel_label(axes[3], "D", x=-0.12)
    for ax in axes:
        clean_axis(ax)
    save_bundle(fig, "data_and_kg_profile")


if __name__ == "__main__":
    main()
