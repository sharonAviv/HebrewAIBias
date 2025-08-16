#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots from topic_wd_compare.json:
  • Per-topic charts (6 variants)
  • Global summary charts (6 variants)
  • Multi-page grid comparisons (6 variants)

Styling:
  • Colors = MODEL (consistent across figs)
  • Language via hatch: English=solid, Hebrew=///

Robust layout:
  • Grid pages use a dedicated bottom legend row (no overlap).
  • Default grid layout is a single column of panels (--grid-layout column).
  • Progress bar + elapsed time.

Usage (defaults):
  python plot_topic_wd_compare.py

Options:
  --outdir wd_charts --formats png,pdf --dpi 250
  --grid-layout column|row|grid  (default: column)
  --grid-rows 7                  (only for column layout)
  --palette "modelA:#e41a1c,modelB:#377eb8"
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make LaTeX-ready charts from topic_wd_compare.json")
    p.add_argument("--in", dest="infile", default="topic_wd_compare.json",
                   help="Path to topic_wd_compare.json (default: topic_wd_compare.json)")
    p.add_argument("--outdir", default="wd_charts",
                   help="Folder for all outputs (default: wd_charts)")
    p.add_argument("--formats", default="png,pdf",
                   help="Comma-separated: png,pdf,svg (default: png,pdf)")
    p.add_argument("--dpi", type=int, default=250, help="PNG/SVG DPI (default: 250)")
    p.add_argument("--grid-cols", type=int, default=3, help="Columns for grid layout when --grid-layout=grid (default: 3)")
    p.add_argument("--grid-layout", choices=["column","row","grid"], default="column",
                   help="Layout for general comparison pages (default: column)")
    p.add_argument("--grid-rows", type=int, default=7,
                   help="Panels per page for --grid-layout=column (default: 7)")
    p.add_argument("--palette", default="", help="CSV 'model:hex,model:hex,...' to override model colors")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    return p.parse_args()

# ---------------- IO ----------------

def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

# ---------------- Data shaping ----------------

VARIANTS = ("no_refusal", "with_refusal")
LANG_SCOPES = ("all", "hebrew", "english")

def records_from_topic(topic: str, topic_blob: dict) -> pd.DataFrame:
    rows = []
    bymlv = topic_blob["summary"]["by_model_language_variant"]
    for variant in VARIANTS:
        if variant not in bymlv:
            continue
        for rec in bymlv[variant]:
            rows.append({
                "topic": topic,
                "model": rec.get("model", ""),
                "language": str(rec.get("language", "")).lower().strip(),
                "variant": variant,
                "avg_wd": rec.get("avg_WD_over_topic_questions", None),
                "n": rec.get("n_questions_counted", None),
            })
    return pd.DataFrame(rows)

def records_from_global(glob: dict) -> pd.DataFrame:
    rows = []
    bymlv = glob["by_model_language_variant"]
    for variant in VARIANTS:
        if variant not in bymlv:
            continue
        for rec in bymlv[variant]:
            rows.append({
                "topic": "Summary",
                "model": rec.get("model", ""),
                "language": str(rec.get("language", "")).lower().strip(),
                "variant": variant,
                "avg_wd": rec.get("avg_of_file_level_average_WD", None),
                "n": rec.get("n_files", None),
            })
    return pd.DataFrame(rows)

def build_master_frames(js: dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    topics = js.get("topics", {})
    topic_list = sorted(list(topics.keys()))
    parts = [records_from_topic(t, topics[t]) for t in topic_list]
    topics_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["topic", "model", "language", "variant", "avg_wd", "n"]
    )
    summary_df = records_from_global(js["global_summary"])
    return topics_df, summary_df, topic_list

# ---------------- Styling ----------------

def parse_palette_override(spec: str) -> Dict[str, str]:
    out = {}
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        key, color = tok.split(":", 1)
        out[key.strip()] = color.strip()
    return out

def model_palette(models: Iterable[str], override: Dict[str,str]) -> Dict[str,str]:
    models = list(dict.fromkeys(models))  # stable order, de-duped
    tab = sns.color_palette("tab20", n_colors=max(20, len(models)))
    pal = {}
    for i, m in enumerate(models):
        pal[m] = override.get(m, to_hex(tab[i % len(tab)]))
    return pal

# ---------------- Plotting helpers ----------------

def slugify(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")

def apply_lang_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    scope = scope.lower()
    if scope == "hebrew":
        return df[df["language"] == "hebrew"].copy()
    if scope == "english":
        return df[df["language"] == "english"].copy()
    return df.copy()

def shared_xlim(df_list: List[pd.DataFrame]) -> Tuple[float,float]:
    vals = []
    for df in df_list:
        vals.extend([v for v in df["avg_wd"].tolist() if pd.notna(v)])
    if not vals:
        return (0.0, 1.0)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    return (lo - pad, hi + pad)

def draw_barh(ax, df: pd.DataFrame, model_pal: Dict[str,str], title: str, xlabel: str,
              show_legends: bool = True, compact: bool = False):
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, alpha=0.7)
        ax.set_axis_off()
        return

    df = df.sort_values("avg_wd", ascending=False).reset_index(drop=True)
    df["label"] = df["model"] + " (" + df["language"] + ")"
    y = np.arange(len(df))

    colors = [model_pal.get(m, "#888888") for m in df["model"]]
    bars = ax.barh(y, df["avg_wd"].values, color=colors, edgecolor="black", linewidth=0.4)
    for patch, lang in zip(bars.patches, df["language"]):
        patch.set_hatch("///" if str(lang).lower() == "hebrew" else "")

    fs = 8 if compact else 9
    ax.set_yticks(y, df["label"].tolist(), fontsize=fs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=fs+1)
    ax.set_title(title, fontsize=12, pad=8)

    for yi, val in zip(y, df["avg_wd"].values):
        ax.text(val, yi, f" {val:.3f}", va="center", fontsize=fs)

    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.grid(False, axis="y")

    if show_legends:
        unique_models = list(dict.fromkeys(df["model"].tolist()))
        handles_models = [Patch(facecolor=model_pal.get(m, "#888888"), edgecolor="black", label=m) for m in unique_models]
        leg1 = ax.legend(handles=handles_models, title="Model", loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True, fontsize=fs)
        ax.add_artist(leg1)

        handles_lang = [
            Patch(facecolor="#DDDDDD", edgecolor="black", hatch="", label="english"),
            Patch(facecolor="#DDDDDD", edgecolor="black", hatch="///", label="hebrew"),
        ]
        ax.legend(handles=handles_lang, title="Language", frameon=True, loc="lower right", fontsize=fs)

def save_figure(fig: plt.Figure, base: Path, formats: List[str], dpi: int):
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = base.with_suffix("." + fmt)
        fig.savefig(path, dpi=(dpi if fmt != "pdf" else None), bbox_inches="tight")

# ---------------- Progress ----------------

def _progress(enabled: bool, prefix: str, i: int, total: int, start_time: float):
    if not enabled:
        return
    bar_len = 32
    frac = 0 if total == 0 else i / total
    filled = int(bar_len * frac)
    bar = "█" * filled + "─" * (bar_len - filled)
    elapsed = time.time() - start_time
    msg = f"\r{prefix} [{bar}] {i}/{total}  Elapsed: {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    print(msg, end="", flush=True)
    if i >= total:
        print()

# ---------------- Chart builders ----------------

def per_topic_charts(topics_df: pd.DataFrame, outdir: Path, formats: List[str], dpi: int,
                     model_pal: Dict[str,str], progress: bool, start: float):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    total = len(topics_df["topic"].unique()) * len(VARIANTS) * len(LANG_SCOPES)
    done = 0
    for topic, df_t in topics_df.groupby("topic"):
        topic_slug = slugify(topic)
        for variant in VARIANTS:
            df_v = df_t[df_t["variant"] == variant]
            for scope in LANG_SCOPES:
                df_vs = apply_lang_scope(df_v, scope)
                fig, ax = plt.subplots(constrained_layout=True,
                                       figsize=(8.4, max(2.8, 0.35 * max(1, len(df_vs)))))
                title = f"{topic} — {variant.replace('_',' ').title()} — {scope.title()}"
                draw_barh(ax, df_vs, model_pal, title=title, xlabel="Average WD (topic average)")
                save_figure(fig, outdir / f"topic-{topic_slug}_variant-{variant}_scope-{scope}", formats, dpi)
                plt.close(fig)
                done += 1
                _progress(progress, "Per-topic charts", done, total, start)

def summary_charts(summary_df: pd.DataFrame, outdir: Path, formats: List[str], dpi: int,
                   model_pal: Dict[str,str], progress: bool, start: float):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    total = len(VARIANTS) * len(LANG_SCOPES)
    done = 0
    for variant in VARIANTS:
        df_v = summary_df[summary_df["variant"] == variant]
        for scope in LANG_SCOPES:
            df_vs = apply_lang_scope(df_v, scope)
            fig, ax = plt.subplots(constrained_layout=True,
                                   figsize=(8.4, max(2.8, 0.35 * max(1, len(df_vs)))))
            title = f"Summary — {variant.replace('_',' ').title()} — {scope.title()}"
            draw_barh(ax, df_vs, model_pal, title=title, xlabel="Average WD (file-level averages)")
            save_figure(fig, outdir / f"summary_variant-{variant}_scope-{scope}", formats, dpi)
            plt.close(fig)
            done += 1
            _progress(progress, "Summary charts", done, total, start)

# --- NEW robust grid pages (supports column/row/grid layouts) ---

def grid_charts(topics_df: pd.DataFrame, summary_df: pd.DataFrame, topic_list: List[str],
                outdir: Path, formats: List[str], dpi: int, model_pal: Dict[str,str],
                grid_cols: int, progress: bool, start: float, grid_layout: str, grid_rows: int):
    sns.set_theme(style="whitegrid", font_scale=0.95)

    if grid_layout == "column":
        # One column per page, N rows; legend occupies a dedicated bottom row
        rows = max(1, grid_rows)
        n_per_page = rows  # summary will replace the last slot on last page

        # total pages for progress
        pages_total = 0
        for _ in VARIANTS:
            for _ in LANG_SCOPES:
                pages_total += max(1, int(np.ceil(len(topic_list) / (n_per_page - 1))))  # -1 for summary slot
        pages_done = 0

        for variant in VARIANTS:
            for scope in LANG_SCOPES:
                grid_base = outdir / f"grid_variant-{variant}_scope-{scope}"
                pdf_path = grid_base.with_suffix(".pdf")
                pdf = PdfPages(pdf_path)

                # pagination (reserve one slot for summary on the last page)
                topics_only = topic_list[:]
                chunk = n_per_page - 1 if len(topics_only) > n_per_page - 1 else n_per_page
                pages = [topics_only[i:i+chunk] for i in range(0, len(topics_only), chunk)] or [[]]

                # Compute consistent xlimits
                dfs_for_xlim = []
                for t in topic_list:
                    df_tv = topics_df[(topics_df["topic"] == t) & (topics_df["variant"] == variant)]
                    dfs_for_xlim.append(apply_lang_scope(df_tv, scope))
                df_sv = summary_df[summary_df["variant"] == variant]
                df_svs = apply_lang_scope(df_sv, scope)
                dfs_for_xlim.append(df_svs)
                xlo, xhi = shared_xlim(dfs_for_xlim)

                for page_idx, page_topics in enumerate(pages):
                    include_summary_here = True  # always place summary as last panel
                    n_panels = len(page_topics) + (1 if include_summary_here else 0)
                    # Compute a panel height based on the largest bar count on this page
                    max_bars = 1
                    for t in page_topics:
                        df_tv = topics_df[(topics_df["topic"] == t) & (topics_df["variant"] == variant)]
                        max_bars = max(max_bars, len(apply_lang_scope(df_tv, scope)))
                    max_bars = max(max_bars, len(df_svs))
                    panel_h = max(2.6, 0.26 * max_bars)
                    legend_h_inches = 1.2
                    fig_h = n_panels * panel_h + legend_h_inches + 0.8  # +title margin
                    fig = plt.figure(figsize=(9.0, fig_h))

                    # rows for panels + 1 legend row
                    gs = gridspec.GridSpec(n_panels + 1, 1, figure=fig,
                                           height_ratios=[1.0]*n_panels + [0.22],
                                           hspace=0.45)

                    # plot axes
                    axes = []
                    for r in range(n_panels):
                        axes.append(fig.add_subplot(gs[r, 0]))

                    # draw topics
                    models_on_page: List[str] = []
                    for ax, t in zip(axes, page_topics):
                        df_tv = topics_df[(topics_df["topic"] == t) & (topics_df["variant"] == variant)]
                        df_tvs = apply_lang_scope(df_tv, scope)
                        draw_barh(ax, df_tvs, model_pal, title=t, xlabel="Average WD",
                                  show_legends=False, compact=True)
                        ax.set_xlim(xlo, xhi)
                        models_on_page.extend(df_tvs["model"].tolist())

                    # summary as last panel
                    if include_summary_here:
                        ax = axes[-1] if len(axes) == n_panels else fig.add_subplot(gs[n_panels-1, 0])
                        draw_barh(ax, df_svs, model_pal, title="Summary", xlabel="Average WD",
                                  show_legends=False, compact=True)
                        ax.set_xlim(xlo, xhi)
                        models_on_page.extend(df_svs["model"].tolist())

                    # bottom legend row
                    legend_ax = fig.add_subplot(gs[n_panels, 0])
                    legend_ax.axis("off")
                    unique_models = list(dict.fromkeys(models_on_page))
                    model_handles = [Patch(facecolor=model_pal.get(m, "#888888"), edgecolor="black", label=m)
                                     for m in unique_models]
                    # Auto columns for model legend
                    mcount = len(unique_models)
                    if mcount <= 10: ncol = 2; mfs = 9
                    elif mcount <= 20: ncol = 3; mfs = 8
                    elif mcount <= 30: ncol = 4; mfs = 7
                    else: ncol = 5; mfs = 7

                    model_leg = legend_ax.legend(handles=model_handles, title="Model",
                                                 loc="center left", bbox_to_anchor=(0.01, 0.5),
                                                 frameon=True, fontsize=mfs, ncol=ncol)
                    legend_ax.add_artist(model_leg)

                    lang_handles = [
                        Patch(facecolor="#DDDDDD", edgecolor="black", hatch="", label="english"),
                        Patch(facecolor="#DDDDDD", edgecolor="black", hatch="///", label="hebrew"),
                    ]
                    legend_ax.legend(handles=lang_handles, title="Language",
                                     loc="center right", bbox_to_anchor=(0.99, 0.5),
                                     frameon=True, fontsize=9)

                    fig.suptitle(f"General Comparison — {variant.replace('_',' ').title()} — {scope.title()}",
                                 fontsize=14, y=0.995)

                    pdf.savefig(fig, bbox_inches="tight")
                    save_figure(fig, grid_base.with_name(grid_base.name + f"_page{page_idx+1}"),
                                [f for f in formats if f != "pdf"], dpi)
                    plt.close(fig)

                    pages_done += 1
                    _progress(progress, "Grid pages", pages_done, pages_total, start)

                pdf.close()
        return  # done

    # --- Optional: legacy modes if you ever want them ---
    def _grid_or_row(mode: str):
        # mode="row": rows=1, many cols; mode="grid": 3 rows x grid_cols
        rows = 1 if mode == "row" else 3
        plot_cols = (max(1, len(topic_list))) if mode == "row" else grid_cols
        n_per_page = rows * plot_cols

        pages_total = 0
        for _ in VARIANTS:
            for _ in LANG_SCOPES:
                pages_total += max(1, int(np.ceil(len(topic_list) / n_per_page)))
        pages_done = 0

        for variant in VARIANTS:
            for scope in LANG_SCOPES:
                grid_base = outdir / f"grid_variant-{variant}_scope-{scope}"
                pdf_path = grid_base.with_suffix(".pdf")
                pdf = PdfPages(pdf_path)

                all_topics = topic_list[:]
                pages = [all_topics[i:i+n_per_page] for i in range(0, len(all_topics), n_per_page)] or [[]]

                dfs_for_xlim = []
                for t in topic_list:
                    df_tv = topics_df[(topics_df["topic"] == t) & (topics_df["variant"] == variant)]
                    dfs_for_xlim.append(apply_lang_scope(df_tv, scope))
                df_sv = summary_df[summary_df["variant"] == variant]
                df_svs = apply_lang_scope(df_sv, scope)
                dfs_for_xlim.append(df_svs)
                xlo, xhi = shared_xlim(dfs_for_xlim)

                for page_idx, page_topics in enumerate(pages):
                    total_slots = rows * plot_cols
                    include_summary_here = (page_idx == len(pages)-1)
                    topics_on_this_page = page_topics[:min(len(page_topics), total_slots - (1 if include_summary_here else 0))]

                    fig = plt.figure(figsize=(plot_cols*5.2 + 0.1, rows*3.4 + 1.4))
                    gs = gridspec.GridSpec(rows + 1, plot_cols, figure=fig,
                                           height_ratios=[1.0]*rows + [0.25],
                                           wspace=0.35, hspace=0.45)

                    axes = np.empty((rows, plot_cols), dtype=object)
                    for r in range(rows):
                        for c in range(plot_cols):
                            axes[r, c] = fig.add_subplot(gs[r, c])

                    legend_ax = fig.add_subplot(gs[rows, :])
                    legend_ax.axis("off")

                    idx = 0
                    models_on_page: List[str] = []
                    for t in topics_on_this_page:
                        r, c = divmod(idx, plot_cols)
                        ax = axes[r, c]
                        df_tv = topics_df[(topics_df["topic"] == t) & (topics_df["variant"] == variant)]
                        df_tvs = apply_lang_scope(df_tv, scope)
                        draw_barh(ax, df_tvs, model_pal, title=t, xlabel="Average WD",
                                  show_legends=False, compact=True)
                        ax.set_xlim(xlo, xhi)
                        models_on_page.extend(df_tvs["model"].tolist())
                        idx += 1

                    while idx < total_slots:
                        r, c = divmod(idx, plot_cols)
                        axes[r, c].axis("off")
                        idx += 1

                    if include_summary_here:
                        r, c = divmod(total_slots-1, plot_cols)
                        ax = axes[r, c]
                        draw_barh(ax, df_svs, model_pal, title="Summary", xlabel="Average WD",
                                  show_legends=False, compact=True)
                        ax.set_xlim(xlo, xhi)
                        models_on_page.extend(df_svs["model"].tolist())

                    unique_models = list(dict.fromkeys(models_on_page))
                    model_handles = [Patch(facecolor=model_pal.get(m, "#888888"), edgecolor="black", label=m)
                                     for m in unique_models]
                    mcount = len(unique_models)
                    if mcount <= 12: ncol = 2; mfs = 9
                    elif mcount <= 24: ncol = 3; mfs = 8
                    else: ncol = 4; mfs = 7
                    model_leg = legend_ax.legend(handles=model_handles, title="Model",
                                                 loc="center left", bbox_to_anchor=(0.01, 0.5),
                                                 frameon=True, fontsize=mfs, ncol=ncol)
                    legend_ax.add_artist(model_leg)

                    lang_handles = [
                        Patch(facecolor="#DDDDDD", edgecolor="black", hatch="", label="english"),
                        Patch(facecolor="#DDDDDD", edgecolor="black", hatch="///", label="hebrew"),
                    ]
                    legend_ax.legend(handles=lang_handles, title="Language",
                                     loc="center right", bbox_to_anchor=(0.99, 0.5),
                                     frameon=True, fontsize=9)

                    fig.suptitle(f"General Comparison — {variant.replace('_',' ').title()} — {scope.title()}",
                                 fontsize=14, y=0.995)

                    pdf.savefig(fig, bbox_inches="tight")
                    save_figure(fig, grid_base.with_name(grid_base.name + f"_page{page_idx+1}"),
                                [f for f in formats if f != "pdf"], dpi)
                    plt.close(fig)

                    pages_done += 1
                    _progress(progress, f"Grid pages ({mode})", pages_done, pages_total, start)

                pdf.close()

    if grid_layout == "row":
        _grid_or_row("row")
    elif grid_layout == "grid":
        _grid_or_row("grid")

# ---------------- Main ----------------

def main():
    args = parse_args()
    start = time.time()
    progress = not args.no_progress

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()] or ["png", "pdf"]

    if progress:
        print("Loading JSON...", end="", flush=True)
    js = load_json(infile)
    topics_df, summary_df, topic_list = build_master_frames(js)
    if progress:
        print(" done.")

    override = parse_palette_override(args.palette)
    models = set(topics_df["model"]).union(set(summary_df["model"]))
    model_pal = model_palette(models, override)

    per_topic_charts(topics_df, outdir, formats, args.dpi, model_pal, progress, start)
    summary_charts(summary_df, outdir, formats, args.dpi, model_pal, progress, start)
    grid_charts(
        topics_df, summary_df, topic_list,
        outdir, formats, args.dpi, model_pal,
        args.grid_cols, progress, start,
        grid_layout=args.grid_layout, grid_rows=args.grid_rows
    )

    elapsed = time.time() - start
    print(f"✓ Charts saved to: {outdir.resolve()}  (Elapsed: {int(elapsed)//60:02d}:{int(elapsed)%60:02d})")

if __name__ == "__main__":
    main()
