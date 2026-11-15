"""Plots the all-reduce benchmark and prints its markdown table.

The input is either a JSON file holding a list of results, or a log of cs336_systems/distributed_benchmark.py
(its JSON lines are picked out). Each result has world_size, size_mib, mean_ms and std_ms. Run from the repo root:

    uv run python scripts/plot_all_reduce.py logs/all_reduce_t4x6.json
    uv run python scripts/plot_all_reduce.py logs/all_reduce.log --output all_reduce.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Light surface and inks; the three series colors are checked for colorblind separation as a set, in this order.
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_results(path: Path) -> list[dict]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:  # a log: one JSON object per line, between other output
        return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def size_label(size_mib: float) -> str:
    return f"{size_mib / 1024:g} GiB" if size_mib >= 1024 else f"{size_mib:g} MiB"


def markdown_table(results: list[dict]) -> str:
    world_sizes = sorted({r["world_size"] for r in results})
    cell = {(r["size_mib"], r["world_size"]): f"{r['mean_ms']:.2f} ± {r['std_ms']:.2f}" for r in results}
    lines = ["| data size | " + " | ".join(f"{n} GPUs (ms)" for n in world_sizes) + " |", "|---|" + "---|" * len(world_sizes)]
    for size in sorted({r["size_mib"] for r in results}):
        lines.append(f"| {size_label(size)} | " + " | ".join(cell.get((size, n), "") for n in world_sizes) + " |")
    return "\n".join(lines)


def style(ax, sizes: list[float], title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel("data size per process", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: size_label(value)))
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot(results: list[dict], output: Path):
    world_sizes = sorted({r["world_size"] for r in results})
    sizes = sorted({r["size_mib"] for r in results})
    if len(world_sizes) > len(SERIES_COLORS):
        raise SystemExit(f"{len(world_sizes)} world sizes, but only {len(SERIES_COLORS)} checked series colors")
    devices = sorted({r["device"] for r in results if "device" in r})

    fig, (ax_time, ax_per_mib) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for world_size, color in zip(world_sizes, SERIES_COLORS):
        rows = sorted((r for r in results if r["world_size"] == world_size), key=lambda r: r["size_mib"])
        x = [r["size_mib"] for r in rows]
        marks = dict(color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                     solid_capstyle="round", solid_joinstyle="round", ecolor=color, elinewidth=1, label=f"{world_size} GPUs")
        ax_time.errorbar(x, [r["mean_ms"] for r in rows], yerr=[r["std_ms"] for r in rows], **marks)
        per_mib = [r["mean_ms"] / r["size_mib"] for r in rows]
        ax_per_mib.errorbar(x, per_mib, yerr=[r["std_ms"] / r["size_mib"] for r in rows], **marks)
        # Name each line at its left end, where the lines are furthest apart.
        ax_per_mib.annotate(f"{world_size} GPUs", (x[0], per_mib[0]), xytext=(9, 3), textcoords="offset points",
                            color=INK_SECONDARY, fontsize=9, va="bottom")

    style(ax_time, sizes, "Time of one all-reduce", "time (ms)")
    ax_time.set_yscale("log")
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_time.yaxis.set_minor_locator(NullLocator())
    ax_time.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")

    style(ax_per_mib, sizes, "Time per MiB of data", "time per MiB (ms)")
    ax_per_mib.set_ylim(bottom=0)

    title = "All-reduce time by data size and number of GPUs"
    fig.suptitle(title + (f"  ({', '.join(devices)})" if devices else ""), x=0.02, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="JSON file with a list of results, or a benchmark log with JSON lines")
    parser.add_argument("--output", type=Path, help="where to save the plot (default: next to the input, as .png)")
    args = parser.parse_args()

    results = load_results(args.results)
    print(markdown_table(results))
    plot(results, args.output or args.results.with_suffix(".png"))
