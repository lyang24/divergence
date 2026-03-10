#!/usr/bin/env python3
"""
Plot DiskANN vs Divergence comparison from a narrative TSV file like:
  results/diskann_vs_divergence_2026-03-08.tsv

This script has ZERO third-party plotting deps (no matplotlib).
It extracts the two embedded tables (DiskANN sweep + Divergence points),
writes machine-readable TSVs, and renders a few simple SVG plots.

Outputs (under --out dir):
  - diskann_sweep.tsv
  - divergence_points.tsv
  - recall_vs_io.svg
  - qps_vs_recall.svg
  - latency_vs_recall.svg

Usage:
  python3 scripts/plot_diskann_vs_divergence.py \
    --in results/diskann_vs_divergence_2026-03-08.tsv
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class DiskannRow:
    L: int
    recall_pct: float
    qps: float
    mean_lat_us: float
    p95_us: float
    p999_us: float
    ios_per_q: float


@dataclasses.dataclass(frozen=True)
class DivergenceRow:
    config: str
    recall_pct: float
    qps: float
    p50_ms: float
    p99_ms: float
    phy_per_q: float


@dataclasses.dataclass(frozen=True)
class DivergenceSaqRow:
    config: str
    recall_pct: float
    qps: float
    p50_ms: float
    p99_ms: float
    adj_per_q: float
    ref_per_q: float
    io_per_q: float
    ref_ms: float


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_diskann_table(lines: list[str]) -> list[DiskannRow]:
    # Find header line like:
    #    L   recall@100    QPS   mean_lat_us   p95_us   p999_us   IOs/q ...
    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*L\s+recall@(\d+)\s+QPS\s+mean_lat_us\s+p95_us\s+p999_us\s+IOs/q\b", line):
            header_idx = i
            break
    if header_idx is None:
        return []

    out: list[DiskannRow] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("NOTE:") or line.startswith("==="):
            break
        if not re.match(r"^\s*\d+", line):
            continue

        parts = line.split()
        # Expected minimum:
        # L recall QPS mean_us p95_us p999_us ios/q io_us cpu_us
        if len(parts) < 7:
            continue
        try:
            L = int(parts[0])
            recall_pct = float(parts[1])
            qps = float(parts[2])
            mean_lat_us = float(parts[3])
            p95_us = float(parts[4])
            p999_us = float(parts[5])
            ios_per_q = float(parts[6])
        except ValueError:
            continue

        out.append(
            DiskannRow(
                L=L,
                recall_pct=recall_pct,
                qps=qps,
                mean_lat_us=mean_lat_us,
                p95_us=p95_us,
                p999_us=p999_us,
                ios_per_q=ios_per_q,
            )
        )
    return out


def _parse_divergence_table(lines: list[str]) -> list[DivergenceRow]:
    # Accept both headers:
    #   config recall@100 QPS p50_ms p99_ms phy/q ...
    #   config ef recall@100 QPS p50_ms p99_ms phy/q ...
    header_idx = None
    has_ef = False
    for i, line in enumerate(lines):
        if re.match(r"^\s*config\s+recall@(\d+)\s+QPS\s+p50_ms\s+p99_ms\s+phy/q\b", line):
            header_idx = i
            has_ef = False
            break
        if re.match(r"^\s*config\s+ef\s+recall@(\d+)\s+QPS\s+p50_ms\s+p99_ms\s+phy/q\b", line):
            header_idx = i
            has_ef = True
            break
    if header_idx is None:
        return []

    out: list[DivergenceRow] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("NOTE:") or line.startswith("==="):
            break
        parts = line.split()
        need = 7 if has_ef else 6
        if len(parts) < need:
            continue
        try:
            config = parts[0]
            idx = 1
            if has_ef:
                idx += 1  # skip ef
            recall_pct = float(parts[idx + 0])
            qps = float(parts[idx + 1])
            p50_ms = float(parts[idx + 2])
            p99_ms = float(parts[idx + 3])
            phy_per_q = float(parts[idx + 4])
        except ValueError:
            continue

        out.append(
            DivergenceRow(
                config=config,
                recall_pct=recall_pct,
                qps=qps,
                p50_ms=p50_ms,
                p99_ms=p99_ms,
                phy_per_q=phy_per_q,
            )
        )
    return out


def _parse_divergence_saq_table(lines: list[str]) -> list[DivergenceSaqRow]:
    # Header like:
    # config ef R recall@100 QPS p50_ms p99_ms adj/q ref/q io/q ref_ms ...
    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*config\s+ef\s+R\s+recall@(\d+)\s+QPS\s+p50_ms\s+p99_ms\s+adj/q\s+ref/q\s+io/q\b", line):
            header_idx = i
            break
    if header_idx is None:
        return []

    out: list[DivergenceSaqRow] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("NOTE:") or line.startswith("===") or line.lstrip().startswith("SAQ-only"):
            break
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            config = parts[0]
            recall_pct = float(parts[3])
            qps = float(parts[4])
            p50_ms = float(parts[5])
            p99_ms = float(parts[6])
            adj_per_q = float(parts[7])
            ref_per_q = float(parts[8])
            io_per_q = float(parts[9])
            ref_ms = float(parts[10])
        except ValueError:
            continue

        out.append(
            DivergenceSaqRow(
                config=config,
                recall_pct=recall_pct,
                qps=qps,
                p50_ms=p50_ms,
                p99_ms=p99_ms,
                adj_per_q=adj_per_q,
                ref_per_q=ref_per_q,
                io_per_q=io_per_q,
                ref_ms=ref_ms,
            )
        )
    return out


def _write_tsv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _fmt(n: float) -> str:
    if abs(n) >= 100:
        return f"{n:.0f}"
    if abs(n) >= 10:
        return f"{n:.1f}"
    return f"{n:.2f}"


def _svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_plot(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
    out_path: Path,
    width: int = 1000,
    height: int = 650,
    pad_l: int = 85,
    pad_r: int = 35,
    pad_t: int = 55,
    pad_b: int = 70,
) -> None:
    # series item:
    # { "name": str, "xs": [float], "ys": [float], "color": str, "kind": "line|scatter", "marker": "circle|square" }
    all_x = [x for s in series for x in s["xs"]]
    all_y = [y for s in series for y in s["ys"]]
    if not all_x or not all_y:
        return

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    # Expand bounds a bit for readability
    def expand(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
        if lo == hi:
            d = 1.0 if lo == 0 else abs(lo) * 0.1
            return lo - d, hi + d
        d = (hi - lo) * frac
        return lo - d, hi + d

    xmin, xmax = expand(xmin, xmax)
    ymin, ymax = expand(ymin, ymax)

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x2px(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * plot_w

    def y2px(y: float) -> float:
        return pad_t + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h

    # ticks: 6 ticks each axis
    def ticks(lo: float, hi: float, nt: int = 6) -> list[float]:
        if lo == hi:
            return [lo]
        step = (hi - lo) / (nt - 1)
        return [lo + i * step for i in range(nt)]

    xt = ticks(xmin, xmax, 6)
    yt = ticks(ymin, ymax, 6)

    # Build SVG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ')
        f.write(f'viewBox="0 0 {width} {height}">\n')
        f.write('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>\n')

        # Title
        f.write(
            f'<text x="{pad_l}" y="32" font-family="ui-sans-serif, system-ui" font-size="20" fill="#111">'
            f"{_svg_escape(title)}</text>\n"
        )

        # Axes
        x0, y0 = pad_l, pad_t + plot_h
        x1, y1 = pad_l + plot_w, pad_t
        f.write(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1"/>\n')
        f.write(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1"/>\n')

        # Grid + ticks labels
        for v in xt:
            px = x2px(v)
            f.write(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y1}" stroke="#eee"/>\n')
            f.write(
                f'<text x="{px:.2f}" y="{y0+22}" text-anchor="middle" '
                f'font-family="ui-sans-serif, system-ui" font-size="12" fill="#333">{_svg_escape(_fmt(v))}</text>\n'
            )
        for v in yt:
            py = y2px(v)
            f.write(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#eee"/>\n')
            f.write(
                f'<text x="{x0-8}" y="{py+4:.2f}" text-anchor="end" '
                f'font-family="ui-sans-serif, system-ui" font-size="12" fill="#333">{_svg_escape(_fmt(v))}</text>\n'
            )

        # Axis labels
        f.write(
            f'<text x="{pad_l + plot_w/2:.2f}" y="{height-22}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui" font-size="14" fill="#111">{_svg_escape(x_label)}</text>\n'
        )
        # y label rotated
        f.write(
            f'<text x="22" y="{pad_t + plot_h/2:.2f}" text-anchor="middle" transform="rotate(-90 22 {pad_t + plot_h/2:.2f})" '
            f'font-family="ui-sans-serif, system-ui" font-size="14" fill="#111">{_svg_escape(y_label)}</text>\n'
        )

        # Draw series
        for s in series:
            xs = s["xs"]
            ys = s["ys"]
            color = s.get("color", "#0b62ff")
            kind = s.get("kind", "line")
            marker = s.get("marker", "circle")

            pts = [(x2px(x), y2px(y)) for x, y in zip(xs, ys)]
            if kind == "line" and len(pts) >= 2:
                d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
                f.write(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.0"/>\n')
            # scatter markers
            for x, y in pts:
                if marker == "square":
                    f.write(f'<rect x="{x-4:.2f}" y="{y-4:.2f}" width="8" height="8" fill="{color}" opacity="0.95"/>\n')
                else:
                    f.write(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" opacity="0.95"/>\n')

        # Legend
        lx = pad_l + 10
        ly = pad_t + 10
        lh = 18
        for i, s in enumerate(series):
            name = s.get("name", f"s{i}")
            color = s.get("color", "#0b62ff")
            marker = s.get("marker", "circle")
            y = ly + i * lh
            if marker == "square":
                f.write(f'<rect x="{lx}" y="{y-9}" width="10" height="10" fill="{color}"/>\n')
            else:
                f.write(f'<circle cx="{lx+5}" cy="{y-4}" r="5" fill="{color}"/>\n')
            f.write(
                f'<text x="{lx+16}" y="{y}" font-family="ui-sans-serif, system-ui" font-size="13" fill="#111">'
                f"{_svg_escape(name)}</text>\n"
            )

        f.write("</svg>\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input narrative TSV (results/diskann_vs_divergence_*.tsv)")
    ap.add_argument(
        "--out",
        dest="out",
        default="",
        help="Output directory (default: results/plots/<input_stem>/)",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    out_dir = Path(args.out) if args.out else Path("results/plots") / inp.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = _read_lines(inp)
    disk = _parse_diskann_table(lines)
    div = _parse_divergence_table(lines)
    div_saq = _parse_divergence_saq_table(lines)

    if not disk:
        print("ERROR: could not find DiskANN table in input")
        return 2
    if not div:
        print("ERROR: could not find Divergence table in input")
        return 2

    _write_tsv(
        out_dir / "diskann_sweep.tsv",
        ["L", "recall_pct", "qps", "mean_lat_us", "p95_us", "p999_us", "ios_per_q"],
        [
            [r.L, r.recall_pct, r.qps, r.mean_lat_us, r.p95_us, r.p999_us, r.ios_per_q]
            for r in disk
        ],
    )
    _write_tsv(
        out_dir / "divergence_points.tsv",
        ["config", "recall_pct", "qps", "p50_ms", "p99_ms", "phy_per_q"],
        [[r.config, r.recall_pct, r.qps, r.p50_ms, r.p99_ms, r.phy_per_q] for r in div],
    )
    if div_saq:
        _write_tsv(
            out_dir / "divergence_saq_points.tsv",
            ["config", "recall_pct", "qps", "p50_ms", "p99_ms", "adj_per_q", "ref_per_q", "io_per_q", "ref_ms"],
            [
                [r.config, r.recall_pct, r.qps, r.p50_ms, r.p99_ms, r.adj_per_q, r.ref_per_q, r.io_per_q, r.ref_ms]
                for r in div_saq
            ],
        )

    # Plot 1: recall vs IO/q
    _svg_plot(
        title=f"Recall vs IO per Query ({inp.name})",
        x_label="IOs/query (DiskANN) or phy/q (Divergence)",
        y_label="Recall@100 (%)",
        series=[
            {
                "name": "DiskANN sweep",
                "xs": [r.ios_per_q for r in disk],
                "ys": [r.recall_pct for r in disk],
                "color": "#d23f31",
                "kind": "line",
                "marker": "circle",
            },
            {
                "name": "Div FP32 points",
                "xs": [r.phy_per_q for r in div],
                "ys": [r.recall_pct for r in div],
                "color": "#0b62ff",
                "kind": "scatter",
                "marker": "square",
            },
            *(
                [
                    {
                        "name": "Div SAQ points (io/q)",
                        "xs": [r.io_per_q for r in div_saq],
                        "ys": [r.recall_pct for r in div_saq],
                        "color": "#7a3ef0",
                        "kind": "scatter",
                        "marker": "square",
                    },
                ]
                if div_saq
                else []
            ),
        ],
        out_path=out_dir / "recall_vs_io.svg",
    )

    # Plot 2: QPS vs recall
    _svg_plot(
        title=f"QPS vs Recall ({inp.name})",
        x_label="Recall@100 (%)",
        y_label="QPS (1 thread)",
        series=[
            {
                "name": "DiskANN",
                "xs": [r.recall_pct for r in disk],
                "ys": [r.qps for r in disk],
                "color": "#d23f31",
                "kind": "line",
                "marker": "circle",
            },
            {
                "name": "Div FP32",
                "xs": [r.recall_pct for r in div],
                "ys": [r.qps for r in div],
                "color": "#0b62ff",
                "kind": "scatter",
                "marker": "square",
            },
            *(
                [
                    {
                        "name": "Div SAQ",
                        "xs": [r.recall_pct for r in div_saq],
                        "ys": [r.qps for r in div_saq],
                        "color": "#7a3ef0",
                        "kind": "scatter",
                        "marker": "square",
                    },
                ]
                if div_saq
                else []
            ),
        ],
        out_path=out_dir / "qps_vs_recall.svg",
    )

    # Plot 3: latency vs recall (DiskANN mean/p95/p999; Divergence p50/p99)
    _svg_plot(
        title=f"Latency vs Recall ({inp.name})",
        x_label="Recall@100 (%)",
        y_label="Latency (us)",
        series=[
            {
                "name": "DiskANN mean",
                "xs": [r.recall_pct for r in disk],
                "ys": [r.mean_lat_us for r in disk],
                "color": "#d23f31",
                "kind": "line",
                "marker": "circle",
            },
            {
                "name": "DiskANN p95",
                "xs": [r.recall_pct for r in disk],
                "ys": [r.p95_us for r in disk],
                "color": "#ff8f4a",
                "kind": "line",
                "marker": "circle",
            },
            {
                "name": "DiskANN p999",
                "xs": [r.recall_pct for r in disk],
                "ys": [r.p999_us for r in disk],
                "color": "#7a3ef0",
                "kind": "line",
                "marker": "circle",
            },
            {
                "name": "Div p50",
                "xs": [r.recall_pct for r in div],
                "ys": [r.p50_ms * 1000.0 for r in div],
                "color": "#0b62ff",
                "kind": "scatter",
                "marker": "square",
            },
            {
                "name": "Div p99",
                "xs": [r.recall_pct for r in div],
                "ys": [r.p99_ms * 1000.0 for r in div],
                "color": "#1a9c5a",
                "kind": "scatter",
                "marker": "square",
            },
            *(
                [
                    {
                        "name": "SAQ p50",
                        "xs": [r.recall_pct for r in div_saq],
                        "ys": [r.p50_ms * 1000.0 for r in div_saq],
                        "color": "#7a3ef0",
                        "kind": "scatter",
                        "marker": "square",
                    },
                    {
                        "name": "SAQ p99",
                        "xs": [r.recall_pct for r in div_saq],
                        "ys": [r.p99_ms * 1000.0 for r in div_saq],
                        "color": "#9b7dff",
                        "kind": "scatter",
                        "marker": "square",
                    },
                ]
                if div_saq
                else []
            ),
        ],
        out_path=out_dir / "latency_vs_recall.svg",
    )

    print(f"Wrote: {out_dir / 'diskann_sweep.tsv'}")
    print(f"Wrote: {out_dir / 'divergence_points.tsv'}")
    if div_saq:
        print(f"Wrote: {out_dir / 'divergence_saq_points.tsv'}")
    print(f"Wrote: {out_dir / 'recall_vs_io.svg'}")
    print(f"Wrote: {out_dir / 'qps_vs_recall.svg'}")
    print(f"Wrote: {out_dir / 'latency_vs_recall.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
