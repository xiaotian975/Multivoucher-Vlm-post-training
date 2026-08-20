"""Generate editable draw.io sources and matching PNGs for the Phase 10 report."""

from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    fill: str
    stroke: str = "#37474F"


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: str
    target: str
    label: str = ""
    dashed: bool = False
    color: str = "#546E7A"


def _diagram_xml(title: str, nodes: list[Node], edges: list[Edge], width: int, height: int) -> str:
    mxfile = ET.Element(
        "mxfile",
        {"host": "app.diagrams.net", "agent": "Codex", "version": "24.7.17"},
    )
    diagram = ET.SubElement(mxfile, "diagram", {"name": title, "id": title.replace(" ", "-")})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(width),
            "dy": str(height),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(width),
            "pageHeight": str(height),
            "math": "0",
            "shadow": "0",
            "defaultFontFamily": "Noto Sans JP",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    for edge in edges:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeColor={edge.color};strokeWidth=2;endArrow=block;"
            "fontFamily=Noto Sans JP;fontSize=14;"
        )
        if edge.dashed:
            style += "dashed=1;"
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": edge.edge_id,
                "value": html.escape(edge.label),
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": edge.source,
                "target": edge.target,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    for node in nodes:
        style = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
            f"fillColor={node.fill};strokeColor={node.stroke};strokeWidth=2;"
            "fontColor=#263238;fontFamily=Noto Sans JP;fontSize=16;spacing=8;"
        )
        cell = ET.SubElement(
            root,
            "mxCell",
            {"id": node.node_id, "value": html.escape(node.label), "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(node.x),
                "y": str(node.y),
                "width": str(node.width),
                "height": str(node.height),
                "as": "geometry",
            },
        )
    ET.indent(mxfile, space="  ")
    return ET.tostring(mxfile, encoding="unicode", xml_declaration=False) + "\n"


def _center(node: Node) -> tuple[float, float]:
    return node.x + node.width / 2, node.y + node.height / 2


def _render_png(title: str, nodes: list[Node], edges: list[Edge], width: int, height: int, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(16, 9), dpi=160)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    by_id = {node.node_id: node for node in nodes}
    for edge in edges:
        source = by_id[edge.source]
        target = by_id[edge.target]
        start = _center(source)
        end = _center(target)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=2,
            color=edge.color,
            linestyle="--" if edge.dashed else "-",
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
        axis.add_patch(arrow)
        if edge.label:
            axis.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 - 12,
                edge.label,
                ha="center",
                va="center",
                fontsize=10,
                color="#37474F",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
                zorder=3,
            )
    for node in nodes:
        patch = FancyBboxPatch(
            (node.x, node.y),
            node.width,
            node.height,
            boxstyle="round,pad=0.015,rounding_size=8",
            facecolor=node.fill,
            edgecolor=node.stroke,
            linewidth=2,
            zorder=2,
        )
        axis.add_patch(patch)
        axis.text(
            node.x + node.width / 2,
            node.y + node.height / 2,
            node.label,
            ha="center",
            va="center",
            fontsize=10.5 if width > 1600 else 12,
            color="#263238",
            linespacing=1.35,
            zorder=3,
        )
    axis.text(40, 48, title, fontsize=22, fontweight="bold", color="#1F2937", ha="left", va="center")
    axis.text(
        40,
        height - 22,
        "Train-only development evidence; sample500/Test/final holdout are excluded from selection.",
        fontsize=9.5,
        color="#607D8B",
        ha="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pipeline() -> tuple[str, list[Node], list[Edge]]:
    nodes = [
        Node("train_data", "Train-only MultiVoucher cases\nstructured targets + evidence", 55, 130, 245, 105, "#E3F2FD"),
        Node("sft", "Structured Repair SFT v3\n480-case repair mix", 365, 130, 245, 105, "#E8F5E9", "#2E7D32"),
        Node("rollout", "Model sampling\n240 cases x 4 completions", 675, 130, 245, 105, "#FFF8E1", "#F9A825"),
        Node("pairs", "Model-error-mined pairs\n120 train + 24 probe", 985, 130, 245, 105, "#F3E5F5", "#7B1FA2"),
        Node("dpo", "DPO v3\nweak 40-step + strong continuation", 1295, 130, 245, 105, "#FCE4EC", "#C2185B"),
        Node("probe", "Case-disjoint alignment probe\nreward +0.167; pair rate +11.1pp", 790, 365, 300, 115, "#E0F2F1", "#00796B"),
        Node("selected", "Checkpoint-15 selected\nprobe-local alignment signal", 1165, 365, 275, 115, "#FFF3E0", "#EF6C00"),
        Node("full_gate", "152-case train_decode_dev gate\nJSON / schema / audit / HRM / evidence", 935, 610, 335, 125, "#ECEFF1", "#455A64"),
    ]
    edges = [
        Edge("e1", "train_data", "sft", "SFT"),
        Edge("e2", "sft", "rollout", "generate"),
        Edge("e3", "rollout", "pairs", "score + audit"),
        Edge("e4", "pairs", "dpo", "preference train"),
        Edge("e5", "sft", "probe", "baseline", True),
        Edge("e6", "dpo", "probe", "checkpoints"),
        Edge("e7", "probe", "selected", "earliest eligible"),
        Edge("e8", "selected", "full_gate", "full validation"),
        Edge("e9", "sft", "full_gate", "business baseline", True, "#2E7D32"),
    ]
    return "MultiVoucher-Audit VLM post-training pipeline", nodes, edges


def _selection() -> tuple[str, list[Node], list[Edge]]:
    nodes = [
        Node("sft_input", "SFT v3\nAudit 96.71% | HRM 5.75%\nJSON/schema 100%", 80, 125, 330, 125, "#E8F5E9", "#2E7D32"),
        Node("dpo_input", "DPO v3 checkpoint-15\nprobe reward +0.167\npair rate +11.1pp", 1190, 125, 330, 125, "#FFF3E0", "#EF6C00"),
        Node("contract", "Contract gate\nJSON validity + schema compliance", 475, 125, 270, 105, "#E3F2FD", "#1565C0"),
        Node("evidence", "Evidence gate\nsupport + hallucination + bbox", 855, 125, 270, 105, "#F3E5F5", "#7B1FA2"),
        Node("business", "Business gate\nAudit accuracy + High-risk Miss", 470, 365, 285, 110, "#FFEBEE", "#C62828"),
        Node("attribution", "Paired attribution\n5 old HRM retained + 7 introduced\n14 reject -> manual_review", 850, 355, 300, 130, "#ECEFF1", "#455A64"),
        Node("production", "repair_sft_r3\nPRODUCTION_CANDIDATE\nNOT_DEPLOYED", 245, 620, 350, 130, "#C8E6C9", "#1B5E20"),
        Node("research", "DPO checkpoint-15\nALIGNMENT_RESEARCH_CANDIDATE\ndeployment_eligible=false", 1005, 620, 380, 130, "#FFE0B2", "#E65100"),
        Node("holdout", "Final holdout not run\nNo production deployment claim", 625, 670, 315, 105, "#FAFAFA", "#616161"),
    ]
    edges = [
        Edge("s1", "sft_input", "contract"),
        Edge("s2", "dpo_input", "evidence"),
        Edge("s3", "contract", "business"),
        Edge("s4", "evidence", "attribution"),
        Edge("s5", "business", "production", "passes dev gate", False, "#2E7D32"),
        Edge("s6", "attribution", "research", "full gate not met", False, "#C62828"),
        Edge("s7", "production", "holdout", "before deployment", True),
        Edge("s8", "research", "holdout", "research only", True),
    ]
    return "Model selection and release gate", nodes, edges


def _parallelism() -> tuple[str, list[Node], list[Edge]]:
    nodes = [
        Node("sft_launch", "SFT v3 launcher\ntorchrun --nproc_per_node=5", 45, 115, 255, 105, "#E3F2FD", "#1565C0"),
        Node("sft_workers", "5-process DDP\nGPU 0..4: full Qwen3-VL + LoRA replica\n480 cases / global batch 5 = 96 steps", 390, 95, 450, 145, "#E8F5E9", "#2E7D32"),
        Node("sft_sync", "NCCL gradient synchronization\nLoRA trainable params: 43.65M\nBF16 + gradient checkpointing", 940, 105, 370, 125, "#E0F2F1", "#00796B"),
        Node("dpo_launch", "DPO v2/v3 launcher\none Python process; 5 visible GPUs", 45, 330, 255, 105, "#FFF3E0", "#EF6C00"),
        Node("dpo_shard", "device_map=auto / balanced\nPolicy + reference layers split over GPU 0..4\nModel parallelism, not DDP", 390, 305, 450, 145, "#FCE4EC", "#C2185B"),
        Node("dpo_tradeoff", "Why: two 8B models exceed one 32GB GPU\nBenefit: fits memory\nTradeoff: utilization may be uneven", 940, 315, 370, 125, "#FFEBEE", "#C62828"),
        Node("infer_launch", "Inference / model mining\nnum_shards=5", 45, 555, 255, 105, "#F3E5F5", "#7B1FA2"),
        Node("infer_workers", "GPU 0..4 independent case shards\nEach worker writes one JSONL\n240 x 4 rollouts use the same partition", 390, 530, 450, 145, "#EDE7F6", "#5E35B1"),
        Node("merge", "Deterministic shard merge\ncase-id completeness + disjointness checks", 940, 540, 370, 125, "#ECEFF1", "#455A64"),
        Node("probe", "Checkpoint-level parallel probe\nsteps 5/10/15/20\non separate GPUs\nselected checkpoint runs full gate", 1390, 515, 350, 155, "#FFF8E1", "#F9A825"),
    ]
    edges = [
        Edge("p1", "sft_launch", "sft_workers"),
        Edge("p2", "sft_workers", "sft_sync"),
        Edge("p3", "dpo_launch", "dpo_shard"),
        Edge("p4", "dpo_shard", "dpo_tradeoff"),
        Edge("p5", "infer_launch", "infer_workers"),
        Edge("p6", "infer_workers", "merge"),
        Edge("p7", "dpo_tradeoff", "probe", dashed=True),
        Edge("p8", "merge", "probe", dashed=True),
    ]
    return "Five-GPU execution topology: DDP, model sharding, and task sharding", nodes, edges


def _write(name: str, title: str, nodes: list[Node], edges: list[Edge], output_dir: Path) -> None:
    width, height = (1800, 900) if name == "multi_gpu_execution_topology" else (1600, 850)
    (output_dir / f"{name}.drawio").write_text(
        _diagram_xml(title, nodes, edges, width, height),
        encoding="utf-8",
    )
    _render_png(title, nodes, edges, width, height, output_dir / f"{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="docs/experiments/phase10_model_error_mined_dpo_v3/figures",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write("post_training_pipeline", *_pipeline(), output_dir)
    _write("model_selection_gate", *_selection(), output_dir)
    _write("multi_gpu_execution_topology", *_parallelism(), output_dir)
    print(f"generated_diagrams={output_dir}")


if __name__ == "__main__":
    main()
