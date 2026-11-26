"""Graphify-compatible Python AST Code Knowledge Graph Builder for SE374.R11.

Scans all tracked academic Python modules across `src/`, `scripts/`, and `tests/`
using Python standard library only (`ast`, `json`, `pathlib`) and writes:
- `graphify-out/graph.json`: module/class/function/LOC/layer nodes, internal import
  edges, layer summaries, and hub fan-in rankings.
- `graphify-out/GRAPH_REPORT.md`: Vietnamese architecture & call-chain report with
  Mermaid diagrams for the 3-person team.

Pure ASCII hyphen '-' (0x2D) only; zero external dependencies.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "scripts", "tests")
OUT_DIR = PROJECT_ROOT / "graphify-out"

RESEARCH_FOCUS = (
    "src.datasets.patient_split", "src.datasets.otsu_filter", "src.models.wsss_inference_methods",
    "src.stage2_v2.models", "src.stage2_v2.trainer", "src.stage2_v2.ensemble", "src.stage2_v2.visual_audit",
)
SOFTWARE_FOCUS = (
    "src.visualization.api_server", "src.visualization.clinical_viewer",
    "src.visualization.draw_gate7_monochrome_figures", "src.utils.deliverable_auditor",
    "src.pipeline.wsi_lowmag_highmag_pipeline", "src.colab_pkg.model_loader",
)
CALL_CHAINS = (
    ("src.stage2_v2.trainer", "src.stage2_v2.memmap_dataset", "load memmap"),
    ("src.stage2_v2.trainer", "src.stage2_v2.models", "4-Arm Dual-Head"),
    ("src.stage2_v2.trainer", "src.stage2_v2.losses", "OEEM/KD loss"),
    ("src.stage2_v2.ensemble", "src.stage2_v2.models", "4-Arm infer"),
    ("src.stage2_v2.ensemble", "src.stage2_v2.evaluator", "val/test mIoU"),
    ("src.stage2_v2.visual_audit", "src.stage2_v2.ensemble", "5-Expert CGM"),
    ("src.stage2_v2.visual_audit", "src.datasets.otsu_filter", "Pure-Glass Guard"),
    ("src.visualization.api_server", "src.stage2_v2.ensemble", "Gate-7 infer"),
    ("src.visualization.api_server", "src.models.wsss_inference_methods", "M1-M7 infer"),
    ("src.visualization.api_server", "src.visualization.clinical_viewer", "CAP PDF/PNG"),
)


def _module_id(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else rel_path.stem


def _layer_of(rel_path: Path) -> str:
    parts = rel_path.parts
    if not parts:
        return "root"
    return f"src/{parts[1]}" if parts[0] == "src" and len(parts) > 2 else ("src/root" if parts[0] == "src" else parts[0])


def _resolve_rel(current_mod: str, is_init: bool, level: int, module: str | None) -> str:
    pkg = current_mod.split(".") if current_mod else []
    if not is_init and pkg:
        pkg = pkg[:-1]
    up = max(0, level - 1)
    base = pkg[: len(pkg) - up] if up <= len(pkg) else []
    if module:
        base.extend(module.split("."))
    return ".".join(p for p in base if p)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_name(node.value)}.{node.attr}"
    return _base_name(node.value) if isinstance(node, ast.Subscript) else "expr"


def _parse_module(py_path: Path, root: Path) -> Dict[str, Any]:
    rel = py_path.relative_to(root)
    rel_posix, mod_id, is_init = rel.as_posix(), _module_id(rel), py_path.name == "__init__.py"
    raw = py_path.read_text(encoding="utf-8", errors="replace")
    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    raw_imports: List[Tuple[str, List[str]]] = []
    doc_first, parse_err = "", ""

    try:
        tree = ast.parse(raw, filename=rel_posix)
        doc = (ast.get_docstring(tree) or "").strip()
        doc_first = doc.splitlines()[0][:140] if doc else ""
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                m_cnt = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in node.body)
                classes.append({"name": node.name, "lineno": node.lineno, "bases": [_base_name(b) for b in node.bases], "methods_count": m_cnt})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({"name": node.name, "lineno": node.lineno, "is_async": isinstance(node, ast.AsyncFunctionDef)})
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                raw_imports.extend((alias.name, []) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                tgt = _resolve_rel(mod_id, is_init, node.level, node.module) if node.level > 0 else (node.module or "")
                syms = [a.name for a in node.names if a.name != "*"]
                if tgt:
                    raw_imports.append((tgt, syms))
    except SyntaxError as exc:
        parse_err = f"SyntaxError line {exc.lineno}: {exc.msg}"

    return {
        "id": mod_id, "type": "module", "module": mod_id, "path": rel_posix, "layer": _layer_of(rel),
        "loc": len(raw.splitlines()), "docstring": doc_first, "classes": classes, "functions": functions,
        "raw_imports": raw_imports, "parse_error": parse_err,
    }


def _match_targets(
    src: str, is_init: bool, imp: str, syms: List[str],
    known: Set[str], known_pkgs: Set[str], shorts: Dict[str, str],
) -> List[Tuple[str, List[str]]]:
    """Resolves an import statement to exact (target_module, matched_symbols) pairs."""
    pkg = src if is_init else src.rpartition(".")[0]
    cands = [imp]
    if imp and not imp.startswith("src."):
        cands.append(f"src.{imp}")
    if pkg and imp and "." not in imp:
        cands.append(f"{pkg}.{imp}")
    if imp in shorts:
        cands.append(shorts[imp])

    resolved_base = next((c for c in cands if c in known), "")
    prefix = resolved_base or (f"src.{imp}" if imp and f"src.{imp}" in known_pkgs else imp)
    submod_hits, rest_syms = [], []
    for sym in syms:
        sub = f"{prefix}.{sym}" if prefix else sym
        if sub in known:
            submod_hits.append((sub, [sym]))
        else:
            rest_syms.append(sym)

    out: List[Tuple[str, List[str]]] = list(submod_hits)
    if resolved_base and (rest_syms or not submod_hits):
        out.append((resolved_base, rest_syms))
    if out:
        return out

    for cand in (imp, f"src.{imp}" if imp and not imp.startswith("src.") else ""):
        parts = cand.split(".") if cand else []
        min_len = 2 if cand.startswith("src.") else 1
        for i in range(len(parts) - 1, min_len - 1, -1):
            pref = ".".join(parts[:i])
            if pref in known:
                return [(pref, syms)]
    return []


def build_code_graph(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Scans configured roots and builds the full AST code knowledge graph."""
    py_files = [
        p for r in SCAN_ROOTS if (project_root / r).is_dir()
        for p in sorted((project_root / r).rglob("*.py")) if "__pycache__" not in p.parts
    ]
    nodes = [_parse_module(p, project_root) for p in py_files]
    known = {n["id"] for n in nodes}
    known_pkgs = {k.rpartition(".")[0] for k in known if "." in k}
    is_init_map = {n["id"]: n["path"].endswith("/__init__.py") or n["path"] == "__init__.py" for n in nodes}
    stem_hits: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        stem = Path(n["path"]).stem
        if n["layer"] in ("scripts", "src/colab_pkg") and stem != "__init__":
            stem_hits[stem].append(n["id"])
    shorts = {stem: mids[0] for stem, mids in stem_hits.items() if len(mids) == 1}

    edge_syms: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    fan_in: Dict[str, Set[str]] = defaultdict(set)
    fan_out: Dict[str, Set[str]] = defaultdict(set)
    ext_counts: Counter[str] = Counter()

    for node in nodes:
        src = node["id"]
        ext_mods: Set[str] = set()
        for imp, syms in node.pop("raw_imports"):
            matches = _match_targets(src, is_init_map[src], imp, syms, known, known_pkgs, shorts)
            if matches:
                for tgt, msyms in matches:
                    if tgt != src:
                        edge_syms[(src, tgt)].update(msyms)
                        fan_in[tgt].add(src)
                        fan_out[src].add(tgt)
            elif imp:
                top = imp.split(".")[0]
                ext_mods.add(top)
                ext_counts[top] += 1
        node["external_imports"] = sorted(ext_mods)

    for node in nodes:
        nid = node["id"]
        node["internal_imports"] = sorted(fan_out.get(nid, set()))
        node["imported_by"] = sorted(fan_in.get(nid, set()))
        node["fan_in"], node["fan_out"] = len(node["imported_by"]), len(node["internal_imports"])

    edges = [
        {"source": s, "target": t, "kind": "internal_import", "symbols": sorted(syms)[:12]}
        for (s, t), syms in sorted(edge_syms.items())
    ]
    by_id = {n["id"]: n for n in nodes}
    layers: Dict[str, Dict[str, int]] = {}
    for n in nodes:
        st = layers.setdefault(n["layer"], {"modules": 0, "classes": 0, "functions": 0, "loc": 0, "fan_in_sum": 0})
        st["modules"] += 1
        st["classes"] += len(n["classes"])
        st["functions"] += len(n["functions"])
        st["loc"] += n["loc"]
        st["fan_in_sum"] += n["fan_in"]

    l_edges: Counter[Tuple[str, str]] = Counter()
    for e in edges:
        ls, lt = by_id[e["source"]]["layer"], by_id[e["target"]]["layer"]
        if ls != lt:
            l_edges[(ls, lt)] += 1

    hubs = [
        {
            "rank": i, "module": h["id"], "path": h["path"], "layer": h["layer"], "fan_in": h["fan_in"],
            "fan_out": h["fan_out"], "loc": h["loc"], "classes": len(h["classes"]), "functions": len(h["functions"]),
        }
        for i, h in enumerate(sorted(nodes, key=lambda x: (-x["fan_in"], -x["loc"], x["id"]))[:20], start=1)
        if h["fan_in"] > 0
    ]
    return {
        "schema_version": "graphify-ast-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_ast_code_graph.py",
        "scanned_roots": list(SCAN_ROOTS),
        "summary": {
            "total_modules": len(nodes),
            "total_classes": sum(len(n["classes"]) for n in nodes),
            "total_functions": sum(len(n["functions"]) for n in nodes),
            "total_loc": sum(n["loc"] for n in nodes),
            "total_internal_edges": len(edges),
            "parse_errors": sum(1 for n in nodes if n["parse_error"]),
        },
        "layers": dict(sorted(layers.items())),
        "layer_edges": [
            {"source_layer": s, "target_layer": t, "weight": w}
            for (s, t), w in sorted(l_edges.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "hubs_by_fan_in": hubs,
        "top_external_packages": [{"package": p, "import_refs": c} for p, c in ext_counts.most_common(15)],
        "nodes": nodes,
        "edges": edges,
    }


def _mid(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").replace(".", "_")


def _format_focus_lines(mod_ids: Tuple[str, ...], by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for mid in mod_ids:
        nd = by_id.get(mid)
        if not nd:
            continue
        deps = ", ".join(f"`{d}`" for d in nd["internal_imports"]) or "không có (tầng nền độc lập)"
        out.append(
            f"- [{nd['path']}](../{nd['path']}) (`{nd['loc']}` LOC, fan-in `{nd['fan_in']}`, "
            f"fan-out `{nd['fan_out']}`): phụ thuộc nội bộ: {deps}."
        )
    return out


def render_graph_report_md(graph: Dict[str, Any]) -> str:
    """Renders Vietnamese + Mermaid architecture & call-chain report for the 3-person team."""
    s, layers = graph["summary"], graph["layers"]
    by_id = {n["id"]: n for n in graph["nodes"]}
    edge_set = {(e["source"], e["target"]) for e in graph["edges"]}
    core_layers = [k for k in layers if k not in ("tests", "scripts")]

    lines: List[str] = [
        "# Báo Cáo Đồ Thị Tri Thức Mã Nguồn AST (`graphify-out/GRAPH_REPORT.md`)",
        "",
        f"Tài liệu này được sinh tự động bằng trình phân tích cú pháp Python AST (`scripts/build_ast_code_graph.py`) trên 3 thư mục `src/`, `scripts/` và `tests/` vào `{graph['generated_at_utc']}`.",
        "",
        "## 1. Tổng Quan Quy Mô Mã Nguồn",
        "",
        "| Chỉ Số | Giá Trị Thực Tế Từ AST |",
        "| :--- | :---: |",
        f"| Tổng số module Python (`.py`) | **{s['total_modules']}** |",
        f"| Tổng số lớp (`class`) | **{s['total_classes']}** |",
        f"| Tổng số hàm cấp module (`def` / `async def`) | **{s['total_functions']}** |",
        f"| Tổng số dòng mã (LOC) | **{s['total_loc']:,}** |",
        f"| Tổng số cạnh import nội bộ | **{s['total_internal_edges']}** |",
        f"| Lỗi cú pháp AST (`parse_errors`) | **{s['parse_errors']}** |",
        "",
        "## 2. Phân Bổ Theo Tầng Kiến Trúc (Layer Map)",
        "",
        "| Tầng Kiến Trúc | Số Module | Số Lớp | Số Hàm | Tổng LOC | Tổng Fan-In |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        *[f"| `{k}` | {v['modules']} | {v['classes']} | {v['functions']} | {v['loc']:,} | {v['fan_in_sum']} |" for k, v in layers.items()],
        "",
        "## 3. Sơ Đồ Kiến Trúc Phụ Thuộc Giữa Các Tầng (Mermaid)",
        "",
        "```mermaid",
        "flowchart LR",
        *[f'    {_mid(k)}["{k} ({layers[k]["modules"]} py, {layers[k]["loc"]} LOC)"]' for k in core_layers],
        *[
            f"    {_mid(le['source_layer'])} -->|{le['weight']}| {_mid(le['target_layer'])}"
            for le in graph["layer_edges"]
            if le["source_layer"] in core_layers and le["target_layer"] in core_layers
        ],
        "```",
        "",
        "## 4. Top Module Trọng Yếu Theo Fan-In (Impact Hubs)",
        "",
        "Những module có `Fan-In` cao là điểm hội tụ phụ thuộc lớn nhất; nhóm 3 người cần chạy `pytest` liên quan ngay sau khi chỉnh sửa:",
        "",
        "| Hạng | Module | Tệp Nguồn | Tầng | Fan-In | Fan-Out | LOC |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: |",
        *[
            f"| {h['rank']} | `{h['module']}` | [{h['path']}](../{h['path']}) | `{h['layer']}` | **{h['fan_in']}** | {h['fan_out']} | {h['loc']} |"
            for h in graph["hubs_by_fan_in"][:15]
        ],
        "",
        "## 5. Chuỗi Phụ Thuộc Trọng Yếu (Nghiên Cứu vs Phần Mềm)",
        "",
        "### 5.1. Luồng Nghiên Cứu (Training, Stage-2 v2 Gate-7 & Evaluation)",
        "",
        *_format_focus_lines(RESEARCH_FOCUS, by_id),
        "",
        "### 5.2. Luồng Phần Mềm (Trạm Lâm Sàng FastAPI, Pipeline WSI & Deliverable Auditor)",
        "",
        *_format_focus_lines(SOFTWARE_FOCUS, by_id),
        "",
        "### 5.3. Sơ Đồ Chuỗi Gọi Mô-Đun Đầu-Cuối (AST-Verified Call Chains)",
        "",
        "```mermaid",
        "flowchart TD",
        *[f'    {_mid(src)}["{src}"] -->|{lbl}| {_mid(tgt)}["{tgt}"]' for src, tgt, lbl in CALL_CHAINS if (src, tgt) in edge_set],
        "```",
        "",
        "## 6. Cách Cập Nhật Đồ Thị Sau Khi Sửa Mã Nguồn",
        "",
        "```powershell",
        "python scripts/build_ast_code_graph.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Builds and writes `graphify-out/graph.json` and `graphify-out/GRAPH_REPORT.md`."""
    graph = build_code_graph(PROJECT_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path, md_path = OUT_DIR / "graph.json", OUT_DIR / "GRAPH_REPORT.md"
    json_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_graph_report_md(graph), encoding="utf-8")
    s = graph["summary"]
    print(
        f"[OK] Generated {json_path.relative_to(PROJECT_ROOT)} & {md_path.relative_to(PROJECT_ROOT)} "
        f"(modules={s['total_modules']}, classes={s['total_classes']}, "
        f"functions={s['total_functions']}, loc={s['total_loc']}, edges={s['total_internal_edges']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
