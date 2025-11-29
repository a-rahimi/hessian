#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "src" / "block_partitioned_matrices.py"
README_PATH = PROJECT_ROOT / "README.md"

BEGIN_MARKER = "<!-- BEGIN MATRIX HIERARCHY -->"
END_MARKER = "<!-- END MATRIX HIERARCHY -->"


def _base_name(expr: ast.expr) -> str:
    """Return a string representation for a base class expression."""
    try:
        return ast.unparse(expr)
    except AttributeError:  # pragma: no cover - Py<3.9 fallback
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            parts = []
            while isinstance(expr, ast.Attribute):
                parts.append(expr.attr)
                expr = expr.value
            if isinstance(expr, ast.Name):
                parts.append(expr.id)
            return ".".join(reversed(parts))
        raise ValueError(f"Unsupported base expression: {ast.dump(expr)}")


def _parse_classes() -> dict[str, list[str]]:
    module = ast.parse(SOURCE_PATH.read_text())
    bases: dict[str, list[str]] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            bases[node.name] = [_base_name(base) for base in node.bases]
    return bases


def _build_hierarchy(
    class_bases: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    memo: dict[str, bool] = {}

    def inherits_from_matrix(name: str) -> bool:
        if name == "Matrix":
            return True
        if name in memo:
            return memo[name]
        bases = class_bases.get(name)
        if not bases:
            memo[name] = False
            return False
        memo[name] = any(inherits_from_matrix(base) for base in bases)
        return memo[name]

    tree: dict[str, list[str]] = defaultdict(list)
    extras: dict[str, list[str]] = {}

    for cls_name in class_bases:
        if cls_name == "Matrix" or not inherits_from_matrix(cls_name):
            continue
        bases = class_bases[cls_name]
        parent = "Matrix"
        for base in bases:
            if base == "Matrix" or inherits_from_matrix(base):
                parent = base
                break
        tree[parent].append(cls_name)
        extra_bases = [
            base for base in bases if base != parent and not inherits_from_matrix(base)
        ]
        if extra_bases:
            extras[cls_name] = extra_bases

    tree.setdefault("Matrix", [])
    return tree, extras


def _render_tree(
    tree: dict[str, list[str]], extras: dict[str, list[str]], order: str
) -> str:
    def node_label(name: str) -> str:
        if name in extras:
            extra = ", ".join(extras[name])
            return f"{name} (also {extra})"
        return name

    lines: list[str] = [node_label("Matrix")]

    def walk(parent: str, prefix: str = "") -> None:
        children = tree.get(parent, [])
        if order == "alphabetical":
            children = sorted(children)
        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node_label(child)}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            walk(child, child_prefix)

    walk("Matrix")
    return "\n".join(lines)


def _update_readme(hierarchy: str) -> None:
    block = "\n".join([BEGIN_MARKER, "```", hierarchy, "```", END_MARKER])
    readme_text = README_PATH.read_text()
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL
    )
    if pattern.search(readme_text):
        updated = pattern.sub(block, readme_text)
    else:
        if not readme_text.endswith("\n"):
            readme_text += "\n"
        updated = (
            readme_text + "\n## Partitioned Matrix Class Hierarchy\n\n" + block + "\n"
        )
    README_PATH.write_text(updated)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update README.md with the hierarchy of Matrix subclasses defined in "
            "src/block_partitioned_matrices.py."
        )
    )
    parser.add_argument(
        "--order",
        choices=["source", "alphabetical"],
        default="source",
        help="Order children as they appear in source or alphabetically (default: source).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    class_bases = _parse_classes()
    tree, extras = _build_hierarchy(class_bases)
    hierarchy = _render_tree(tree, extras, order=args.order)
    _update_readme(hierarchy)


if __name__ == "__main__":
    main()
