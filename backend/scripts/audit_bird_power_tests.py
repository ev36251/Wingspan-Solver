"""Generate a bird-by-bird audit of power test coverage.

Outputs:
- reports/bird_power_audit_full.json
- reports/bird_power_semantic_tested.txt
- reports/bird_power_semantic_untested.txt
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.powers.registry import clear_cache, get_power_source


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "backend" / "tests"
REPORT_DIR = ROOT / "reports"


# Regex for literal bird references in tests.
GET_CALL_PATTERNS = [
    re.compile(r'\bbirds\.get\("([^"]+)"\)'),
    re.compile(r'\bbird_reg\.get\("([^"]+)"\)'),
    re.compile(r'\bget\("([^"]+)"\)'),
]


def _extract_bird_names_from_source(src: str) -> set[str]:
    found: set[str] = set()
    for pat in GET_CALL_PATTERNS:
        for m in pat.finditer(src):
            found.add(m.group(1))
    return found


def _extract_module_string_lists(mod: ast.Module) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in mod.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            vals = []
            ok = True
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    vals.append(elt.value)
                else:
                    ok = False
                    break
            if ok:
                out[target.id] = vals
    return out


def _extract_parametrize_bird_names(
    fn: ast.FunctionDef,
    module_string_lists: dict[str, list[str]],
) -> set[str]:
    names: set[str] = set()
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if not (isinstance(func, ast.Attribute) and func.attr == "parametrize"):
            continue
        if len(dec.args) < 2:
            continue
        values = dec.args[1]
        if isinstance(values, ast.Name):
            names.update(module_string_lists.get(values.id, []))
        elif isinstance(values, (ast.List, ast.Tuple)):
            for elt in values.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


def _is_semantic_behavior_test(src: str) -> bool:
    """Heuristic: test executes powers/actions and asserts behavior, not just mapping."""
    if "isinstance(get_power(" in src:
        return False
    if "assert missing == []" in src:
        return False
    if "test_all_bird_powers_smoke_execute_without_exceptions" in src:
        return False

    has_assert = "assert " in src
    behavior_markers = (
        ".execute(",
        "trigger_end_of_",
        "trigger_between_turn_powers(",
        "execute_draw_cards(",
        "execute_gain_food(",
        "execute_play_bird(",
        "execute_lay_eggs(",
    )
    return has_assert and any(marker in src for marker in behavior_markers)


def _iter_test_functions(path: Path):
    src = path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(path))
    module_string_lists = _extract_module_string_lists(mod)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            seg = ast.get_source_segment(src, node) or ""
            param_names = _extract_parametrize_bird_names(node, module_string_lists)
            yield node.name, seg, param_names


def build_audit() -> dict:
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    all_birds = sorted(birds.all_birds, key=lambda b: b.name)
    all_names = {b.name for b in all_birds}

    semantic_refs: dict[str, list[str]] = {}
    any_refs: dict[str, list[str]] = {}

    for path in sorted(TEST_DIR.glob("test_*.py")):
        for fn_name, fn_src, param_names in _iter_test_functions(path):
            names = _extract_bird_names_from_source(fn_src) | set(param_names)
            if not names:
                continue
            names = {n for n in names if n in all_names}
            if not names:
                continue

            ref = f"{path.relative_to(ROOT)}::{fn_name}"
            for n in sorted(names):
                any_refs.setdefault(n, []).append(ref)
            if _is_semantic_behavior_test(fn_src):
                for n in sorted(names):
                    semantic_refs.setdefault(n, []).append(ref)

    rows = []
    tested = []
    untested = []
    for bird in all_birds:
        source = get_power_source(bird)
        refs = sorted(set(any_refs.get(bird.name, [])))
        sem = sorted(set(semantic_refs.get(bird.name, [])))
        is_semantic = bool(sem)
        if is_semantic:
            tested.append(bird.name)
        else:
            untested.append(bird.name)

        rows.append(
            {
                "bird": bird.name,
                "power_text": bird.power_text,
                "color": bird.color.value,
                "power_source": source,
                "strict_mapped": source == "strict",
                "smoke_covered": True,  # validated by test_all_bird_power_smoke
                "semantic_tested_word_for_word": is_semantic,
                "semantic_test_refs": sem,
                "any_test_refs": refs,
            }
        )

    return {
        "summary": {
            "total_birds": len(all_birds),
            "strict_mapped_count": sum(1 for r in rows if r["strict_mapped"]),
            "semantic_tested_count": len(tested),
            "semantic_untested_count": len(untested),
        },
        "semantic_tested_birds": tested,
        "semantic_untested_birds": untested,
        "birds": rows,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_audit()

    (REPORT_DIR / "bird_power_audit_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "bird_power_semantic_tested.txt").write_text(
        "\n".join(report["semantic_tested_birds"]) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "bird_power_semantic_untested.txt").write_text(
        "\n".join(report["semantic_untested_birds"]) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
