"""Generate a bird-by-bird signed checklist for power mapping certification.

Outputs:
- reports/bird_power_signed_checklist.csv
- reports/bird_power_signed_checklist.md
- reports/bird_power_signed_checklist.meta.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.powers.registry import clear_cache, get_power, get_power_source
from backend.scripts.audit_bird_power_tests import build_audit


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
CSV_PATH = REPORT_DIR / "bird_power_signed_checklist.csv"
MD_PATH = REPORT_DIR / "bird_power_signed_checklist.md"
META_PATH = REPORT_DIR / "bird_power_signed_checklist.meta.json"


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _row_signature(payload: dict) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:24]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audit = build_audit()
    audit_rows = {row["bird"]: row for row in audit["birds"]}

    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()

    generated_at = datetime.now(timezone.utc).isoformat()
    revision = _git_revision()

    rows: list[dict] = []
    for bird in sorted(birds.all_birds, key=lambda b: b.name):
        power = get_power(bird)
        source = get_power_source(bird)
        audit_row = audit_rows.get(bird.name, {})

        mapped_class = f"{type(power).__module__}.{type(power).__name__}"
        mapped_kwargs = dict(getattr(power, "__dict__", {}))
        strict_mapped = source == "strict"
        semantic_tested = bool(audit_row.get("semantic_tested_word_for_word", False))
        smoke_covered = bool(audit_row.get("smoke_covered", False))

        status = "PASS" if strict_mapped and semantic_tested and smoke_covered else "REVIEW_REQUIRED"
        notes = ""
        if not bird.power_text:
            notes = "No power text on card"

        payload = {
            "bird": bird.name,
            "power_text": bird.power_text,
            "mapped_class": mapped_class,
            "mapped_kwargs": mapped_kwargs,
            "source": source,
            "strict_mapped": strict_mapped,
            "semantic_tested": semantic_tested,
            "smoke_covered": smoke_covered,
            "status": status,
            "revision": revision,
            "generated_at": generated_at,
        }

        rows.append(
            {
                "bird": bird.name,
                "power_text": bird.power_text,
                "mapped_class": mapped_class,
                "mapped_kwargs": _stable_json(mapped_kwargs),
                "power_source": source,
                "strict_mapped": strict_mapped,
                "semantic_tested_word_for_word": semantic_tested,
                "smoke_covered": smoke_covered,
                "status": status,
                "notes": notes,
                "row_signature": _row_signature(payload),
            }
        )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bird",
                "power_text",
                "mapped_class",
                "mapped_kwargs",
                "power_source",
                "strict_mapped",
                "semantic_tested_word_for_word",
                "smoke_covered",
                "status",
                "notes",
                "row_signature",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    csv_bytes = CSV_PATH.read_bytes()
    checklist_signature = hashlib.sha256(csv_bytes).hexdigest()

    total = len(rows)
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    review_count = total - pass_count
    strict_count = sum(1 for r in rows if r["strict_mapped"])
    semantic_count = sum(1 for r in rows if r["semantic_tested_word_for_word"])

    reviewer = "Codex (GPT-5)"
    signed_by = f"{reviewer} @ {generated_at} UTC"

    md = [
        "# Bird-by-Bird Signed Checklist",
        "",
        "## Certification",
        f"- Reviewer: {reviewer}",
        f"- Generated at: {generated_at}",
        f"- Git revision: `{revision}`",
        f"- Checklist signature (SHA-256): `{checklist_signature}`",
        f"- Signed by: `{signed_by}`",
        "",
        "## Summary",
        f"- Total birds: **{total}**",
        f"- PASS: **{pass_count}**",
        f"- REVIEW_REQUIRED: **{review_count}**",
        f"- Strict mapped count: **{strict_count}**",
        f"- Semantic tested count: **{semantic_count}**",
        "",
        "## Files",
        f"- CSV: `{CSV_PATH.relative_to(ROOT)}`",
        f"- Metadata: `{META_PATH.relative_to(ROOT)}`",
        "",
        "## Sign-Off Statement",
        "I certify this checklist was generated by verifying each bird against the Excel power text,",
        "its resolved mapped implementation, and semantic test coverage status at generation time.",
    ]
    MD_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")

    meta = {
        "generated_at": generated_at,
        "reviewer": reviewer,
        "git_revision": revision,
        "total_birds": total,
        "pass_count": pass_count,
        "review_required_count": review_count,
        "strict_mapped_count": strict_count,
        "semantic_tested_count": semantic_count,
        "checklist_signature_sha256": checklist_signature,
        "csv_path": str(CSV_PATH.relative_to(ROOT)),
        "markdown_path": str(MD_PATH.relative_to(ROOT)),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
