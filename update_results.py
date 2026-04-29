#!/usr/bin/env python3
"""
Update final/results.md with post-training results.
Called after each phase's post-training eval.

Usage:
  python update_results.py phase1 results/phase1_after.json
  python update_results.py phase2 results/phase2_after.json
  python update_results.py phase3 results/phase3_after.json
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_MD = os.path.join(ROOT, "final", "results.md")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def replace_cell(text, pattern, value):
    """Replace first '-' in a table row matching pattern with value."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if pattern in line:
            # Replace the first ' - ' or '| - |' occurrence after the match
            lines[i] = line.replace(" - |", f" {value} |", 1)
            return "".join(lines)
    return text


def update_phase1(md, data, tag):
    """Fill Post-SFT column in Phase 1 tables."""
    hi = data.get("hindi", {})
    en = data.get("english", {})
    hi_acc = f"{hi.get('accuracy', 0):.1f}%"
    en_acc = f"{en.get('accuracy', 0):.1f}%"
    avg = (hi.get('accuracy', 0) + en.get('accuracy', 0)) / 2

    # Overall table
    md = replace_in_row(md, "Hindi", 2, hi_acc)
    md = replace_in_row(md, "English", 2, en_acc)
    md = replace_in_row(md, "**Average**", 2, f"**{avg:.1f}%**")

    # Delta
    hi_base = hi.get('accuracy', 0)
    en_base = en.get('accuracy', 0)
    # Can't compute delta without baseline here — just fill Post-SFT
    # Domain table — add a "Post-SFT" column section
    domain_section = "\n### Post-SFT — Per Domain\n\n"
    domain_section += "| Domain | Hindi Acc | English Acc |\n"
    domain_section += "|--------|-----------|--------------|\n"
    hi_dom = hi.get("per_domain", {})
    en_dom = en.get("per_domain", {})
    all_domains = sorted(set(list(hi_dom.keys()) + list(en_dom.keys())))
    for d in all_domains:
        ha = f"{hi_dom[d]['accuracy']:.1f}%" if d in hi_dom else "-"
        ea = f"{en_dom[d]['accuracy']:.1f}%" if d in en_dom else "-"
        domain_section += f"| {d} | {ha} | {ea} |\n"

    # Append domain section before Phase 2 header
    if "## Phase 2" in md:
        md = md.replace("---\n\n## Phase 2", domain_section + "\n---\n\n## Phase 2")
    else:
        md += domain_section
    return md


def update_phase2(md, data, tag):
    """Fill Post-Distill column in Phase 2 tables."""
    normad = data.get("normad", {})
    bhed = data.get("bhed", {})
    gop = data.get("globalopinion", {})

    normad_acc = normad.get("accuracy", 0)
    if normad_acc < 1:  # ratio
        normad_acc *= 100
    bhed_score = bhed.get("stereotype_score", 0)
    js_sim = gop.get("avg_js_similarity", 0)

    md = replace_in_row(md, "Accuracy | 69", 2, f"{normad_acc:.1f}%")
    md = replace_in_row(md, "**Overall**", 2, f"**{bhed_score:.1f}%**")
    md = replace_in_row(md, "JS Similarity", 2, f"{js_sim:.3f}")

    # Per-country post-distill section
    country_section = "\n### Post-Distill — Per Country\n\n"
    country_section += "| Country | Accuracy |\n"
    country_section += "|---------|----------|\n"
    for c, v in sorted(normad.get("per_country", {}).items()):
        acc = v.get("acc", 0)
        country_section += f"| {c.title()} | {acc:.1f}% |\n"

    bhed_section = "\n### Post-Distill — BhED by Category\n\n"
    bhed_section += "| Category | Stereotype Score |\n"
    bhed_section += "|----------|-----------------|\n"
    for cat, v in sorted(bhed.get("by_category", {}).items()):
        bhed_section += f"| {cat.title()} | {v.get('stereo_score', 0):.1f}% |\n"

    if "## Phase 3" in md:
        md = md.replace("---\n\n## Phase 3",
                        country_section + bhed_section + "\n---\n\n## Phase 3")
    else:
        md += country_section + bhed_section
    return md


def update_phase3(md, data, tag):
    """Fill Post-DPO column in Phase 3 table and add per-subset tables."""
    by_lang = data.get("by_language", {})
    avg_acc = data.get("avg_accuracy", 0)
    if avg_acc < 1:
        avg_acc *= 100

    for lang, v in by_lang.items():
        acc = v.get("accuracy", 0)
        if acc < 1:
            acc *= 100
        md = replace_in_row(md, f"| {lang.title()}", 2, f"{acc:.1f}%")
    md = replace_in_row(md, "**Average**", 2, f"**{avg_acc:.1f}%**")

    # Per-subset post-DPO
    subset_section = "\n### Post-DPO — Per Language Per Subset\n\n"
    for lang, v in by_lang.items():
        subset_section += f"\n**{lang.title()}** — {v.get('accuracy', 0)*100:.1f}% overall\n\n"
        subset_section += "| Subset | Correct/Total | Accuracy |\n"
        subset_section += "|--------|--------------|----------|\n"
        for s, sv in sorted(v.get("per_subset", {}).items()):
            subset_section += f"| {s.title()} | {sv['c']}/{sv['t']} | {sv['acc']:.1f}% |\n"

    md += subset_section
    return md


def replace_in_row(md, row_contains, col_index, new_val):
    """Replace the col_index-th '-' cell in a markdown table row containing row_contains."""
    lines = md.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if row_contains in line and "|" in line:
            cells = line.split("|")
            dash_count = 0
            for j, cell in enumerate(cells):
                if cell.strip() == "-":
                    dash_count += 1
                    if dash_count == col_index:
                        cells[j] = f" {new_val} "
                        lines[i] = "|".join(cells)
                        return "".join(lines)
    return md


def main():
    if len(sys.argv) < 3:
        print("Usage: python update_results.py <phase1|phase2|phase3> <results_json>")
        sys.exit(1)

    phase = sys.argv[1]
    json_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    if not os.path.exists(RESULTS_MD):
        print(f"ERROR: {RESULTS_MD} not found")
        sys.exit(1)

    data = load_json(json_path)
    tag = data.get("tag", "post-training")

    with open(RESULTS_MD) as f:
        md = f.read()

    if phase == "phase1":
        md = update_phase1(md, data, tag)
    elif phase == "phase2":
        md = update_phase2(md, data, tag)
    elif phase == "phase3":
        md = update_phase3(md, data, tag)
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)

    with open(RESULTS_MD, "w") as f:
        f.write(md)
    print(f"Updated {RESULTS_MD} with {phase} {tag} results.")


if __name__ == "__main__":
    main()
