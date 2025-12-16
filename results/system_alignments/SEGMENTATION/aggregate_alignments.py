#!/usr/bin/env python3
"""
Aggregate segmented Arabic tokens in both alignment_string AND alignments list.
Properly merges alignments for segmented tokens (e.g., يمكن + ني -> يمكنني).

Inputs: data/SEGMENTATION/split_{D1,D2,ATB,D4}/alignments_ft_wo_dev/alignments.json
Outputs: results/system_alignments/SEGMENTATION/split_{D1,D2,ATB,D4}_alignments_agg.json
"""

import json
from pathlib import Path
from collections import defaultdict


def fix_taa_marbuta(word):
    """Fix تاء مربوطة (ة) in the middle of words to تاء (ت)."""
    if len(word) <= 1:
        return word
    fixed = ""
    for i, char in enumerate(word):
        fixed += "ت" if (char == "ة" and i < len(word) - 1) else char
    return fixed


def build_segmentation_map(ar_tokens, ar_agg_tokens):
    """
    Build mapping from original token indices to aggregated token indices.
    Returns seg_map and cleaned aggregated tokens.
    """
    seg_map = {}
    agg_tokens_clean = []
    orig_idx = 0

    for agg_idx, agg_token in enumerate(ar_agg_tokens):
        if "+" in agg_token or "_" in agg_token:
            clean_token = fix_taa_marbuta(agg_token.replace("+", "").replace("_", ""))
            agg_tokens_clean.append(clean_token)
            parts = agg_token.split("+")
            num_segments = len([p for p in parts if p.strip() and p.strip() != "_"])
            for _ in range(num_segments):
                seg_map[orig_idx] = (agg_idx, clean_token)
                orig_idx += 1
        else:
            agg_tokens_clean.append(agg_token)
            seg_map[orig_idx] = (agg_idx, agg_token)
            orig_idx += 1

    return seg_map, agg_tokens_clean


def aggregate_alignments_list(alignments, seg_map):
    """Aggregate alignments list based on segmentation map."""
    agg_groups = defaultdict(lambda: {"ar_word": None, "ar_index": None, "en_words": [], "en_indices": []})

    for align in alignments:
        orig_idx = align["ar_index"]
        if orig_idx not in seg_map:
            continue
        agg_idx, agg_token = seg_map[orig_idx]

        if agg_groups[agg_idx]["ar_word"] is None:
            agg_groups[agg_idx]["ar_word"] = agg_token
            agg_groups[agg_idx]["ar_index"] = agg_idx

        agg_groups[agg_idx]["en_words"].extend(align.get("en_words", []))
        agg_groups[agg_idx]["en_indices"].extend(align.get("en_indices", []))

    agg_alignments = []
    for agg_idx in sorted(agg_groups.keys()):
        group = agg_groups[agg_idx]
        seen_words = set()
        seen_indices = set()
        dedup_words = []
        dedup_indices = []

        for word, idx in zip(group["en_words"], group["en_indices"]):
            if word not in seen_words:
                seen_words.add(word)
                dedup_words.append(word)
            if idx not in seen_indices:
                seen_indices.add(idx)
                dedup_indices.append(idx)

        agg_alignments.append(
            {
                "ar_word": group["ar_word"],
                "ar_index": group["ar_index"],
                "en_words": dedup_words,
                "en_indices": sorted(dedup_indices),
            }
        )

    return agg_alignments


def parse_alignment_string(alignment_str):
    """Parse alignment string into list of (arabic_part, english_part) tuples."""
    pairs = []
    depth = 0
    current = []
    buffer = ""
    seen_separator = False

    for char in alignment_str + ",":
        if char == "(":
            depth += 1
            if depth == 1:
                buffer = ""
                seen_separator = False
        elif char == ")":
            depth -= 1
            if depth == 0:
                current.append(buffer.strip())
                buffer = ""
                seen_separator = False
        elif char == "," and depth == 1 and not seen_separator:
            if buffer.strip():
                current.append(buffer.strip())
            buffer = ""
            seen_separator = True
        elif depth == 1:
            buffer += char
        elif char == "," and depth == 0:
            if len(current) == 2:
                pairs.append((current[0], current[1]))
            current = []

    return pairs


def parse_side(side_str):
    """Parse a side (Arabic or English) into list of tokens."""
    side_str = side_str.strip()
    if not side_str or side_str == "-":
        return []
    if side_str.startswith("[") and side_str.endswith("]"):
        inner = side_str[1:-1]
        tokens = [t.strip() for t in inner.split(",")]
        return [t for t in tokens if t]
    return [side_str]


def aggregate_alignment_string(alignment_str, ar_tokens, ar_agg_tokens):
    """Aggregate Arabic tokens in alignment string."""
    seg_map, agg_tokens_clean = build_segmentation_map(ar_tokens, ar_agg_tokens)
    pairs = parse_alignment_string(alignment_str)
    agg_groups = {}

    for ar_side, en_side in pairs:
        ar_tokens_list = parse_side(ar_side)
        en_tokens_list = parse_side(en_side)
        if not ar_tokens_list:
            continue
        for ar_tok in ar_tokens_list:
            found_idx = None
            for idx, orig_tok in enumerate(ar_tokens):
                if orig_tok == ar_tok and idx in seg_map:
                    found_idx = idx
                    break
            if found_idx is not None and found_idx in seg_map:
                agg_idx, agg_token = seg_map[found_idx]
                if agg_idx not in agg_groups:
                    agg_groups[agg_idx] = {"token": agg_token, "en_tokens": []}
                agg_groups[agg_idx]["en_tokens"].extend(en_tokens_list)

    agg_pairs = []
    for agg_idx in sorted(agg_groups.keys()):
        agg_token = agg_groups[agg_idx]["token"]
        en_list = agg_groups[agg_idx]["en_tokens"]
        seen = set()
        en_list_dedup = []
        for en_tok in en_list:
            if en_tok not in seen:
                seen.add(en_tok)
                en_list_dedup.append(en_tok)
        en_formatted = en_list_dedup[0] if len(en_list_dedup) == 1 else "[" + ", ".join(en_list_dedup) + "]"
        agg_pairs.append(f"({agg_token}, {en_formatted})")

    return " ".join(agg_pairs), agg_tokens_clean


def aggregate_split(split_name: str, data_dir: Path, output_dir: Path):
    """Aggregate alignments for a single split."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name}")
    print(f"{'='*60}")

    split_dir = data_dir / split_name
    alignments_file = split_dir / "alignments_ft_wo_dev" / "alignments.json"
    test_file = split_dir / "test.json"
    output_file = output_dir / f"{split_name}_alignments_agg.json"

    if not alignments_file.exists():
        print(f"❌ Alignments file not found: {alignments_file}")
        return False
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"Reading test data from: {test_file}")
    with open(test_file) as f:
        test_data = json.load(f)
    ar_agg_map = {item["id"]: item["ar_agg"] for item in test_data}
    print(f"  Loaded {len(ar_agg_map)} test entries")

    print(f"Reading alignments from: {alignments_file}")
    with open(alignments_file) as f:
        alignments_data = json.load(f)
    print(f"  Loaded {len(alignments_data)} alignment entries")

    aggregated_data = []
    warnings = 0

    for align_item in alignments_data:
        item_id = align_item["id"]
        ar_tokens = align_item["ar"].split()
        ar_agg_sent = ar_agg_map.get(item_id)

        if not ar_agg_sent:
            print(f"  ⚠️  Warning: No ar_agg for {item_id}, keeping original")
            aggregated_data.append(align_item)
            warnings += 1
            continue

        ar_agg_tokens = ar_agg_sent.split()
        seg_map, agg_tokens_clean = build_segmentation_map(ar_tokens, ar_agg_tokens)

        agg_alignment_str, _ = aggregate_alignment_string(
            align_item["alignment_string"],
            ar_tokens,
            ar_agg_tokens
        )

        agg_alignments_list = aggregate_alignments_list(
            align_item.get("alignments", []),
            seg_map
        )

        aggregated_data.append({
            "id": item_id,
            "ar": " ".join(agg_tokens_clean),
            "en": align_item["en"],
            "alignment_string": agg_alignment_str,
            "alignments": agg_alignments_list
        })

    print(f"Writing aggregated alignments to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(aggregated_data)} aggregated entries")
    if warnings > 0:
        print(f"⚠️  {warnings} warnings")

    return True


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "data" / "SEGMENTATION"
    output_dir = repo_root / "results" / "system_alignments" / "SEGMENTATION"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["split_D1", "split_D2", "split_ATB", "split_D4"]

    print("=" * 60)
    print("Aggregating alignments for all segmented splits")
    print("=" * 60)

    success_count = 0
    for split in splits:
        if aggregate_split(split, data_dir, output_dir):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: Successfully processed {success_count}/{len(splits)} splits")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


