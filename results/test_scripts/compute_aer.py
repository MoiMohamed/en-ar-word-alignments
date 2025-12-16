#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


Pair = Tuple[str, str]


PUNCT_CHARS = set(".,!?;:\"'[]()،؟«»-:…/\\")


def is_punct_token(token: str) -> bool:
    """Return True if token is purely punctuation (to be ignored for AER)."""
    cleaned = token.strip()
    if not cleaned:
        return False
    no_space = cleaned.replace(" ", "")
    return all(ch in PUNCT_CHARS for ch in no_space)


def parse_alignment_pairs(alignment: str) -> List[Pair]:
    """Parse alignment string into list of (ar, en) pairs.

    Robust to commas inside bracketed lists like [مدخنا, لست] or [a, nonsmoker].
    """
    pairs: List[Pair] = []
    i = 0
    n = len(alignment)
    while i < n:
        if alignment[i] != "(":
            i += 1
            continue

        # find matching ')'
        j = alignment.find(")", i + 1)
        if j == -1:
            raise ValueError(f"Unmatched '(' in alignment segment: {alignment[i:]}")

        content = alignment[i + 1 : j]

        # split on first comma at bracket depth 0
        bracket_depth = 0
        split_idx = None
        for k, ch in enumerate(content):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif ch == "," and bracket_depth == 0:
                split_idx = k
                break

        if split_idx is None:
            raise ValueError(f"Could not find top-level comma in pair: ({content})")

        ar = content[:split_idx].strip()
        en = content[split_idx + 1 :].strip()
        pairs.append((ar, en))

        i = j + 1

    return pairs


def parse_side_tokens(seg: str) -> List[str]:
    """Parse a side segment, handling brackets and single tokens."""
    seg = seg.strip()
    if seg == "-" or seg == "":
        return []
    if seg.startswith("[") and seg.endswith("]"):
        inner = seg[1:-1].strip()
        if not inner:
            return []
        return [t.strip() for t in inner.split(",") if t.strip()]
    return [seg]


def build_gold_links(entry: Dict) -> List[Pair]:
    """Build list of (ar_word, en_word) pairs from gold alignment_string.
    
    Returns a multiset (list) to preserve duplicates.
    Ignores punctuation tokens.
    """
    alignment = entry["alignment_string"]
    parsed_pairs = parse_alignment_pairs(alignment)

    word_pairs: List[Pair] = []
    
    for ar_seg, en_seg in parsed_pairs:
        ar_list = [t for t in parse_side_tokens(ar_seg) if not is_punct_token(t)]
        en_list = [t for t in parse_side_tokens(en_seg) if not is_punct_token(t)]

        # If this pair is only punctuation on one or both sides, skip it
        if not ar_list or not en_list:
            continue

        # Expand to individual (ar_word, en_word) tuples
        for ar_word in ar_list:
            for en_word in en_list:
                word_pairs.append((ar_word, en_word))

    return word_pairs


def build_system_links(entry: Dict) -> List[Pair]:
    """Build list of (ar_word, en_word) pairs from system's alignments field.
    
    Returns a multiset (list) to preserve duplicates.
    Ignores punctuation tokens.
    """
    pairs: List[Pair] = []
    
    for align_item in entry.get("alignments", []):
        ar_word = align_item["ar_word"]
        
        # Skip if Arabic word is punctuation
        if is_punct_token(ar_word):
            continue
        
        # Expand each English word alignment
        for en_word in align_item["en_words"]:
            # Skip if English word is punctuation
            if is_punct_token(en_word):
                continue
            
            pairs.append((ar_word, en_word))
    
    return pairs


def compute_aer_subset(
    gold_by_id: Dict[str, Dict],
    sys_by_id: Dict[str, Dict],
    ids: List[str],
) -> Tuple[float, float, float, float]:
    """Compute AER, precision, recall, F1 over a subset of ids.
    
    Treats alignments as multisets (counts duplicates).
    """
    total_A = 0
    total_S = 0
    total_OK = 0

    for sent_id in ids:
        gold_entry = gold_by_id[sent_id]
        sys_entry = sys_by_id[sent_id]

        S = build_gold_links(gold_entry)  # gold pairs (multiset)
        A = build_system_links(sys_entry)  # system pairs (multiset)

        # Count intersection (multiset intersection)
        S_counts = {}
        for pair in S:
            S_counts[pair] = S_counts.get(pair, 0) + 1
        
        A_counts = {}
        for pair in A:
            A_counts[pair] = A_counts.get(pair, 0) + 1
        
        # Intersection count
        h = 0
        for pair, count in A_counts.items():
            if pair in S_counts:
                h += min(count, S_counts[pair])
        
        total_A += len(A)
        total_S += len(S)
        total_OK += h

    if total_A == 0 or total_S == 0:
        precision = 0.0
        recall = 0.0
        aer = 1.0
        f1 = 0.0
    else:
        precision = total_OK / total_A
        recall = total_OK / total_S
        # AER = 1 - (2 * |A ∩ S|) / (|A| + |S|)
        aer = 1.0 - (2.0 * total_OK) / (total_A + total_S)
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

    return aer, precision, recall, f1


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_gold = repo_root / "data" / "GOLD" / "gold_test.json"
    default_sys = repo_root / "results" / "system_alignments" / "baseline" / "alignments_ft_wo_dev.json"

    parser = argparse.ArgumentParser(description="Compute AER between gold and system alignments.")
    parser.add_argument(
        "--gold",
        type=str,
        default=str(default_gold),
        help=f"Path to gold JSON file (default: {default_gold})",
    )
    parser.add_argument(
        "--sys",
        type=str,
        default=str(default_sys),
        help=f"Path to system JSON file (default: {default_sys})",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    sys_path = Path(args.sys)

    with gold_path.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with sys_path.open("r", encoding="utf-8") as f:
        sys_data = json.load(f)

    gold_by_id: Dict[str, Dict] = {item["id"]: item for item in gold_data}
    sys_by_id: Dict[str, Dict] = {item["id"]: item for item in sys_data}

    common_ids = sorted(set(gold_by_id.keys()) & set(sys_by_id.keys()))
    if not common_ids:
        raise ValueError("No common sentence ids between gold and system files.")

    subsets = {
        "a_only": [i for i in common_ids if i.startswith("a")],
        "m_only": [i for i in common_ids if i.startswith("m")],
        "a_and_m": [i for i in common_ids if i.startswith(("a", "m"))],
    }

    for name, ids in subsets.items():
        if not ids:
            print(f"{name}: no sentences in this subset")
            continue

        aer, prec, rec, f1 = compute_aer_subset(gold_by_id, sys_by_id, ids)
        print(
            f"{name}: N={len(ids)} "
            f"AER={aer:.4f}  precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
