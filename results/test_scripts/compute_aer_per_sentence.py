#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


Pair = Tuple[str, str]


PUNCT_CHARS = set(".,!?;:\"'[]()،؟«»-:…")


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


def compute_sentence_aer(gold_entry: Dict, sys_entry: Dict) -> Dict:
    """Compute AER for a single sentence."""
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
    
    total_A = len(A)
    total_S = len(S)
    
    if total_A == 0 or total_S == 0:
        precision = 0.0
        recall = 0.0
        aer = 1.0
    else:
        precision = h / total_A
        recall = h / total_S
        aer = 1.0 - (2.0 * h) / (total_A + total_S)
    
    return {
        "id": gold_entry["id"],
        "gold_pairs": S,
        "predicted_pairs": A,
        "precision": precision,
        "recall": recall,
        "AER": aer
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_gold = repo_root / "data" / "GOLD" / "gold_test.json"

    parser = argparse.ArgumentParser(description="Compute per-sentence AER between gold and system alignments.")
    parser.add_argument(
        "--gold",
        type=str,
        default=str(default_gold),
        help=f"Path to gold JSON file (default: {default_gold})",
    )
    parser.add_argument(
        "--sys",
        type=str,
        required=True,
        help="Path to system JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    sys_path = Path(args.sys)
    output_path = Path(args.output)

    with gold_path.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with sys_path.open("r", encoding="utf-8") as f:
        sys_data = json.load(f)

    gold_by_id: Dict[str, Dict] = {item["id"]: item for item in gold_data}
    sys_by_id: Dict[str, Dict] = {item["id"]: item for item in sys_data}

    common_ids = sorted(set(gold_by_id.keys()) & set(sys_by_id.keys()))
    if not common_ids:
        raise ValueError("No common sentence ids between gold and system files.")

    results = []
    for sent_id in common_ids:
        result = compute_sentence_aer(gold_by_id[sent_id], sys_by_id[sent_id])
        results.append(result)

    # Write JSONL output
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Written {len(results)} per-sentence AER results to {output_path}")


if __name__ == "__main__":
    main()

