#!/usr/bin/env python3
"""
Compute AER for multiple systems and generate CSV report.
"""

import json
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import functions from compute_aer.py
sys.path.insert(0, str(Path(__file__).parent))
from compute_aer import (
    build_gold_links,
    build_system_links,
    compute_aer_subset
)

def compute_aer_for_system(gold_file: str, sys_file: str) -> Dict[str, Dict[str, float]]:
    """Compute AER for a system and return results for all subsets."""
    gold_path = Path(gold_file)
    sys_path = Path(sys_file)
    
    with gold_path.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with sys_path.open("r", encoding="utf-8") as f:
        sys_data = json.load(f)
    
    gold_by_id: Dict[str, Dict] = {item["id"]: item for item in gold_data}
    sys_by_id: Dict[str, Dict] = {item["id"]: item for item in sys_data}
    
    common_ids = sorted(set(gold_by_id.keys()) & set(sys_by_id.keys()))
    if not common_ids:
        return {}
    
    subsets = {
        "a_only": [i for i in common_ids if i.startswith("a")],
        "m_only": [i for i in common_ids if i.startswith("m")],
        "a_and_m": [i for i in common_ids if i.startswith(("a", "m"))],
    }
    
    results = {}
    for name, ids in subsets.items():
        if not ids:
            continue
        
        aer, prec, rec, f1 = compute_aer_subset(gold_by_id, sys_by_id, ids)
        results[name] = {
            "N": len(ids),
            "AER": aer,
            "precision": prec,
            "recall": rec,
            "F1": f1
        }
    
    return results

def main():
    # Define all systems to evaluate (using predicted_alignments folder)
    systems = [
        {
            "name": "wo_ft_a",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/baseline/alignments_wo_ft.json"
        },
        {
            "name": "ft_wo_dev_a",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/baseline/alignments_ft_wo_dev.json"
        },
        {
            "name": "ner_4.1_restored",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test_ner.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/ner_methods/split_d4.1_join_multiword_alignments.json"
        },
        {
            "name": "ner_4.2_restored",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test_ner.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/ner_methods/split_d4.2_transliteration_matching_alignments.json"
        },
        {
            "name": "ner_4.3_restored",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test_ner.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/ner_methods/split_d4.3_entity_placeholder_alignments.json"
        },
        {
            "name": "ner_4.4_restored",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test_ner.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/ner_methods/split_d4.4_entity_tagging_alignments.json"
        },
        {
            "name": "seg_bert_b1",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/segmented_bert_an/split_b1_alignments_agg.json"
        },
        {
            "name": "seg_bert_b2",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/segmented_bert_an/split_b2_alignments_agg.json"
        },
        {
            "name": "seg_bert_b3",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/segmented_bert_an/split_b3_alignments_agg.json"
        },
        {
            "name": "seg_bert_b4",
            "gold": "/scratch/mom8702/ACL/predicted_alignments/gold/gold_test.json",
            "sys": "/scratch/mom8702/ACL/predicted_alignments/segmented_bert_an/split_b4_alignments_agg.json"
        },
    ]
    
    # Collect all results
    all_results = []
    
    for system in systems:
        print(f"Computing AER for {system['name']}...")
        try:
            results = compute_aer_for_system(system["gold"], system["sys"])
            
            for subset_name, metrics in results.items():
                all_results.append({
                    "system": system["name"],
                    "subset": subset_name,
                    "N": metrics["N"],
                    "AER": metrics["AER"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "F1": metrics["F1"]
                })
        except Exception as e:
            print(f"Error computing AER for {system['name']}: {e}")
            continue
    
    # Write CSV report
    output_file = "/scratch/mom8702/ACL/test/aer_report.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if not all_results:
            print("No results to write")
            return
        
        fieldnames = ["system", "subset", "N", "AER", "precision", "recall", "F1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Group by system for better readability
        current_system = None
        for row in all_results:
            if current_system != row["system"]:
                if current_system is not None:
                    # Add empty row as separator
                    writer.writerow({})
                current_system = row["system"]
            
            writer.writerow({
                "system": row["system"],
                "subset": row["subset"],
                "N": row["N"],
                "AER": f"{row['AER']:.4f}",
                "precision": f"{row['precision']:.4f}",
                "recall": f"{row['recall']:.4f}",
                "F1": f"{row['F1']:.4f}"
            })
    
    print(f"\nCSV report saved to: {output_file}")
    print(f"Total rows: {len(all_results)}")

if __name__ == "__main__":
    main()



