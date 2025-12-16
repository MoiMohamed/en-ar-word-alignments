English–MSA Alignment Release
=============================

This bundle collects the minimal assets needed to reproduce the alignment experiments described in *English–MSA Bitext Word Alignment for End-User Consumption*.

Contents
--------
- `models/awesome-align/` — fine-tuned Awesome-Align checkpoints (segmentation D1/D2/ATB/D4 and NER C1–C4), source, and helper scripts.
- `data/` — raw corpora (`raw/`), baseline split (`BASELINE_A/`), segmented sets (`SEGMENTATION/` split_D1/D2/ATB/D4), NER-processed sets (`NER/` split_C1–C4), gold (`GOLD/`), and preprocessing scripts (`scripts/`).
- `results/` — gold references (`gold/`), system alignments (`system_alignments/`), per-sentence metrics (`metrics_per_sentence/`), error analysis tables (`error_analysis/`), and evaluation scripts under `results/test_scripts/`.
- `visualization_dashboard/` — lightweight viewer for inspecting alignments in the browser.

Abstract
--------
Bitext word and phrase alignment is typically a backend process in machine translation and corpus construction. In this project, we investigate alignment models for English–Modern Standard Arabic (MSA) with a user-facing application: integration into Aralects, a gamified Arabic learning app. The app requires accurate, visually interpretable alignments to help learners map English and Arabic words. We evaluate Awesome-Align with an mBERT backbone in two settings: an off-the-shelf (zero-shot) baseline and a model fine-tuned on an English–MSA training split. To address systematic failure modes, we introduce (i) morphology-driven segmentation interventions using CAMeL Tools to reduce fertility mismatch, and (ii) named-entity (NER) interventions that merge multi-word entities into atomic tokens using Arabic and English NER taggers. We complement quantitative results with sentence-length analysis and a linguistic error taxonomy highlighting implicit subject encoding, multi-word expressions, function-word behavior, and high-fertility alignment failures.

Quick start
-----------
1) Environment (Python 3.10+ recommended):
```
python -m venv .venv
source .venv/bin/activate
pip install -r models/awesome-align/requirements.txt
pip install -r visualization_dashboard/requirements.txt
```

2) Evaluate existing alignments (examples):
```
# End-to-end AER CSV across systems
python results/test_scripts/compute_aer_report.py

# Per-sentence AER for one system
python results/test_scripts/compute_aer_per_sentence.py --gold data/GOLD/gold_test.json --sys results/system_alignments/BASELINE_A/alignments_baseline_ft_A2.json --output results/metrics_per_sentence/custom.jsonl
```

3) Explore alignments in the dashboard:
```
cd visualization_dashboard
python app.py
# open the printed local URL
```

Note on hardware
----------------
- All experiments and checkpoints in this release were trained/evaluated on NVIDIA A100 GPUs.

Data and preprocessing
----------------------
- `data/scripts/process_pipeline.py` normalizes/tokenizes the raw corpora and builds stratified splits.
- `data/scripts/segm_bert_intervention.py` runs CAMeL Tools BERT disambiguator to produce segmented variants (D1/D2/ATB/D4).
- `data/scripts/ner_intervention.py` applies the reversible NER interventions (C1–C4).
- Included corpora: `data/raw/aralects.json`, `data/raw/madar.json`.
- Baseline split: `data/BASELINE_A/{train,dev,test}.json`.
- Segmented datasets: `data/SEGMENTATION/split_{D1,D2,ATB,D4}/`.
- NER-processed datasets: `data/NER/split_{C1..C4}/`.

Models
------
- Segmentation models: `fine_tuned_seg_bert_d1`, `d2`, `d3` (ATB), `d4`
- NER models: `finetuned_ner_c1` … `c4`
- Fine-tuned baseline: `fine_tuned_a`
- Supporting code: `awesome_align/`, `run_align.py`, `run_train.py`, and the custom runners in `scripts/`.

Results
-------
- Gold references: `results/gold/`
- System outputs: `results/system_alignments/{BASELINE_A,NER,SEGMENTATION}/`
- Per-sentence metrics: `results/metrics_per_sentence/`
- Error analysis: `results/error_analysis/`
- Aggregation helper for segmented alignments: `results/system_alignments/SEGMENTATION/aggregate_alignments.py`
