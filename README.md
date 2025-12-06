# Word Alignment Dashboard

A web-based dashboard for visualizing word alignments and comparing AER metrics across different alignment systems.

## Features

- **Sentence Selection**: Browse and search through all sentences in the dataset
- **Alignment Visualization**: Visual representation of word alignments with color-coded lines:
  - Green: Correct alignments (match gold)
  - Red: Incorrect alignments
  - Orange (dashed): Gold-only alignments (missed by system)
- **Metrics Comparison**: Side-by-side comparison of AER, Precision, Recall, and F1 scores
- **Multiple Systems**: Compare baseline, NER methods, and segmented BERT systems

## Installation

```bash
pip install -r requirements.txt
```

## Running the Dashboard

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Usage

1. **Homepage**: Choose to use default data or upload custom files
2. **Dashboard**: 
   - Select a sentence from the sidebar
   - View alignment visualizations for all systems
   - Compare metrics in the table

## Data Structure

Default data is bundled under `dashboard/predicted_alignments/`:
- Per-sentence AER results (JSONL) in `predicted_alignments/results/`
- Alignment files (JSON) in `predicted_alignments/`
- Gold standard files in `predicted_alignments/gold/`

## API Endpoints

- `GET /` - Homepage
- `GET /dashboard` - Main dashboard
- `GET /api/sentences` - Get list of all sentence IDs
- `GET /api/sentence/<sentence_id>` - Get data for a specific sentence






