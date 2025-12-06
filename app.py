#!/usr/bin/env python3
"""
Flask dashboard for visualizing word alignments and AER metrics.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import glob

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Default data paths (use bundled predicted_alignments)
DEFAULT_RESULTS_DIR = BASE_DIR / 'predicted_alignments' / 'results'
DEFAULT_ALIGNMENTS_DIR = BASE_DIR / 'predicted_alignments'
DEFAULT_GOLD_DIR = BASE_DIR / 'predicted_alignments' / 'gold'

# System definitions
SYSTEMS = {
    'baseline': {
        'wo_ft_a': {
            'results_file': 'wo_ft_a_aer_per_sentence.jsonl',
            'alignments_file': 'baseline/alignments_wo_ft.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Baseline (wo_ft_a)'
        },
        'ft_wo_dev_a': {
            'results_file': 'ft_wo_dev_a_aer_per_sentence.jsonl',
            'alignments_file': 'baseline/alignments_ft_wo_dev.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Baseline (ft_wo_dev_a)'
        }
    },
    'ner': {
        'ner_4.1': {
            'results_file': 'ner_4.1_restored_aer_per_sentence.jsonl',
            'alignments_file': 'ner_methods/split_d4.1_join_multiword_alignments.json',
            'gold_file': 'gold_test_ner.json',
            'display_name': 'NER 4.1 (join_multiword)'
        },
        'ner_4.2': {
            'results_file': 'ner_4.2_restored_aer_per_sentence.jsonl',
            'alignments_file': 'ner_methods/split_d4.2_transliteration_matching_alignments.json',
            'gold_file': 'gold_test_ner.json',
            'display_name': 'NER 4.2 (transliteration_matching)'
        },
        'ner_4.3': {
            'results_file': 'ner_4.3_restored_aer_per_sentence.jsonl',
            'alignments_file': 'ner_methods/split_d4.3_entity_placeholder_alignments.json',
            'gold_file': 'gold_test_ner.json',
            'display_name': 'NER 4.3 (entity_placeholder)'
        },
        'ner_4.4': {
            'results_file': 'ner_4.4_restored_aer_per_sentence.jsonl',
            'alignments_file': 'ner_methods/split_d4.4_entity_tagging_alignments.json',
            'gold_file': 'gold_test_ner.json',
            'display_name': 'NER 4.4 (entity_tagging)'
        }
    },
    'seg_bert': {
        'seg_bert_b1': {
            'results_file': 'seg_bert_b1_aer_per_sentence.jsonl',
            'alignments_file': 'segmented_bert_an/split_b1_alignments_agg.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Segmented BERT b1'
        },
        'seg_bert_b2': {
            'results_file': 'seg_bert_b2_aer_per_sentence.jsonl',
            'alignments_file': 'segmented_bert_an/split_b2_alignments_agg.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Segmented BERT b2'
        },
        'seg_bert_b3': {
            'results_file': 'seg_bert_b3_aer_per_sentence.jsonl',
            'alignments_file': 'segmented_bert_an/split_b3_alignments_agg.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Segmented BERT b3'
        },
        'seg_bert_b4': {
            'results_file': 'seg_bert_b4_aer_per_sentence.jsonl',
            'alignments_file': 'segmented_bert_an/split_b4_alignments_agg.json',
            'gold_file': 'gold_test.json',
            'display_name': 'Segmented BERT b4'
        }
    }
}

def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_gold_alignments(gold_data):
    """Extract gold alignment indices from gold data."""
    gold_alignments = []
    if 'alignment_string' in gold_data:
        # Parse alignment_string to get indices
        ar_tokens = gold_data.get('ar', '').split()
        en_tokens = gold_data.get('en', '').split()
        
        # Simple parsing - find word positions
        alignment_str = gold_data['alignment_string']
        import re
        # Match patterns like (word, word) or (word, [word1, word2])
        pairs = re.findall(r'\(([^,]+),\s*([^)]+)\)', alignment_str)
        for ar_seg, en_seg in pairs:
            ar_seg = ar_seg.strip()
            en_seg = en_seg.strip()
            
            # Handle brackets in en_seg
            if en_seg.startswith('[') and en_seg.endswith(']'):
                en_words = [w.strip() for w in en_seg[1:-1].split(',')]
            else:
                en_words = [en_seg]
            
            # Find indices
            if ar_seg in ar_tokens:
                ar_idx = ar_tokens.index(ar_seg)
                for en_word in en_words:
                    if en_word in en_tokens:
                        en_idx = en_tokens.index(en_word)
                        gold_alignments.append({
                            'ar_index': ar_idx,
                            'en_index': en_idx,
                            'ar_word': ar_seg,
                            'en_word': en_word
                        })
    return gold_alignments

def get_sentence_data(sentence_id, results_dir, alignments_dir, gold_dir):
    """Get all data for a specific sentence."""
    sentence_data = {
        'id': sentence_id,
        'systems': {}
    }
    
    # Store gold data for each category
    gold_data_cache = {}
    
    # Load data for each system
    for category, systems in SYSTEMS.items():
        for sys_key, sys_config in systems.items():
            results_path = results_dir / sys_config['results_file']
            alignments_path = alignments_dir / sys_config['alignments_file']
            gold_path = gold_dir / sys_config['gold_file']
            
            if not results_path.exists() or not alignments_path.exists() or not gold_path.exists():
                continue
            
            # Load gold data for this system
            if category not in gold_data_cache:
                gold_all = load_json(gold_path)
                gold_entry = next((item for item in gold_all if item['id'] == sentence_id), None)
                if gold_entry:
                    gold_alignments = extract_gold_alignments(gold_entry)
                    gold_data_cache[category] = {
                        'ar': gold_entry.get('ar', ''),
                        'en': gold_entry.get('en', ''),
                        'ar_tokens': gold_entry.get('ar', '').split(),
                        'en_tokens': gold_entry.get('en', '').split(),
                        'alignments': gold_alignments
                    }
                else:
                    # Skip this system if gold entry not found
                    continue
            
            gold_data = gold_data_cache.get(category)
            if not gold_data:
                continue
            
            # Load per-sentence results
            results_data = load_jsonl(results_path)
            result_entry = next((item for item in results_data if item['id'] == sentence_id), None)
            
            if not result_entry:
                continue
            
            # Load alignment data for original sentences
            alignments_data = load_json(alignments_path)
            alignment_entry = next((item for item in alignments_data if item['id'] == sentence_id), None)
            
            if not alignment_entry:
                continue
            
            # Extract alignment indices from alignment_entry
            predicted_alignments = []
            if 'alignments' in alignment_entry:
                for align in alignment_entry['alignments']:
                    ar_idx = align.get('ar_index', -1)
                    for en_idx, en_word in zip(align.get('en_indices', []), align.get('en_words', [])):
                        predicted_alignments.append({
                            'ar_index': ar_idx,
                            'en_index': en_idx,
                            'ar_word': align.get('ar_word', ''),
                            'en_word': en_word
                        })
            
            sentence_data['systems'][sys_key] = {
                'display_name': sys_config['display_name'],
                'category': category,
                'metrics': {
                    'AER': result_entry.get('AER', 0),
                    'precision': result_entry.get('precision', 0),
                    'recall': result_entry.get('recall', 0),
                    'f1': 2 * (result_entry.get('precision', 0) * result_entry.get('recall', 0)) / 
                          (result_entry.get('precision', 0) + result_entry.get('recall', 0)) 
                          if (result_entry.get('precision', 0) + result_entry.get('recall', 0)) > 0 else 0
                },
                'ar': alignment_entry.get('ar', ''),
                'en': alignment_entry.get('en', ''),
                'ar_tokens': alignment_entry.get('ar', '').split(),
                'en_tokens': alignment_entry.get('en', '').split(),
                'gold_pairs': result_entry.get('gold_pairs', []),
                'predicted_pairs': result_entry.get('predicted_pairs', []),
                'predicted_alignments': predicted_alignments,
                'gold': gold_data  # Include gold data for this system
            }
    
    # Set main gold data (use first available, or create empty if none found)
    if gold_data_cache:
        sentence_data['gold'] = list(gold_data_cache.values())[0]
    else:
        # Return None only if no systems have data at all
        if not sentence_data['systems']:
            return None
        # Otherwise, use empty gold data
        sentence_data['gold'] = {
            'ar': '',
            'en': '',
            'ar_tokens': [],
            'en_tokens': [],
            'alignments': []
        }
    
    return sentence_data
    
    return sentence_data

@app.route('/')
def index():
    """Homepage with file upload."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard."""
    return render_template('dashboard.html')

@app.route('/api/sentences')
def get_sentences():
    """Get list of all sentence IDs."""
    results_dir = Path(request.args.get('results_dir', DEFAULT_RESULTS_DIR))
    
    # Try to get sentence IDs from first available results file
    sentence_ids = set()
    for category, systems in SYSTEMS.items():
        for sys_key, sys_config in systems.items():
            results_path = results_dir / sys_config['results_file']
            if results_path.exists():
                results_data = load_jsonl(results_path)
                sentence_ids.update(item['id'] for item in results_data)
                break
        if sentence_ids:
            break
    
    return jsonify({'sentence_ids': sorted(list(sentence_ids))})

@app.route('/api/sentence/<sentence_id>')
def get_sentence(sentence_id):
    """Get data for a specific sentence."""
    results_dir = Path(request.args.get('results_dir', DEFAULT_RESULTS_DIR))
    alignments_dir = Path(request.args.get('alignments_dir', DEFAULT_ALIGNMENTS_DIR))
    gold_dir = Path(request.args.get('gold_dir', DEFAULT_GOLD_DIR))
    
    try:
        sentence_data = get_sentence_data(sentence_id, results_dir, alignments_dir, gold_dir)
        
        if not sentence_data:
            # Try to find which files have this sentence
            found_in = []
            for category, systems in SYSTEMS.items():
                for sys_key, sys_config in systems.items():
                    results_path = results_dir / sys_config['results_file']
                    if results_path.exists():
                        results_data = load_jsonl(results_path)
                        if any(item['id'] == sentence_id for item in results_data):
                            found_in.append(sys_config['results_file'])
            
            error_msg = f'Sentence {sentence_id} not found'
            if found_in:
                error_msg += f'. Found in: {", ".join(found_in)} but missing gold data.'
            return jsonify({'error': error_msg}), 404
        
        return jsonify(sentence_data)
    except Exception as e:
        import traceback
        return jsonify({'error': f'Error loading sentence: {str(e)}', 'traceback': traceback.format_exc()}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    if 'results_dir' not in request.files:
        return jsonify({'error': 'No results directory uploaded'}), 400
    
    # For now, just return success - in production, you'd save files
    return jsonify({'message': 'Files uploaded successfully', 'upload_dir': app.config['UPLOAD_FOLDER']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

