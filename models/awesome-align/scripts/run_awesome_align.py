#!/usr/bin/env python3
"""
Run awesome-align on test data and display alignments in readable format.
"""

import json
import os
import subprocess
import sys
import tempfile

def convert_json_to_awesome_format(json_file, output_file, max_sentences=None):
    """Convert JSON file to awesome-align input format."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit number of sentences if specified
    if max_sentences is not None and max_sentences > 0:
        data = data[:max_sentences]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            ar = entry['ar']
            en = entry['en']
            f.write(f'{ar} ||| {en}\n')
    
    return len(data)

def parse_alignment_line(line):
    """Parse a line of alignment output in Pharaoh format (i-j pairs)."""
    alignments = []
    if not line.strip():
        return alignments
    
    pairs = line.strip().split()
    for pair in pairs:
        if '-' in pair:
            try:
                src_idx, tgt_idx = map(int, pair.split('-'))
                alignments.append((src_idx, tgt_idx))
            except ValueError:
                continue
    return alignments

def format_alignments(ar_words, en_words, alignments):
    """Format alignments in readable format."""
    # Group target words by source word
    src_to_tgt = {}
    for src_idx, tgt_idx in alignments:
        if src_idx < len(ar_words) and tgt_idx < len(en_words):
            if src_idx not in src_to_tgt:
                src_to_tgt[src_idx] = []
            src_to_tgt[src_idx].append(tgt_idx)
    
    # Format as (source, [targets])
    formatted = []
    for src_idx in sorted(src_to_tgt.keys()):
        tgt_indices = sorted(src_to_tgt[src_idx])
        src_word = ar_words[src_idx]
        tgt_words = [en_words[i] for i in tgt_indices]
        
        if len(tgt_words) == 1:
            formatted.append(f'({src_word}, {tgt_words[0]})')
        else:
            formatted.append(f'({src_word}, [{", ".join(tgt_words)}])')
    
    return ', '.join(formatted)

def get_structured_alignments(ar_words, en_words, alignments):
    """Get alignments as structured data (list of dicts)."""
    # Group target words by source word
    src_to_tgt = {}
    for src_idx, tgt_idx in alignments:
        if src_idx < len(ar_words) and tgt_idx < len(en_words):
            if src_idx not in src_to_tgt:
                src_to_tgt[src_idx] = []
            src_to_tgt[src_idx].append(tgt_idx)
    
    # Create structured format
    structured = []
    for src_idx in sorted(src_to_tgt.keys()):
        tgt_indices = sorted(src_to_tgt[src_idx])
        src_word = ar_words[src_idx]
        tgt_words = [en_words[i] for i in tgt_indices]
        
        structured.append({
            'ar_word': src_word,
            'ar_index': src_idx,
            'en_words': tgt_words,
            'en_indices': tgt_indices
        })
    
    return structured

def main():
    import argparse
    
    # Get script directory for relative path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '../../..'))
    awesome_align_dir = os.path.normpath(os.path.join(script_dir, '..'))
    
    # Default paths (release structure)
    default_input = os.path.normpath(os.path.join(project_root, 'data/BASELINE_A/test.json'))
    default_output = os.path.normpath(os.path.join(project_root, 'results/system_alignments/manual_run'))
    default_cache = os.path.normpath(os.path.join(awesome_align_dir, 'cache'))
    default_model = os.path.normpath(os.path.join(awesome_align_dir, 'fine_tuned_a'))
    
    parser = argparse.ArgumentParser(description='Run awesome-align on test data')
    parser.add_argument('--input', type=str, default=default_input,
                       help='Input JSON file (default: ../../data/BASELINE_A/test.json)')
    parser.add_argument('--model', type=str, default=default_model,
                       help='Model name or path (default: ../fine_tuned_a)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--output-dir', type=str, default=default_output,
                       help='Output directory for alignment files (default: ../../results/system_alignments/manual_run)')
    parser.add_argument('--num-examples', type=int, default=10,
                       help='Number of example alignments to display (default: 10)')
    parser.add_argument('--max-sentences', type=int, default=None,
                       help='Maximum number of sentences to process (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0, set to -1 for CPU)')
    parser.add_argument('--cache-dir', type=str, default=default_cache,
                       help=f'Directory to cache models (default: {default_cache})')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep the temporary input file after processing (default: False)')
    
    args = parser.parse_args()
    
    # Resolve relative paths to absolute paths
    if not os.path.isabs(args.input):
        args.input = os.path.abspath(os.path.join(os.getcwd(), args.input))
    else:
        args.input = os.path.normpath(args.input)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(os.getcwd(), args.output_dir))
    else:
        args.output_dir = os.path.normpath(args.output_dir)
    
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.abspath(os.path.join(os.getcwd(), args.cache_dir))
    else:
        args.cache_dir = os.path.normpath(args.cache_dir)
    
    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print('=' * 80)
    print('Running awesome-align on test data')
    print('=' * 80)
    
    # Convert JSON to awesome-align format
    print(f'\nConverting {args.input} to awesome-align format...')
    if args.max_sentences:
        print(f'  Limiting to {args.max_sentences} sentences')
    # Save temp file in output directory for easier access
    temp_input = os.path.join(args.output_dir, 'temp_input.src-tgt')
    os.makedirs(args.output_dir, exist_ok=True)
    num_sentences = convert_json_to_awesome_format(args.input, temp_input, args.max_sentences)
    print(f'  Converted {num_sentences} sentence pairs')
    print(f'  Temporary input file: {temp_input}')
    
    # Output files
    output_file = os.path.join(args.output_dir, 'alignments.out')
    prob_file = os.path.join(args.output_dir, 'alignments.prob')
    word_file = os.path.join(args.output_dir, 'alignments.words')
    json_file = os.path.join(args.output_dir, 'alignments.json')
    
    # Run awesome-align
    gpu_msg = f'GPU {args.gpu}' if args.gpu >= 0 else 'CPU'
    print(f'\nRunning awesome-align with model: {args.model} on {gpu_msg}...')
    
    # Set up environment for GPU and OpenBLAS
    env = os.environ.copy()
    if args.gpu >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    
    # Limit OpenBLAS threads to avoid resource limit issues
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    
    cmd = [
        'awesome-align',
        '--output_file', output_file,
        '--model_name_or_path', args.model,
        '--data_file', temp_input,
        '--extraction', 'softmax',
        '--batch_size', str(args.batch_size),
        '--output_prob_file', prob_file,
        '--output_word_file', word_file,
        '--cache_dir', args.cache_dir
    ]
    
    try:
        # Run with real-time output to show progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, env=env, bufsize=1, universal_newlines=True)
        
        # Show output in real-time
        for line in process.stdout:
            print(f'  {line.rstrip()}')
        
        process.wait()
        
        if process.returncode != 0:
            print(f'  Error: awesome-align exited with code {process.returncode}')
            sys.exit(1)
        
        print('  Alignment extraction completed')
    except subprocess.CalledProcessError as e:
        print(f'  Error running awesome-align: {e}')
        if hasattr(e, 'stderr') and e.stderr:
            print(f'  stderr: {e.stderr}')
            sys.exit(1)
    finally:
        # Clean up temp file unless --keep-temp is set
        if not args.keep_temp and os.path.exists(temp_input):
            os.remove(temp_input)
            print(f'  Cleaned up temporary file: {temp_input}')
        elif args.keep_temp and os.path.exists(temp_input):
            print(f'  Kept temporary file: {temp_input}')
    
    # Load original data and alignments
    print('\nLoading alignments...')
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit data to match processed sentences
    if args.max_sentences:
        data = data[:args.max_sentences]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        alignment_lines = f.readlines()
    
    # Process all alignments and create JSON output
    print('\nProcessing alignments and creating JSON output...')
    json_output = []
    
    for i in range(len(data)):
        entry = data[i]
        ar_words = entry['ar'].split()
        en_words = entry['en'].split()
        
        if i < len(alignment_lines):
            alignments = parse_alignment_line(alignment_lines[i])
            formatted = format_alignments(ar_words, en_words, alignments)
            structured = get_structured_alignments(ar_words, en_words, alignments)
            
            json_entry = {
                'id': entry['id'],
                'ar': entry['ar'],
                'en': entry['en'],
                'alignment_string': formatted,
                'alignments': structured
            }
            json_output.append(json_entry)
        else:
            json_entry = {
                'id': entry['id'],
                'ar': entry['ar'],
                'en': entry['en'],
                'alignment_string': '',
                'alignments': []
            }
            json_output.append(json_entry)
    
    # Save JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f'  Saved JSON with {len(json_output)} entries')
    
    # Display example alignments
    print(f'\nDisplaying first {args.num_examples} alignments:')
    print('=' * 80)
    
    for i in range(min(args.num_examples, len(json_output))):
        entry = json_output[i]
        print(f'\nEntry {i+1} (ID: {entry["id"]}):')
        print(f'  AR: {entry["ar"]}')
        print(f'  EN: {entry["en"]}')
        print(f'  Alignment: {entry["alignment_string"]}')
    
    print('\n' + '=' * 80)
    print(f'Alignment files saved to: {args.output_dir}')
    print(f'  - alignments.out: Pharaoh format alignments')
    print(f'  - alignments.prob: Alignment probabilities')
    print(f'  - alignments.words: Word pairs')
    print(f'  - alignments.json: JSON format with all alignments')
    print('=' * 80)

if __name__ == '__main__':
    main()

