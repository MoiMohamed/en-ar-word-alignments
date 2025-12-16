#!/usr/bin/env python3
"""
Complete pipeline to process raw JSON files into train/test/dev splits.

Pipeline steps:
1. Load raw JSON files (aralects.json and madar.json)
2. Tokenize (simple_tokenize)
3. Normalize numbers (normalize_numbers)
4. Apply camel_tools normalization (normalize_alef_ar and dediac_ar) - optional
5. Sort by sentence length
6. Create stratified train/dev/test splits
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict

from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.dediac import dediac_ar
CAMEL_TOOLS_AVAILABLE = True


# Import preprocessing functions
SIMPLE_TOKENIZE_RE = re.compile(r"[\w\u064b-\u065f\u0670']+|[^\w\s']")
ARABIC_INDIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def simple_tokenize(input_str: str) -> str:
    """Perform a simple (whitespace and punctuation) tokenization of a given string."""
    input_norm = unicodedata.normalize('NFKC', input_str)
    return ' '.join(SIMPLE_TOKENIZE_RE.findall(input_norm))


def normalize_numbers(input_str: str) -> str:
    """Normalize digits in a given string - convert Arabic-Indic numerals to Western numerals."""
    input_norm = unicodedata.normalize('NFKC', input_str)
    return input_norm.translate(ARABIC_INDIC_TO_WESTERN)


def get_word_count(entry):
    """Get word count of Arabic sentence."""
    return len(entry['ar'].split())


def stratified_split(data, test_size, dev_size):
    """Create stratified split based on sentence length (word count)."""
    # Group by exact word count
    by_length = defaultdict(list)
    for entry in data:
        word_count = get_word_count(entry)
        by_length[word_count].append(entry)
    
    total = len(data)
    test_prop = test_size / total
    dev_prop = dev_size / total
    
    test_samples = []
    dev_samples = []
    train_samples = []
    
    # First, handle very long sentences (30+) separately to ensure they're distributed
    very_long_entries = []
    regular_entries_by_length = defaultdict(list)
    
    for word_count in sorted(by_length.keys()):
        if word_count >= 30:
            very_long_entries.extend(by_length[word_count])
        else:
            regular_entries_by_length[word_count] = by_length[word_count]
    
    # Distribute very long sentences proportionally
    if len(very_long_entries) > 0:
        random.shuffle(very_long_entries)
        n_test_long = max(1, round(len(very_long_entries) * test_prop))
        n_dev_long = max(1, round(len(very_long_entries) * dev_prop))
        n_train_long = len(very_long_entries) - n_test_long - n_dev_long
        
        n_test_long = min(n_test_long, len(very_long_entries))
        remaining = len(very_long_entries) - n_test_long
        n_dev_long = min(n_dev_long, remaining)
        n_train_long = remaining - n_dev_long
        
        test_samples.extend(very_long_entries[:n_test_long])
        dev_samples.extend(very_long_entries[n_test_long:n_test_long+n_dev_long])
        train_samples.extend(very_long_entries[n_test_long+n_dev_long:])
    
    # Process regular sentences (2-29 words)
    for word_count in sorted(regular_entries_by_length.keys()):
        entries = regular_entries_by_length[word_count]
        random.shuffle(entries)
        
        bin_size = len(entries)
        n_test = round(bin_size * test_prop)
        n_dev = round(bin_size * dev_prop)
        n_train = bin_size - n_test - n_dev
        
        n_test = max(0, min(n_test, bin_size))
        remaining = bin_size - n_test
        n_dev = max(0, min(n_dev, remaining))
        n_train = remaining - n_dev
        
        test_samples.extend(entries[:n_test])
        dev_samples.extend(entries[n_test:n_test+n_dev])
        train_samples.extend(entries[n_test+n_dev:])
    
    # Balance to exact target numbers
    test_needed = test_size - len(test_samples)
    dev_needed = dev_size - len(dev_samples)
    
    def get_by_length(entries_list):
        result = defaultdict(list)
        for e in entries_list:
            result[get_word_count(e)].append(e)
        return result
    
    # Balance test
    if test_needed > 0:
        train_by_len = get_by_length(train_samples)
        for length in sorted(train_by_len.keys()):
            if test_needed <= 0:
                break
            if length in train_by_len:
                take = min(test_needed, len(train_by_len[length]))
                for _ in range(take):
                    if train_by_len[length]:
                        entry = train_by_len[length].pop(0)
                        test_samples.append(entry)
                        train_samples.remove(entry)
                        test_needed -= 1
    
    # Balance dev
    if dev_needed > 0:
        train_by_len = get_by_length(train_samples)
        for length in sorted(train_by_len.keys()):
            if dev_needed <= 0:
                break
            if length in train_by_len:
                take = min(dev_needed, len(train_by_len[length]))
                for _ in range(take):
                    if train_by_len[length]:
                        entry = train_by_len[length].pop(0)
                        dev_samples.append(entry)
                        train_samples.remove(entry)
                        dev_needed -= 1
    
    # Trim if too many
    if len(test_samples) > test_size:
        excess = len(test_samples) - test_size
        train_samples.extend(test_samples[-excess:])
        test_samples = test_samples[:test_size]
    
    if len(dev_samples) > dev_size:
        excess = len(dev_samples) - dev_size
        train_samples.extend(dev_samples[-excess:])
        dev_samples = dev_samples[:dev_size]
    
    return test_samples, dev_samples, train_samples


def process_file(input_file, output_prefix, use_camel_tools=False):
    """Process a single JSON file through the pipeline."""
    print(f'\nProcessing {input_file}...')
    
    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'  Loaded {len(data)} entries')
    
    # Step 1: Tokenize
    print('  Step 1: Tokenizing...')
    tokenized = []
    for entry in data:
        processed_entry = {}
        if 'ar' in entry:
            processed_entry['ar'] = simple_tokenize(entry['ar'])
        if 'en' in entry:
            processed_entry['en'] = simple_tokenize(entry['en'])
        if 'id' in entry:
            processed_entry['id'] = entry['id']
        if 'split' in entry:
            processed_entry['split'] = entry['split']
        tokenized.append(processed_entry)
    
    # Step 2: Normalize numbers
    print('  Step 2: Normalizing numbers...')
    normalized = []
    for entry in tokenized:
        processed_entry = {}
        if 'ar' in entry:
            processed_entry['ar'] = normalize_numbers(entry['ar'])
        if 'en' in entry:
            processed_entry['en'] = normalize_numbers(entry['en'])
        if 'id' in entry:
            processed_entry['id'] = entry['id']
        if 'split' in entry:
            processed_entry['split'] = entry['split']
        normalized.append(processed_entry)
    
    # Step 3: Apply camel_tools normalization
    if use_camel_tools:
        print('  Step 3: Applying camel_tools normalization...')
        camel_normalized = []
        for entry in normalized:
            processed_entry = entry.copy()
            if 'ar' in entry:
                processed_entry['ar'] = dediac_ar(normalize_alef_ar(entry['ar']))
            camel_normalized.append(processed_entry)
        normalized = camel_normalized
    
    return normalized


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Process raw JSON files into train/test/dev splits')
    parser.add_argument('--aralects', type=str, default='raw/aralects.json',
                       help='Path to aralects raw JSON file')
    parser.add_argument('--madar', type=str, default='raw/madar.json',
                       help='Path to madar raw JSON file')
    parser.add_argument('--output-dir', type=str, default='BASELINE_A',
                       help='Output directory for train/test/dev files')
    parser.add_argument('--test-size', type=int, default=400,
                       help='Test set size (default: 400)')
    parser.add_argument('--dev-size', type=int, default=400,
                       help='Dev set size (default: 400)')
    parser.add_argument('--aralects-test', type=int, default=200,
                       help='Number of aralects entries in test (default: 200)')
    parser.add_argument('--madar-test', type=int, default=200,
                       help='Number of madar entries in test (default: 200)')
    parser.add_argument('--aralects-dev', type=int, default=200,
                       help='Number of aralects entries in dev (default: 200)')
    parser.add_argument('--madar-dev', type=int, default=200,
                       help='Number of madar entries in dev (default: 200)')
    parser.add_argument('--use-camel-tools', action='store_true',
                       help='Use camel_tools for normalization (requires conda activate acl)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print('=' * 80)
    print('Complete Pipeline: Raw JSON -> Train/Test/Dev Splits')
    print('=' * 80)
    
    # Process aralects
    aralects_processed = process_file(
        args.aralects,
        'aralects',
        use_camel_tools=args.use_camel_tools
    )
    
    # Process madar
    madar_processed = process_file(
        args.madar,
        'madar',
        use_camel_tools=args.use_camel_tools
    )
    
    print(f'\nTotal processed entries: {len(aralects_processed)} aralects + {len(madar_processed)} madar')
    
    # Step 4: Sort by sentence length
    print('\nStep 4: Sorting by sentence length...')
    aralects_sorted = sorted(aralects_processed, key=get_word_count)
    madar_sorted = sorted(madar_processed, key=get_word_count)
    
    # Step 5: Create stratified splits
    print('\nStep 5: Creating stratified splits...')
    print(f'  Aralects: {args.aralects_test} test, {args.aralects_dev} dev, rest train')
    aralects_test, aralects_dev, aralects_train = stratified_split(
        aralects_sorted, args.aralects_test, args.aralects_dev
    )
    
    print(f'  Madar: {args.madar_test} test, {args.madar_dev} dev, rest train')
    madar_test, madar_dev, madar_train = stratified_split(
        madar_sorted, args.madar_test, args.madar_dev
    )
    
    # Combine and shuffle
    test_set = aralects_test + madar_test
    dev_set = aralects_dev + madar_dev
    train_set = aralects_train + madar_train
    
    random.shuffle(test_set)
    random.shuffle(dev_set)
    random.shuffle(train_set)
    
    print(f'\nCreated splits:')
    print(f'  Test: {len(test_set)} entries ({len(aralects_test)} aralects + {len(madar_test)} madar)')
    print(f'  Dev: {len(dev_set)} entries ({len(aralects_dev)} aralects + {len(madar_dev)} madar)')
    print(f'  Train: {len(train_set)} entries ({len(aralects_train)} aralects + {len(madar_train)} madar)')
    
    # Analyze distributions
    def analyze_distribution(name, data):
        lengths = [get_word_count(x) for x in data]
        return {
            'name': name,
            'count': len(data),
            'min': min(lengths),
            'max': max(lengths),
            'avg': sum(lengths) / len(lengths),
            'median': sorted(lengths)[len(lengths)//2]
        }
    
    test_stats = analyze_distribution('Test', test_set)
    dev_stats = analyze_distribution('Dev', dev_set)
    train_stats = analyze_distribution('Train', train_set)
    
    print('\nDistribution analysis:')
    for stats in [test_stats, dev_stats, train_stats]:
        print(f"  {stats['name']}: min={stats['min']}, max={stats['max']}, "
              f"avg={stats['avg']:.2f}, median={stats['median']}")
    
    # Remove 'split' field from all entries
    print('\nRemoving split field from entries...')
    for entry in test_set:
        entry.pop('split', None)
    for entry in dev_set:
        entry.pop('split', None)
    for entry in train_set:
        entry.pop('split', None)
    
    # Save splits
    print(f'\nSaving splits to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'dev.json'), 'w', encoding='utf-8') as f:
        json.dump(dev_set, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    
    print('  Saved: test.json, dev.json, train.json')
    print('\n' + '=' * 80)
    print('Pipeline complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()

