#!/usr/bin/env python3
"""
Segment Arabic data using different tokenization schemes with BERT disambiguator.

BERT version: More accurate but MUCH slower than MLE.
Recommended only for small datasets or when accuracy is critical.

Output format:
- ar: Space-separated morphemes (for alignment)
- ar_agg: Morphemes with + markers (for aggregation)
- en: English text
- id: Sentence ID

Example:
{
  "ar": "ف س يكتب ها",
  "ar_agg": "ف+ س+ يكتب +ها",
  "en": "and he will write it",
  "id": "12345"
}
"""

import argparse
import json
import os
import re
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.dediac import dediac_ar


def count_words_before_noan(text):
    """Count non-punctuation words before the first NOAN token."""
    words = text.split()
    count = 0
    for word in words:
        if 'NOAN' in word:
            break
        if word not in ['"', "'", '،', '.', '؟', '!', ';', ':', '(', ')', '[', ']']:
            word_clean = word.replace('+', '').replace('_', '')
            if len(word_clean) > 0:
                count += 1
    return count


def replace_noan_in_ar_agg(seg_ar_agg, orig_ar):
    """
    Replace NOAN in ar_agg with the corresponding word from the original text.
    Mirrors logic from replace_noan.py so users don't need a second script.
    """
    if 'NOAN' not in seg_ar_agg:
        return seg_ar_agg

    words_before = count_words_before_noan(seg_ar_agg)

    orig_words = []
    for w in orig_ar.split():
        if w not in ['"', "'", '،', '.', '؟', '!', ';', ':', '(', ')', '[', ']']:
            orig_words.append(w)

    if words_before < len(orig_words):
        replacement_word = orig_words[words_before]

        words = seg_ar_agg.split()
        result_words = []
        i = 0

        while i < len(words):
            word = words[i]
            if 'NOAN' in word:
                seg_word_parts = [word]
                j = i + 1
                while j < len(words) and (words[j].startswith('+') or words[j].startswith('_') or '+' in words[j] or '_' in words[j]):
                    seg_word_parts.append(words[j])
                    j += 1

                result_words.append(replacement_word)
                i = j
            else:
                result_words.append(word)
                i += 1

        seg_ar_agg = ' '.join(result_words)

    return seg_ar_agg


def ar_agg_to_ar(ar_agg):
    """Convert ar_agg with markers to a clean ar string."""
    words = ar_agg.split()
    ar_clean_words = []

    for word in words:
        ar_clean_word = word.replace('_+', '+').replace('+_', '+')
        ar_clean_word = ar_clean_word.replace('_', ' ').replace('+', ' ')
        ar_clean_word = ' '.join(ar_clean_word.split())
        ar_clean_words.append(ar_clean_word)

    return ' '.join(ar_clean_words)


class ArabicSegmenterBERT:
    """Arabic segmentation with multiple schemes using BERT disambiguator."""
    
    def __init__(self, device='cpu'):
        """
        Initialize BERT disambiguator.
        
        Args:
            device: 'cpu' or 'cuda' (GPU)
                   Use 'cuda' if you have GPU for faster processing
        """
        print(f"Loading CAMeL Tools BERT disambiguator (device={device})...")
        print("This may take 10-20 seconds...")
        use_gpu = device == 'cuda'
        self.disambiguator = BERTUnfactoredDisambiguator.pretrained(use_gpu=use_gpu)
        print("CAMeL Tools BERT loaded!")
    
    def _process_segmented(self, segmented_text):
        """
        Process segmented text to create both versions:
        - ar: Clean (spaces only)
        - ar_agg: With + markers
        
        Input: "ف+ س+ يكتب +ها" or "ف_+ س_+ يكتب _+ها"
        Returns: {
            'ar': 'ف س يكتب ها',
            'ar_agg': 'ف+ س+ يكتب +ها'
        }
        """
        # Apply normalization first
        normalized = normalize_alef_ar(segmented_text)
        dediac = dediac_ar(normalized)
        
        # Split by word boundaries (spaces) to process each word separately
        words = dediac.split()
        ar_agg_words = []
        ar_clean_words = []
        
        for word in words:
            ar_agg_words.append(word)
            
            # For ar: normalize _+ and +_ to +, then replace all + and _ with spaces
            ar_clean_word = word.replace('_+', '+').replace('+_', '+')
            ar_clean_word = ar_clean_word.replace('_', ' ').replace('+', ' ')
            ar_clean_word = ' '.join(ar_clean_word.split())  # Clean extra spaces
            ar_clean_words.append(ar_clean_word)
        
        # Join words with spaces
        ar_agg = ' '.join(ar_agg_words)
        ar_clean = ' '.join(ar_clean_words)
        
        return {
            'ar': ar_clean,
            'ar_agg': ar_agg
        }
    
    def segment_d1(self, text):
        """
        split_b1 (D1): Separate conjunctions only (و، ف)
        """
        words = text.split()
        segmented_words = []
        
        for word in words:
            disambig = self.disambiguator.disambiguate([word])
            if not disambig or not disambig[0] or not disambig[0].analyses:
                segmented_words.append(word)
                continue
                
            analysis = disambig[0].analyses[0].analysis
            d1_segmented = analysis.get('d1seg', word)
            segmented_words.append(d1_segmented)
        
        segmented_text = ' '.join(segmented_words)
        return self._process_segmented(segmented_text)
    
    def segment_d2(self, text):
        """
        split_b2 (D2): D1 + Separate prepositions and future particle (ب، ل، س)
        """
        words = text.split()
        segmented_words = []
        
        for word in words:
            disambig = self.disambiguator.disambiguate([word])
            if not disambig or not disambig[0] or not disambig[0].analyses:
                segmented_words.append(word)
                continue
                
            analysis = disambig[0].analyses[0].analysis
            d2_segmented = analysis.get('d2seg', word)
            segmented_words.append(d2_segmented)
        
        segmented_text = ' '.join(segmented_words)
        return self._process_segmented(segmented_text)
    
    def segment_atb(self, text):
        """
        split_b3 (ATB): D2 + Separate pronominal clitics
        """
        words = text.split()
        segmented_words = []
        
        for word in words:
            disambig = self.disambiguator.disambiguate([word])
            if not disambig or not disambig[0] or not disambig[0].analyses:
                segmented_words.append(word)
                continue
                
            analysis = disambig[0].analyses[0].analysis
            atb_segmented = analysis.get('atbseg', word)
            segmented_words.append(atb_segmented)
        
        segmented_text = ' '.join(segmented_words)
        return self._process_segmented(segmented_text)
    
    def segment_d3(self, text):
        """
        split_b4 (D3): Separate all clitics
        """
        words = text.split()
        segmented_words = []
        
        for word in words:
            disambig = self.disambiguator.disambiguate([word])
            if not disambig or not disambig[0] or not disambig[0].analyses:
                segmented_words.append(word)
                continue
                
            analysis = disambig[0].analyses[0].analysis
            d3_segmented = analysis.get('d3seg', word)
            segmented_words.append(d3_segmented)
        
        segmented_text = ' '.join(segmented_words)
        return self._process_segmented(segmented_text)
    
    def segment(self, text, scheme='d1'):
        """
        Main segmentation function
        
        Returns: dict with 'ar' and 'ar_agg' keys
        """
        if scheme == 'd1':
            return self.segment_d1(text)
        elif scheme == 'd2':
            return self.segment_d2(text)
        elif scheme == 'atb':
            return self.segment_atb(text)
        elif scheme == 'd3':
            return self.segment_d3(text)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")


def segment_dataset(input_file, output_file, scheme, segmenter):
    """Segment a dataset and save to output file."""
    print(f"  Segmenting {input_file} with scheme '{scheme}'...")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Segment Arabic text
    segmented_data = []
    for i, entry in enumerate(data):
        if i % 100 == 0:
            print(f"    Processing {i}/{len(data)}...")
        
        # Segment and get both versions
        if 'ar' in entry:
            seg_result = segmenter.segment(entry['ar'], scheme)

            # Replace NOAN tokens inline using original text
            ar_agg = replace_noan_in_ar_agg(seg_result['ar_agg'], entry['ar'])
            # Keep normalization consistent after replacement
            ar_agg = dediac_ar(normalize_alef_ar(ar_agg))
            ar_clean = ar_agg_to_ar(ar_agg)

            # Create new entry with both ar and ar_agg
            segmented_entry = {
                'ar': ar_clean,         # Clean: "ف س يكتب ها"
                'ar_agg': ar_agg,       # With +: "ف+ س+ يكتب +ها" (NOAN resolved)
                'en': entry['en']
            }
            
            # Preserve ID if exists
            if 'id' in entry:
                segmented_entry['id'] = entry['id']
        else:
            segmented_entry = entry.copy()
        
        segmented_data.append(segmented_entry)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Segment Arabic data with different schemes using BERT'
    )
    parser.add_argument('--input-dir', type=str, default='BASELINE_A',
                       help='Directory containing train.json, dev.json, test.json')
    parser.add_argument('--output-dir', type=str, default='SEGMENTATION',
                       help='Output directory for segmented files')
    parser.add_argument('--schemes', nargs='+', 
                       default=['d1', 'd2', 'atb', 'd3'],
                       help='Segmentation schemes: d1, d2, atb, d3 (default: all)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for BERT: cpu or cuda (default: cpu)')
    
    args = parser.parse_args()
    
    print('=' * 80)
    print('Segmenting Arabic Data with BERT Disambiguator')
    print('=' * 80)
    print(f'Device: {args.device}')
    if args.device == 'cpu':
        print('WARNING: BERT is VERY slow on CPU (10-100x slower than MLE)')
        print('         Consider using GPU (--device cuda) if available')
    print('=' * 80)
    
    # Initialize segmenter
    segmenter = ArabicSegmenterBERT(device=args.device)
    
    # Process each split (train, dev, test)
    splits = ['train', 'dev', 'test']
    
    # Map schemes to output directory names
    scheme_to_dir = {
        'd1': 'split_b1',
        'd2': 'split_b2',
        'atb': 'split_b3',
        'd3': 'split_b4'
    }
    
    for scheme in args.schemes:
        if scheme not in scheme_to_dir:
            print(f"  Warning: Unknown scheme '{scheme}', skipping...")
            continue
            
        output_dir_name = scheme_to_dir[scheme]
        print(f"\nScheme: {scheme} -> {output_dir_name}")
        print("-" * 80)
        
        # Create output directory for this scheme
        scheme_dir = os.path.join(args.output_dir, output_dir_name)
        os.makedirs(scheme_dir, exist_ok=True)
        
        for split in splits:
            input_file = os.path.join(args.input_dir, f'{split}.json')
            output_file = os.path.join(scheme_dir, f'{split}.json')
            
            if os.path.exists(input_file):
                segment_dataset(input_file, output_file, scheme, segmenter)
            else:
                print(f"  Warning: {input_file} not found, skipping...")
    
    print('\n' + '=' * 80)
    print('Segmentation complete!')
    print('=' * 80)
    
    # Print output structure
    print(f"\nOutput structure in '{args.output_dir}':")
    for scheme in args.schemes:
        if scheme in scheme_to_dir:
            dir_name = scheme_to_dir[scheme]
            print(f"  {dir_name}/")
            print(f"    ├── train.json  (with 'ar' and 'ar_agg' fields)")
            print(f"    ├── dev.json")
            print(f"    └── test.json")


if __name__ == '__main__':
    main()