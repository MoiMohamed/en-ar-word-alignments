#!/usr/bin/env python3
"""
Phase 1C: Apply NER methods with full reversibility.

All methods preserve:
- original_ar: Original Arabic text (before NER)
- original_en: Original English text (before NER)
- modifications: What was changed and how to reverse it

This allows reconstructing original format after alignment.
"""

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import pipeline
from camel_tools.utils.charmap import CharMapper


class TransformerNER:    
    def __init__(self, device=0):
        print(f"Loading transformer NER models on GPU {device}...")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("WARNING: No GPU detected!")
            device = -1
        
        self.ar_ner = pipeline(
            "ner",
            model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
            aggregation_strategy="simple",
            device=device
        )
        
        self.en_ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=device
        )
        
        # CAMeL Tools transliteration
        self.ar2bw = CharMapper.builtin_mapper('ar2bw')
        
        # Simple in-memory cache for extracted entities per (ar_text, en_text)
        self._entity_cache = {}
        
        print("Models loaded!")
    
    def extract_entities(self, ar_text, en_text):
        """Extract entities with token positions."""
        cache_key = (ar_text, en_text)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        ar_tokens = ar_text.split()
        en_tokens = en_text.split()
        
        # Arabic entities
        ar_results = self.ar_ner(ar_text) if ar_text.strip() else []
        ar_entities = []
        
        for ent in ar_results:
            entity_text = ent['word'].strip()
            entity_type = self._normalize_type(ent['entity_group'])
            token_positions = self._find_token_positions(entity_text, ar_tokens)
            
            if token_positions:
                ar_entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'token_positions': token_positions,
                    'tokens': [ar_tokens[i] for i in token_positions]
                })
        
        # English entities
        en_results = self.en_ner(en_text) if en_text.strip() else []
        en_entities = []
        
        for ent in en_results:
            entity_text = ent['word'].strip()
            entity_type = self._normalize_type(ent['entity_group'])
            token_positions = self._find_token_positions(entity_text, en_tokens)
            
            if token_positions:
                en_entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'token_positions': token_positions,
                    'tokens': [en_tokens[i] for i in token_positions]
                })
        
        return {
            'ar_entities': ar_entities,
            'en_entities': en_entities,
            'ar_tokens': ar_tokens,
            'en_tokens': en_tokens
        }
        # Cache result for this pair of texts
        self._entity_cache[cache_key] = result
        return result
    
    def _find_token_positions(self, entity_text, tokens):
        """Find which token positions an entity occupies."""
        entity_tokens = entity_text.split()
        
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                return list(range(i, i+len(entity_tokens)))
        
        # If we can't find an exact span match, give up (avoid noisy partial matches)
        return []
    
    def _normalize_type(self, entity_type):
        """Normalize entity types.
        
        CAMeL-BERT produces: 'LOC', 'MISC', 'ORG', 'PERS'
        English BERT produces: 'PER', 'PERSON', 'LOC', 'LOCATION', 'ORG', 'ORGANIZATION', 'MISC', 'GPE'
        
        Normalize all to: 'PER', 'LOC', 'ORG', 'MISC'
        """
        mapping = {
            'PER': 'PER', 'PERSON': 'PER', 'PERS': 'PER',  # CAMeL-BERT uses 'PERS'
            'LOC': 'LOC', 'LOCATION': 'LOC', 'GPE': 'LOC',
            'ORG': 'ORG', 'ORGANIZATION': 'ORG',
            'MISC': 'MISC',
        }
        return mapping.get(entity_type.upper(), 'MISC')
    
    def transliterate(self, arabic_text):
        """Transliterate Arabic to Latin using CAMeL Tools."""
        return self.ar2bw(arabic_text)


class NERMethods:
    """All 5 NER methods with full reversibility."""
    
    def __init__(self, ner_model):
        self.ner = ner_model
    
    # ========== D4.1: join_multiword - Simply join multi-word entities with underscore ==========
    def method_join_multiword(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.1: Simply join multi-word entities with underscore.
        
        This is the SIMPLEST NER method:
        - Detects entities in both languages
        - Joins multi-word entities with underscore
        - NO transliteration, NO placeholders, NO tags
        - Just makes multi-word entities into single tokens
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيو يورك مع سارة"
            EN: "Ahmad went to New York with Sarah"
        
        Output:
            AR: "أحمد ذهب إلى نيو_يورك مع سارة"
            EN: "Ahmad went to New_York with Sarah"
        
        Changes:
            - "نيو يورك" (2 tokens) → "نيو_يورك" (1 token)
            - "New York" (2 tokens) → "New_York" (1 token)
            - Single-word entities unchanged: "أحمد", "سارة", "Ahmad", "Sarah"
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        ar_skip = set()
        en_skip = set()
        
        modifications = {
            'method': 'join_multiword',
            'joined_entities': []
        }
        
        # Join Arabic multi-word entities
        for ar_ent in entities['ar_entities']:
            positions = ar_ent['token_positions']
            # Only join if the entity covers 2+ tokens AND they are contiguous
            if len(positions) > 1 and all(
                positions[i + 1] == positions[i] + 1 for i in range(len(positions) - 1)
            ):
                joined = '_'.join(ar_ent['tokens'])
                
                modifications['joined_entities'].append({
                    'original': ar_ent['text'],
                    'joined': joined,
                    'positions': ar_ent['token_positions'],
                    'language': 'ar',
                    'type': ar_ent['type']
                })
                
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = joined
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
        
        # Join English multi-word entities
        for en_ent in entities['en_entities']:
            positions = en_ent['token_positions']
            # Only join if the entity covers 2+ tokens AND they are contiguous
            if len(positions) > 1 and all(
                positions[i + 1] == positions[i] + 1 for i in range(len(positions) - 1)
            ):
                joined = '_'.join(en_ent['tokens'])
                
                modifications['joined_entities'].append({
                    'original': en_ent['text'],
                    'joined': joined,
                    'positions': en_ent['token_positions'],
                    'language': 'en',
                    'type': en_ent['type']
                })
                
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = joined
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    # ========== D4.2: Transliteration Matching ==========
    def method_transliteration(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.2: Replace Arabic entities with transliterations.
        Join multi-word entities with underscore.
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيويورك"
            EN: "Ahmad went to New York"
        
        Output:
            AR: "Ohmd ذهب إلى nywyurk"
            EN: "Ahmad went to New_York"
        
        Changes:
            - "أحمد" → "Ohmd" (transliterated)
            - "نيويورك" → "nywyurk" (transliterated)
            - "New York" → "New_York" (joined)
            - Non-entity words unchanged: "ذهب", "إلى", "went", "to"
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        ar_skip = set()
        en_skip = set()
        modifications = {
            'method': 'transliteration_matching',
            'replacements': []
        }
        
        # Replace Arabic entities with transliterations
        for ar_ent in entities['ar_entities']:
            if not ar_ent['token_positions']:
                continue
            
            # Transliterate
            ar_translit = self.ner.transliterate(ar_ent['text'])
            ar_joined = ar_translit.replace(' ', '_')
            
            # Track modification
            modification = {
                'ar_original': ar_ent['text'],
                'ar_replaced': ar_joined,
                'ar_positions': ar_ent['token_positions'],
                'type': ar_ent['type']
            }
            modifications['replacements'].append(modification)
            
            # Replace in Arabic
            first_pos = ar_ent['token_positions'][0]
            ar_tokens[first_pos] = ar_joined
            for pos in ar_ent['token_positions'][1:]:
                ar_skip.add(pos)
        
        # Join multi-word English entities
        for en_ent in entities['en_entities']:
            if not en_ent['token_positions'] or len(en_ent['token_positions']) <= 1:
                continue
            
            # Join multi-word entity
            en_joined = en_ent['text'].replace(' ', '_')
            
            # Replace in English
            first_pos = en_ent['token_positions'][0]
            en_tokens[first_pos] = en_joined
            for pos in en_ent['token_positions'][1:]:
                en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    # ========== D4.3: Entity Placeholder ==========
    def method_placeholder(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.3: Replace entities with typed placeholders.
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيو يورك مع سارة"
            EN: "Ahmad went to New York with Sarah"
        
        Output:
            AR: "PER_1 ذهب إلى LOC_1 مع PER_2"
            EN: "PER_1 went to LOC_1 with PER_2"
        
        Changes:
            - "أحمد" → "PER_1" (matched with "Ahmad")
            - "Ahmad" → "PER_1" (same placeholder)
            - "نيو يورك" → "LOC_1" (matched with "New York")
            - "New York" → "LOC_1" (same placeholder)
            - "سارة" → "PER_2" (matched with "Sarah")
            - "Sarah" → "PER_2" (same placeholder)
            - Non-entity words unchanged: "ذهب", "إلى", "مع", "went", "to", "with"
        
        Note: Matched entities get SAME placeholder for perfect alignment
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        matches = self._match_entities(
            entities['ar_entities'],
            entities['en_entities']
        )
        
        ar_skip = set()
        en_skip = set()
        counters = defaultdict(int)
        
        modifications = {
            'method': 'entity_placeholder',
            'placeholders': []
        }
        
        # Replace matched entities
        for match in matches:
            ar_ent = match['ar_entity']
            en_ent = match['en_entity']
            
            counters[ar_ent['type']] += 1
            placeholder = f"{ar_ent['type']}_{counters[ar_ent['type']]}"
            
            # Track modification
            modification = {
                'placeholder': placeholder,
                'ar_original': ar_ent['text'],
                'ar_positions': ar_ent['token_positions'],
                'en_original': en_ent['text'],
                'en_positions': en_ent['token_positions'],
                'type': ar_ent['type']
            }
            modifications['placeholders'].append(modification)
            
            # Replace Arabic
            if ar_ent['token_positions']:
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = placeholder
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
            
            # Replace English
            if en_ent['token_positions']:
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = placeholder
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        # Handle unmatched (optional - assign different placeholders)
        unmatched_ar = [e for e in entities['ar_entities'] 
                       if not any(m['ar_entity'] == e for m in matches)]
        for ar_ent in unmatched_ar:
            counters[ar_ent['type']] += 1
            placeholder = f"{ar_ent['type']}_{counters[ar_ent['type']]}"
            
            modifications['placeholders'].append({
                'placeholder': placeholder,
                'ar_original': ar_ent['text'],
                'ar_positions': ar_ent['token_positions'],
                'en_original': None,  # Unmatched
                'type': ar_ent['type']
            })
            
            if ar_ent['token_positions']:
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = placeholder
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
        
        unmatched_en = [e for e in entities['en_entities']
                       if not any(m['en_entity'] == e for m in matches)]
        for en_ent in unmatched_en:
            counters[en_ent['type']] += 1
            placeholder = f"{en_ent['type']}_{counters[en_ent['type']]}"
            
            modifications['placeholders'].append({
                'placeholder': placeholder,
                'ar_original': None,  # Unmatched
                'en_original': en_ent['text'],
                'en_positions': en_ent['token_positions'],
                'type': en_ent['type']
            })
            
            if en_ent['token_positions']:
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = placeholder
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    # ========== D4.4: Entity Tagging ==========
    def method_tagging(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.4: Add entity type tags.
        Format: TYPE:entity_text (with underscore for multi-word)
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيو يورك"
            EN: "Ahmad went to New York"
        
        Output:
            AR: "PER:أحمد ذهب إلى LOC:نيو_يورك"
            EN: "PER:Ahmad went to LOC:New_York"
        
        Changes:
            - "أحمد" → "PER:أحمد" (type prefix added)
            - "Ahmad" → "PER:Ahmad" (type prefix added)
            - "نيو يورك" → "LOC:نيو_يورك" (type prefix + joined)
            - "New York" → "LOC:New_York" (type prefix + joined)
            - Non-entity words unchanged: "ذهب", "إلى", "went", "to"
        
        Note: Keeps original entity text (unlike placeholder), just adds type info
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        ar_skip = set()
        en_skip = set()
        
        modifications = {
            'method': 'entity_tagging',
            'tagged_entities': []
        }
        
        # Tag Arabic
        for ar_ent in entities['ar_entities']:
            if ar_ent['token_positions']:
                entity_joined = '_'.join(ar_ent['tokens'])
                tagged = f"{ar_ent['type']}:{entity_joined}"
                
                modifications['tagged_entities'].append({
                    'original': ar_ent['text'],
                    'tagged': tagged,
                    'positions': ar_ent['token_positions'],
                    'language': 'ar',
                    'type': ar_ent['type']
                })
                
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = tagged
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
        
        # Tag English
        for en_ent in entities['en_entities']:
            if en_ent['token_positions']:
                entity_joined = '_'.join(en_ent['tokens'])
                tagged = f"{en_ent['type']}:{entity_joined}"
                
                modifications['tagged_entities'].append({
                    'original': en_ent['text'],
                    'tagged': tagged,
                    'positions': en_ent['token_positions'],
                    'language': 'en',
                    'type': en_ent['type']
                })
                
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = tagged
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    # ========== D4.6: Generic Entity Tagging ==========
    def method_tagging_generic(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.6: Add generic entity tag (NE) to all entities.
        Format: NE:entity_text (with underscore for multi-word)
        Uses "NE" (Named Entity) as a generic tag for all entity types.
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيو يورك"
            EN: "Ahmad went to New York"
        
        Output:
            AR: "NE:أحمد ذهب إلى NE:نيو_يورك"
            EN: "NE:Ahmad went to NE:New_York"
        
        Changes:
            - "أحمد" → "NE:أحمد" (generic tag added)
            - "Ahmad" → "NE:Ahmad" (generic tag added)
            - "نيو يورك" → "NE:نيو_يورك" (generic tag + joined)
            - "New York" → "NE:New_York" (generic tag + joined)
            - Non-entity words unchanged: "ذهب", "إلى", "went", "to"
        
        Note: Uses "NE" (Named Entity) tag for all entities regardless of type.
        This is familiar to neural models trained on NER tasks.
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        ar_skip = set()
        en_skip = set()
        
        modifications = {
            'method': 'entity_tagging_generic',
            'tagged_entities': []
        }
        
        # Tag Arabic entities with generic NE tag
        for ar_ent in entities['ar_entities']:
            if ar_ent['token_positions']:
                entity_joined = '_'.join(ar_ent['tokens'])
                tagged = f"NE:{entity_joined}"
                
                modifications['tagged_entities'].append({
                    'original': ar_ent['text'],
                    'tagged': tagged,
                    'positions': ar_ent['token_positions'],
                    'language': 'ar',
                    'original_type': ar_ent['type']  # Keep original type for reference
                })
                
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = tagged
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
        
        # Tag English entities with generic NE tag
        for en_ent in entities['en_entities']:
            if en_ent['token_positions']:
                entity_joined = '_'.join(en_ent['tokens'])
                tagged = f"NE:{entity_joined}"
                
                modifications['tagged_entities'].append({
                    'original': en_ent['text'],
                    'tagged': tagged,
                    'positions': en_ent['token_positions'],
                    'language': 'en',
                    'original_type': en_ent['type']  # Keep original type for reference
                })
                
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = tagged
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    # ========== D4.5: Cross-lingual NER ==========
    def method_crosslingual(self, ar_text, en_text, original_ar, original_en, entities=None):
        """
        D4.5: Join multi-word entities + provide match metadata for forced alignment.
        
        Example:
        Input:
            AR: "أحمد ذهب إلى نيو يورك مع سارة"
            EN: "Ahmad went to New York with Sarah"
        
        Output:
            AR: "أحمد ذهب إلى نيو_يورك مع سارة"
            EN: "Ahmad went to New_York with Sarah"
        
        Metadata (entity_matches):
            - {ar: "أحمد", en: "Ahmad", type: "PER", score: 0.92}
            - {ar: "نيو يورك", en: "New York", type: "LOC", score: 0.88}
            - {ar: "سارة", en: "Sarah", type: "PER", score: 0.90}
        
        Changes:
            - Multi-word entities joined (like D4.1)
            - BUT also provides entity matches for forced alignment
            - High-confidence matches (score > 0.85) can be used as alignment constraints
        
        Difference from D4.1:
            - D4.1: Just joins entities, no matching
            - D4.5: Joins entities AND matches them across languages
        
        Use case: Hybrid alignment (force high-confidence entity alignments)
        """
        if entities is None:
            entities = self.ner.extract_entities(ar_text, en_text)
        
        ar_tokens = entities['ar_tokens'].copy()
        en_tokens = entities['en_tokens'].copy()
        
        matches = self._match_entities(
            entities['ar_entities'],
            entities['en_entities']
        )
        
        ar_skip = set()
        en_skip = set()
        
        modifications = {
            'method': 'crosslingual_ner',
            'entity_matches': [],
            'joined_entities': []
        }
        
        # Process matched entities
        for match in matches:
            ar_ent = match['ar_entity']
            en_ent = match['en_entity']
            
            # Store match metadata
            modifications['entity_matches'].append({
                'ar_entity': ar_ent['text'],
                'en_entity': en_ent['text'],
                'type': ar_ent['type'],
                'match_score': match['score']
            })
            
            # Join Arabic multi-word entities
            if len(ar_ent['token_positions']) > 1:
                joined = '_'.join(ar_ent['tokens'])
                
                modifications['joined_entities'].append({
                    'original': ar_ent['text'],
                    'joined': joined,
                    'positions': ar_ent['token_positions'],
                    'language': 'ar'
                })
                
                first_pos = ar_ent['token_positions'][0]
                ar_tokens[first_pos] = joined
                for pos in ar_ent['token_positions'][1:]:
                    ar_skip.add(pos)
            
            # Join English multi-word entities
            if len(en_ent['token_positions']) > 1:
                joined = '_'.join(en_ent['tokens'])
                
                modifications['joined_entities'].append({
                    'original': en_ent['text'],
                    'joined': joined,
                    'positions': en_ent['token_positions'],
                    'language': 'en'
                })
                
                first_pos = en_ent['token_positions'][0]
                en_tokens[first_pos] = joined
                for pos in en_ent['token_positions'][1:]:
                    en_skip.add(pos)
        
        ar_final = [tok for i, tok in enumerate(ar_tokens) if i not in ar_skip]
        en_final = [tok for i, tok in enumerate(en_tokens) if i not in en_skip]
        
        return {
            'ar': ' '.join(ar_final),
            'en': ' '.join(en_final),
            'original_ar': original_ar,
            'original_en': original_en,
            'modifications': modifications
        }
    
    def _match_entities(self, ar_entities, en_entities):
        """Match entities across languages."""
        matches = []
        
        for ar_ent in ar_entities:
            best_match = None
            best_score = 0
            
            ar_translit = self.ner.transliterate(ar_ent['text'])
            
            for en_ent in en_entities:
                if ar_ent['type'] != en_ent['type']:
                    continue
                
                # Position similarity
                ar_pos = ar_ent['token_positions'][0] if ar_ent['token_positions'] else 0
                en_pos = en_ent['token_positions'][0] if en_ent['token_positions'] else 0
                pos_diff = abs(ar_pos - en_pos)
                pos_score = max(0, 1 - pos_diff / 10)
                
                # Transliteration similarity
                trans_score = SequenceMatcher(
                    None,
                    ar_translit.lower(),
                    en_ent['text'].lower()
                ).ratio()
                
                score = 0.4 * pos_score + 0.6 * trans_score
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = en_ent
            
            if best_match:
                matches.append({
                    'ar_entity': ar_ent,
                    'en_entity': best_match,
                    'score': best_score
                })
        
        return matches


def process_dataset(input_file, output_dir, ner_methods, methods_to_apply):
    """Process dataset with all NER methods.
    
    Optimized: Extracts entities once per entry and reuses them for all methods.
    """
    
    print(f"\nProcessing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Persistent cache for recognized entities per sentence id
    cache_dir = os.path.join(output_dir, "ner_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(input_file))
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as cf:
            entity_cache = json.load(cf)
    else:
        entity_cache = {}
    
    # Initialize output dictionaries for each method
    method_outputs = {method_name: [] for method_name in methods_to_apply}
    
    # Process each entry: extract entities once, apply all methods
    print(f"  Extracting entities and applying {len(methods_to_apply)} method(s) to {len(data)} entries...")
    
    for entry in tqdm(data, desc="  Processing entries"):
        ar_text = entry.get('ar', '')
        ar_agg = entry.get('ar_agg', ar_text)
        en_text = entry.get('en', '')
        sent_id = entry.get('id', '')
        
        # Store originals
        original_ar = ar_text
        original_en = en_text
        
        # Extract entities with on-disk cache per sentence id
        if sent_id and sent_id in entity_cache:
            entities = entity_cache[sent_id]
        else:
        entities = ner_methods.ner.extract_entities(ar_text, en_text)
            if sent_id:
                entity_cache[sent_id] = entities
        
        # Apply all methods using the cached entities
        for method_name in methods_to_apply:
            if method_name == 'join_multiword':
                result = ner_methods.method_join_multiword(ar_text, en_text, original_ar, original_en, entities)
            elif method_name == 'transliteration_matching':
                result = ner_methods.method_transliteration(ar_text, en_text, original_ar, original_en, entities)
            elif method_name == 'entity_placeholder':
                result = ner_methods.method_placeholder(ar_text, en_text, original_ar, original_en, entities)
            elif method_name == 'entity_tagging':
                result = ner_methods.method_tagging(ar_text, en_text, original_ar, original_en, entities)
            elif method_name == 'entity_tagging_generic':
                result = ner_methods.method_tagging_generic(ar_text, en_text, original_ar, original_en, entities)
            elif method_name == 'crosslingual_ner':
                result = ner_methods.method_crosslingual(ar_text, en_text, original_ar, original_en, entities)
            else:
                continue
            
            # Create output entry
            output_entry = {
                'ar': result['ar'],
                'ar_agg': ar_agg,
                'en': result['en'],
                'original_ar': result['original_ar'],
                'original_en': result['original_en'],
                'id': entry.get('id', '')
            }
            
            # Add modifications metadata
            if result['modifications']:
                output_entry['modifications'] = result['modifications']
            
            method_outputs[method_name].append(output_entry)
    
    # Save results for each method
    # Map method names to directory names
    method_dir_map = {
        'join_multiword': 'split_d4.1_join_multiword',
        'transliteration_matching': 'split_d4.2_transliteration_matching',
        'entity_placeholder': 'split_d4.3_entity_placeholder',
        'entity_tagging': 'split_d4.4_entity_tagging',
        'entity_tagging_generic': 'split_d4.6_entity_tagging_generic',
        'crosslingual_ner': 'split_d4.5_crosslingual_ner'
    }
    
    for method_name in methods_to_apply:
        print(f"  Saving results for method: {method_name}")
        
        # Use mapped directory name or default to split_d4_{method_name}
        dir_name = method_dir_map.get(method_name, f'split_d4_{method_name}')
        method_dir = os.path.join(output_dir, dir_name)
        os.makedirs(method_dir, exist_ok=True)
        
        output_file = os.path.join(method_dir, os.path.basename(input_file))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(method_outputs[method_name], f, ensure_ascii=False, indent=2)
        
        print(f"    Saved {len(method_outputs[method_name])} entries to: {output_file}")

    # Save updated entity cache to disk
    with open(cache_file, "w", encoding="utf-8") as cf:
        json.dump(entity_cache, cf, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Apply NER methods for alignment (Phase 1C)'
    )
    parser.add_argument('--input-dir', type=str, default='SEGMENTATION/split_b4',
                       help='Input directory (segmented data)')
    parser.add_argument('--output-dir', type=str, default='NER',
                       help='Output directory')
    parser.add_argument('--methods', nargs='+',
                       default=['join_multiword', 'transliteration_matching', 
                               'entity_placeholder', 'entity_tagging',
                               'crosslingual_ner'],
                       help='NER methods to apply')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device (default: 0)')
    
    args = parser.parse_args()
    
    print('=' * 80)
    print('Phase 1C: NER Methods for Word Alignment')
    print('=' * 80)
    print(f'Device: GPU {args.device}')
    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Methods: {", ".join(args.methods)}')
    print('=' * 80)
    
    ner_model = TransformerNER(device=args.device)
    ner_methods = NERMethods(ner_model)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(args.input_dir, f'{split}.json')
        
        if os.path.exists(input_file):
            process_dataset(input_file, args.output_dir, ner_methods, args.methods)
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print('\n' + '=' * 80)
    print('Phase 1C Complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()