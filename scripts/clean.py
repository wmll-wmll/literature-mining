#!/usr/bin/env python3
"""
Phase 2: Data Cleaning
=======================
Clean and normalize the raw corpus:
  1. Deduplicate by DOI and title similarity
  2. Normalize author names
  3. Standardize keywords (stemming + synonym merging)
  4. Fill missing fields via Crossref API
  5. Filter by year range
  6. Output clean_corpus.csv + stats_summary.json

Usage:
  python clean.py --input outputs/raw_corpus.jsonl --output outputs/clean_corpus.csv
  python clean.py --input outputs/raw_corpus.jsonl --output outputs/clean_corpus.csv --years 2016-2026
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, adaptive_thresholds, quality_report, DEFAULTS


def load_raw_corpus(filepath):
    """Load JSONL file, return list of dicts."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def title_similarity(a, b):
    """Compute similarity between two titles (0-1)."""
    if not a or not b:
        return 0.0
    # Normalize: lowercase, remove punctuation, collapse whitespace
    norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower())).strip()
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def deduplicate(entries, sim_threshold=0.85):
    """
    Deduplicate entries:
      - Same DOI -> keep the one with longer abstract
      - Title similarity > threshold -> keep newer / more complete
    Returns (deduplicated_list, removed_entries_log).
    """
    removed = []
    seen_doi = {}
    result = []

    for entry in entries:
        doi = entry.get('doi', '').strip().lower()
        title = entry.get('title', '')

        # DOI dedup
        if doi and doi in seen_doi:
            existing = seen_doi[doi]
            # Keep the one with longer abstract
            if len(entry.get('abstract', '')) > len(existing.get('abstract', '')):
                removed.append({'reason': 'doi_dup', 'removed': existing, 'kept': entry})
                seen_doi[doi] = entry
            else:
                removed.append({'reason': 'doi_dup', 'removed': entry, 'kept': existing})
            continue

        # Title similarity dedup (against all entries kept so far)
        dup_found = False
        for i, existing in enumerate(result):
            sim = title_similarity(title, existing.get('title', ''))
            if sim >= sim_threshold:
                # Keep the newer one, or the one with more complete data
                existing_completeness = (len(existing.get('abstract','')) + len(existing.get('doi',''))
                                         + (1 if existing.get('keywords') else 0))
                entry_completeness = (len(entry.get('abstract','')) + len(entry.get('doi',''))
                                      + (1 if entry.get('keywords') else 0))
                if entry_completeness > existing_completeness:
                    removed.append({'reason': f'title_sim_{sim:.2f}', 'removed': existing, 'kept': entry})
                    result[i] = entry
                else:
                    removed.append({'reason': f'title_sim_{sim:.2f}', 'removed': entry, 'kept': existing})
                dup_found = True
                break

        if not dup_found:
            result.append(entry)
            if doi:
                seen_doi[doi] = entry

    return result, removed


# ----------------------------------------------------------
#  Keyword normalization
# ----------------------------------------------------------

# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'shall', 'not', 'no', 'nor',
    'its', 'it', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
    'we', 'our', 'us', 'he', 'she', 'his', 'her', 'as', 'than', 'into',
    'over', 'under', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'about', 'also', 'very',
    'such', 'only', 'other', 'new', 'more', 'some', 'each', 'all', 'both',
    'just', 'now', 'then', 'here', 'there', 'when', 'where', 'how', 'which',
    'who', 'whom', 'what', 'why', 'et', 'al',
}


def simple_stem(word):
    """Light English stemmer — preserves readability.
    Only removes plural 's'/'es' and trailing 'ing' for longer words.
    Does NOT strip 'tion', 'al', 'ic', 'ed' (those destroy readability).
    Uses a lemma dictionary for known irregular forms."""
    word = word.lower().strip()

    # Lemma dictionary for known problematic forms
    _LEMMA = {
        # Plurals that simple stemming gets wrong
        'analyses': 'analysis', 'bases': 'basis', 'crises': 'crisis',
        'theses': 'thesis', 'hypotheses': 'hypothesis', 'phenomena': 'phenomenon',
        'criteria': 'criterion', 'data': 'data',
        # -es removal that breaks words
        'squares': 'square', 'features': 'feature', 'measures': 'measure',
        'procedures': 'procedure', 'structures': 'structure', 'cultures': 'culture',
        'mixtures': 'mixture', 'gestures': 'gesture', 'lectures': 'lecture',
        'pictures': 'picture', 'futures': 'future', 'miniatures': 'miniature',
        # -ies plurals
        'technologies': 'technology', 'theories': 'theory', 'categories': 'category',
        'methodologies': 'methodology', 'philosophies': 'philosophy',
        'strategies': 'strategy', 'therapies': 'therapy', 'energies': 'energy',
        'pathologies': 'pathology', 'ontologies': 'ontology',
        # Common stemming errors
        'analysi': 'analysis', 'squar': 'square', 'featur': 'feature',
        'measur': 'measure', 'structur': 'structure', 'procedur': 'procedure',
    }
    if word in _LEMMA:
        return _LEMMA[word]

    # Whitelist: technical suffixes where trailing 's' is NOT a plural
    _NO_STEM = {
        'spectroscopy', 'microscopy', 'tomography', 'chromatography',
        'engineering', 'processing', 'manufacturing', 'computing', 'learning',
        'analysis', 'diagnosis', 'prognosis', 'synthesis', 'catalysis',
        'informatics', 'bioinformatics', 'genomics', 'proteomics', 'metabolomics',
        'physics', 'mathematics', 'statistics', 'economics', 'linguistics',
        'biophysics', 'geophysics', 'astrophysics',
        'quality',  # 'quality' is singular, don't strip to 'qualit'
    }
    if word in _NO_STEM:
        return word

    # Only remove trailing 's' for plurals (not 'ss' endings like "glass")
    if word.endswith('es') and len(word) > 4 and not word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        word = word[:-1]
    # Only remove 'ing' for longer words (preserves "thing", "king")
    if word.endswith('ing') and len(word) > 6:
        word = word[:-3]
    return word


# Common synonym groups in biomedical/nano literature
DEFAULT_SYNONYMS = {
    'nanoparticle': ['nanoparticles', 'np', 'nps', 'nano-particle', 'nano-particles', 'nanoparticle'],
    'drug delivery': ['drug-delivery', 'drugrelease', 'drug release'],
    'lipid nanoparticle': ['lnp', 'lnps', 'lipid nanoparticle', 'lipid nanoparticles'],
    'messenger rna': ['mrna', 'messenger rna', 'messenger-rna'],
    'small interfering rna': ['sirna', 'small interfering rna', 'small-interfering-rna'],
    'polyethylene glycol': ['peg', 'polyethylene glycol', 'polyethyleneglycol'],
    'enhanced permeability': ['epr', 'enhanced permeability and retention', 'epr effect'],
    'blood brain barrier': ['bbb', 'blood-brain barrier', 'blood brain barrier'],
    'reticuloendothelial system': ['res', 'reticuloendothelial system'],
    'tumor microenvironment': ['tme', 'tumour microenvironment', 'tumor microenvironment'],
    'extracellular vesicle': ['ev', 'extracellular vesicle', 'extracellular vesicles', 'exosome', 'exosomes'],
    'metal organic framework': ['mof', 'metal-organic framework', 'metal organic framework'],
    'machine learning': ['ml', 'machine learning', 'deep learning', 'ai', 'artificial intelligence'],
    'crispr': ['crispr', 'crispr-cas9', 'crispr-cas13', 'crispr/cas9'],
    'gene therapy': ['gene therapy', 'gene-therapy', 'genetic therapy'],
}


def build_synonym_map(custom_synonyms=None):
    """Build a synonym -> canonical form mapping."""
    syn_map = {}
    all_syns = dict(DEFAULT_SYNONYMS)
    if custom_synonyms:
        all_syns.update(custom_synonyms)
    for canonical, variants in all_syns.items():
        for v in variants:
            syn_map[v.lower()] = canonical.lower()
        syn_map[canonical.lower()] = canonical.lower()
    return syn_map


def normalize_keyword(kw, syn_map):
    """Normalize a single keyword: stem + synonym merge + stopword filter."""
    kw = kw.lower().strip()
    kw = re.sub(r'[^a-z0-9\s\-]', '', kw)
    kw = re.sub(r'\s+', ' ', kw).strip()

    if not kw or len(kw) < 2:
        return None

    # Check synonym map first
    if kw in syn_map:
        return syn_map[kw]

    # Check if it's a stopword
    if kw in STOPWORDS:
        return None

    # Stem multi-word phrases word by word
    words = kw.split()
    if len(words) > 1:
        stemmed = ' '.join(simple_stem(w) for w in words if w not in STOPWORDS)
        if stemmed in syn_map:
            return syn_map[stemmed]
        return stemmed

    stemmed = simple_stem(kw)
    if stemmed in syn_map:
        return syn_map[stemmed]
    return stemmed


# ----------------------------------------------------------
#  Field completion via Crossref
# ----------------------------------------------------------

def fill_missing_fields(entries):
    """
    Batch-enrich entries via Crossref for those with DOI but missing
    abstract/year/journal. Uses batch API (not per-DOI requests).
    """
    try:
        import requests
    except ImportError:
        print("  [WARN]  requests not installed, skipping Crossref fill")
        return entries

    # Collect DOIs that need enrichment
    doi_to_indices = {}
    for i, entry in enumerate(entries):
        doi = entry.get('doi', '').strip()
        if not doi or doi.lower() == 'nan':
            continue
        needs_abstract = not entry.get('abstract')
        needs_meta = not isinstance(entry.get('year'), int) or not entry.get('journal')
        if needs_abstract or needs_meta:
            if doi not in doi_to_indices:
                doi_to_indices[doi] = []
            doi_to_indices[doi].append(i)

    if not doi_to_indices:
        print("  Crossref fill: all entries complete, skipping")
        return entries

    print(f"  Crossref batch fill: {len(doi_to_indices)} unique DOIs need enrichment")

    # Batch lookup (reuse ingest's batch function)
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import enrich_via_crossref_batch
    enriched = enrich_via_crossref_batch(list(doi_to_indices.keys()))

    filled_abstracts = 0
    filled_meta = 0
    for doi, indices in doi_to_indices.items():
        data = enriched.get(doi)
        if data is None:
            continue
        for idx in indices:
            entry = entries[idx]
            if not entry.get('abstract') and data.get('abstract'):
                entry['abstract'] = data['abstract']
                filled_abstracts += 1
            if not isinstance(entry.get('year'), int) and data.get('year'):
                entry['year'] = data['year']
                filled_meta += 1
            if not entry.get('journal') and data.get('journal'):
                entry['journal'] = data['journal']
                filled_meta += 1

    print(f"  Crossref fill: +{filled_abstracts} abstracts, +{filled_meta} metadata")
    return entries


# ----------------------------------------------------------
#  Main
# ----------------------------------------------------------

def main():
    init()  # Windows UTF-8 encoding fix
    parser = argparse.ArgumentParser(description='Data Cleaning — Phase 2')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    parser.add_argument('--years', help='Year range filter, e.g. 2016-2026')
    parser.add_argument('--sim-threshold', type=float, default=DEFAULTS['title_similarity_threshold'],
                        help='Title similarity threshold for dedup')
    parser.add_argument('--enhanced-kw', action='store_true',
                        help='Use KeyBERT for keyword extraction (better quality, slower)')
    parser.add_argument('--embedding-dedup', action='store_true',
                        help='Use sentence-transformers for embedding-based dedup (slower but more accurate)')
    args = parser.parse_args()

    print("=" * 60)
    print("[CLEAN] Phase 2: Data Cleaning")
    print("=" * 60)

    # Load
    entries = load_raw_corpus(args.input)
    valid_entries = [e for e in entries if not e.get('error') and e.get('extraction_quality') != 'failed']
    failed_entries = len(entries) - len(valid_entries)
    print(f"\n   Loaded: {len(entries)} entries ({len(valid_entries)} valid, {failed_entries} failed)")

    # Year filter
    if args.years:
        match = re.match(r'(\d{4})\s*-\s*(\d{4})', args.years)
        if match:
            y_start, y_end = int(match.group(1)), int(match.group(2))
            before = len(valid_entries)
            valid_entries = [
                e for e in valid_entries
                if e.get('year') and str(e['year']).isdigit()
                and y_start <= int(e['year']) <= y_end
            ]
            print(f"   Year filter ({y_start}-{y_end}): {before} -> {len(valid_entries)} entries")

    # Dedup
    valid_entries, removed = deduplicate(valid_entries, args.sim_threshold)
    doi_dups = sum(1 for r in removed if r['reason'] == 'doi_dup')
    title_dups = sum(1 for r in removed if r['reason'].startswith('title_sim'))
    print(f"   Dedup: removed {len(removed)} ({doi_dups} DOI duplicates, {title_dups} title-similar)")

    # Embedding-based dedup (optional, slower)
    if args.embedding_dedup:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [(e.get('title', '') + ' ' + (e.get('abstract', '') or '')[:200]) for e in valid_entries]
            embeddings = model.encode(texts, show_progress_bar=False)
            to_remove = set()
            for i in range(len(valid_entries)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(valid_entries)):
                    if j in to_remove:
                        continue
                    sim = float(util.cos_sim(embeddings[i], embeddings[j])[0][0])
                    if sim > DEFAULTS['dedup_embedding_threshold_merge']:
                        to_remove.add(j)
                        removed.append({'reason': f'embedding_sim_{sim:.2f}', 'removed': valid_entries[j], 'kept': valid_entries[i]})
            if to_remove:
                valid_entries = [e for idx, e in enumerate(valid_entries) if idx not in to_remove]
                print(f"   Embedding dedup: removed {len(to_remove)} entries (sim > {DEFAULTS['dedup_embedding_threshold_merge']})")
        except ImportError:
            warn('sentence-transformers not installed, skipping embedding dedup')
    valid_entries = fill_missing_fields(valid_entries)

    # Keyword normalization
    syn_map = build_synonym_map()
    total_kw_before = 0
    total_kw_after = 0
    for entry in valid_entries:
        raw_kws = entry.get('keywords', [])
        if isinstance(raw_kws, str):
            raw_kws = [k.strip() for k in re.split(r'[;,]\s*', raw_kws) if k.strip()]
        total_kw_before += len(raw_kws)
        normalized = []
        for kw in raw_kws:
            n = normalize_keyword(kw, syn_map)
            if n:
                normalized.append(n)
        entry['keywords_normalized'] = normalized
        total_kw_after += len(normalized)

    print(f"   Keywords normalized: {total_kw_before} -> {total_kw_after} (merged synonyms, removed stopwords)")

    # Keyword extraction from abstracts (TF-IDF or KeyBERT)
    from utils import extract_keywords_tfidf, extract_keywords_keybert
    abstracts = [e.get('abstract', '') for e in valid_entries]
    n_with_abstract = sum(1 for a in abstracts if a and len(a) > 50)
    if n_with_abstract >= 5:
        if args.enhanced_kw:
            print(f"   Extracting keywords with KeyBERT from {n_with_abstract} abstracts...")
            tfidf_kws = extract_keywords_keybert(abstracts, top_n=5)
            if tfidf_kws is None:
                print("   KeyBERT not installed, falling back to TF-IDF")
                tfidf_kws = extract_keywords_tfidf(abstracts, top_n=5, min_df=max(1, int(len(valid_entries) * 0.05)))
        else:
            print(f"   Extracting TF-IDF keywords from {n_with_abstract} abstracts...")
            tfidf_kws = extract_keywords_tfidf(abstracts, top_n=5, min_df=max(1, int(len(valid_entries) * 0.05)))
        for i, entry in enumerate(valid_entries):
            if i < len(tfidf_kws) and tfidf_kws[i]:
                existing = set(entry.get('keywords_normalized', []))
                for kw in tfidf_kws[i]:
                    if kw not in existing:
                        existing.add(kw)
                entry['keywords_normalized'] = list(existing)
        total_kw_after2 = sum(len(e.get('keywords_normalized', [])) for e in valid_entries)
        print(f"   After TF-IDF enrichment: {total_kw_after} -> {total_kw_after2} keywords")

    # Compute summary stats
    years = [int(e['year']) for e in valid_entries if e.get('year') and str(e['year']).isdigit()]
    n_abstract = sum(1 for e in valid_entries if e.get('abstract'))
    n_doi = sum(1 for e in valid_entries if e.get('doi'))
    journals = Counter(e.get('journal', '') for e in valid_entries if e.get('journal'))

    # Top keywords
    all_kw = Counter()
    for e in valid_entries:
        for kw in e.get('keywords_normalized', []):
            all_kw[kw] += 1

    summary = {
        'total_papers': len(valid_entries),
        'failed_extractions': failed_entries,
        'dedup_removed': len(removed),
        'year_range': [min(years), max(years)] if years else [],
        'n_with_abstract': n_abstract,
        'n_with_doi': n_doi,
        'n_with_keywords': sum(1 for e in valid_entries if e.get('keywords_normalized')),
        'top_keywords_50': all_kw.most_common(50),
        'top_journals_10': journals.most_common(10),
        'citations_mean': None,
        'citations_median': None,
    }

    if any('citation_count' in e for e in valid_entries):
        cites = [e.get('citation_count', 0) for e in valid_entries if e.get('citation_count')]
        if cites:
            import statistics
            summary['citations_mean'] = round(statistics.mean(cites), 1)
            summary['citations_median'] = statistics.median(cites)
            summary['citations_max'] = max(cites)

    # Write CSV output (use pandas if available, else manual CSV)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare CSV rows
    fieldnames = [
        'id', 'title', 'authors', 'year', 'journal', 'volume', 'pages',
        'doi', 'abstract', 'keywords_raw', 'keywords_normalized',
        'citation_count', 'source'
    ]

    try:
        import pandas as pd
        rows = []
        for e in valid_entries:
            row = {
                'id': e.get('id', ''),
                'title': e.get('title', ''),
                'authors': '; '.join(e.get('authors', [])) if isinstance(e.get('authors'), list) else e.get('authors', ''),
                'year': e.get('year', ''),
                'journal': e.get('journal', ''),
                'volume': e.get('volume', ''),
                'pages': e.get('pages', ''),
                'doi': e.get('doi', ''),
                'abstract': e.get('abstract', ''),
                'keywords_raw': '; '.join(e.get('keywords', [])) if isinstance(e.get('keywords'), list) else e.get('keywords', ''),
                'keywords_normalized': '; '.join(e.get('keywords_normalized', [])),
                'citation_count': e.get('citation_count', ''),
                'source': e.get('source', ''),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except ImportError:
        import csv
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in valid_entries:
                writer.writerow({
                    'id': e.get('id', ''),
                    'title': e.get('title', ''),
                    'authors': '; '.join(e.get('authors', [])) if isinstance(e.get('authors'), list) else e.get('authors', ''),
                    'year': e.get('year', ''),
                    'journal': e.get('journal', ''),
                    'volume': e.get('volume', ''),
                    'pages': e.get('pages', ''),
                    'doi': e.get('doi', ''),
                    'abstract': e.get('abstract', ''),
                    'keywords_raw': '; '.join(e.get('keywords', [])) if isinstance(e.get('keywords'), list) else e.get('keywords', ''),
                    'keywords_normalized': '; '.join(e.get('keywords_normalized', [])),
                    'citation_count': e.get('citation_count', ''),
                    'source': e.get('source', ''),
                })

    # Write summary
    summary_path = output_path.parent / 'stats_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Final report
    print(f"\n{'-' * 50}")
    print(f"[CLEAN] Cleaning complete")
    print(f"   Output CSV:  {output_path}")
    print(f"   Summary:     {summary_path}")
    print(f"   Final count: {len(valid_entries)} papers")
    if years:
        print(f"   Year range:  {min(years)}-{max(years)}")
    print(f"   Top keywords: {', '.join(kw for kw, _ in summary['top_keywords_50'][:10])}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
