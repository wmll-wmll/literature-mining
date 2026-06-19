#!/usr/bin/env python3
"""
Shared utilities for the literature-mining skill.
Imports: from utils import init, log, warn, error, retry_api, DEFAULTS
"""

import sys
import time
import math
from pathlib import Path

# ── Centralized configurable defaults ──
# All hardcoded thresholds live here. Override via CLI args where supported.
DEFAULTS = {
    # --- Ingestion ---
    'max_search_results': 200,
    'api_retry_max': 3,
    'api_retry_base_delay': 2.0,
    'api_rate_limit_delay': 15.0,
    'crossref_rate_limit_delay': 10.0,
    'openalex_per_page': 50,
    'openalex_max_pages': 4,
    # --- Cleaning ---
    'title_similarity_threshold': 0.85,
    'dedup_embedding_threshold_merge': 0.95,
    'dedup_embedding_threshold_review': 0.85,
    # --- Analysis ---
    'cooccurrence_top_k': 200,
    'trend_early_window': 2,
    'trend_late_window': 2,
    'trend_rising_ratio': 1.3,
    'trend_declining_ratio': 0.7,
    # --- Frontier detection ---
    'frontier_min_cluster_size_factor': 0.05,
    'frontier_min_cluster_size_absolute': 2,
    'burst_gamma': 1.0,
    'burst_min_years_for_detection': 3,
    'frontier_score_burst_weight': 0.35,
    'frontier_score_recency_weight': 0.30,
    'frontier_score_growth_weight': 0.25,
    'frontier_score_cross_weight': 0.10,
    'louvain_resolution': 1.5,  # higher = more granular communities
    # --- Gap detection ---
    'gap_density_proximity_threshold': 0.5,
    'gap_bridge_edge_ratio_small': 0.25,
    'gap_bridge_edge_ratio_large': 0.35,
    'gap_bridge_large_network_threshold': 30,
    'gap_temporal_min_years': 6,
    'gap_temporal_min_papers_factor': 0.05,  # min papers = max(10, n_papers * this)
    'gap_temporal_decline_ratio': 0.3,
    'gap_score_density_weight': 0.35,
    'gap_score_proximity_weight': 0.30,
    'gap_score_bridge_weight': 0.20,
    'gap_score_feasibility_weight': 0.15,
    'gap_semantic_min_distance': 0.3,
    # --- Pipeline ---
    'phase_timeout': 600,
    'cache_dir': 'outputs/.cache',
}


def init():
    """Initialize the environment. Call at start of every script."""
    # Windows UTF-8 encoding fix
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# ──────────────────────────────────────────────────────────
#  Logging (ASCII-safe)
# ──────────────────────────────────────────────────────────

_EMOJI_MAP = {
    'ingest':   '[INGEST]',
    'clean':    '[CLEAN]',
    'analyze':  '[ANALYZE]',
    'frontier': '[FRONTIER]',
    'gaps':     '[GAPS]',
    'report':   '[REPORT]',
    'warn':     '[WARN]',
    'error':    '[ERROR]',
    'ok':       '[OK]',
    'info':     '[INFO]',
}


def log(phase, msg):
    prefix = _EMOJI_MAP.get(phase, f'[{phase.upper()}]')
    print(f"{prefix} {msg}")


def warn(msg):
    print(f"[WARN]  {msg}")


def error(msg):
    print(f"[ERROR] {msg}")


def sep():
    print('-' * 55)


def banner(phase, title):
    print('=' * 60)
    log(phase, title)
    print('=' * 60)


# ──────────────────────────────────────────────────────────
#  API retry with exponential backoff
# ──────────────────────────────────────────────────────────

def retry_api(func, max_retries=3, base_delay=2.0, rate_limit_delay=30.0):
    """
    Call func() with exponential backoff retry.
    Returns (result, None) on success or (None, error_msg) on failure.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = func()
            if result is not None:
                return result, None
            return None, "API returned None"
        except Exception as e:
            last_error = str(e)
            if '429' in last_error or 'Too Many Requests' in last_error:
                wait = rate_limit_delay * (2 ** attempt)
                warn(f"Rate limited (429), waiting {wait:.0f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            elif attempt < max_retries:
                wait = base_delay * (2 ** attempt)
                warn(f"API error: {last_error[:80]}, retrying in {wait:.0f}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                warn(f"API error after {max_retries} retries: {last_error[:120]}")
                return None, last_error
    return None, last_error


# ──────────────────────────────────────────────────────────
#  Adaptive thresholds
# ──────────────────────────────────────────────────────────

def adaptive_thresholds(n_papers, n_keywords, year_span):
    """
    Compute adaptive thresholds based on corpus size and span.
    Returns dict of thresholds.
    """
    # Scale factors
    size_factor = max(0.3, min(1.0, n_papers / 100))
    span_factor = max(0.2, min(1.0, year_span / 10))

    return {
        'n_papers': n_papers,
        'n_keywords': n_keywords,
        'year_span': year_span,
        # Community detection
        'min_cluster_size': max(2, int(n_papers * 0.05)),
        'min_cooc_weight': max(1, int(n_papers * 0.02)),
        # Burst detection
        'burst_gamma': 2.0 - span_factor,  # looser for short spans
        'min_years_for_burst': max(3, int(6 * (1 - span_factor))),
        # Gap detection
        'density_proximity_threshold': max(0.2, 0.5 * size_factor),
        'bridge_edge_ratio_max': min(0.5, 0.2 / max(0.3, size_factor)),
        # Warnings
        'warn_small_corpus': n_papers < 30,
        'warn_narrow_span': year_span < 5,
        'warn_sparse_network': n_keywords < 50,
        # Recommendations
        'recommend_more_data': n_papers < 30 or year_span < 5,
        'recommend_abstract_kw': n_keywords < 30,
    }


def quality_report(thresholds, abstract_pct=0, dedup_rate=0, source_dist=None):
    """Print a data quality report and return (warnings, quality_dict)."""
    quality = {
        'n_papers': thresholds['n_papers'],
        'year_span': thresholds['year_span'],
        'n_keywords': thresholds['n_keywords'],
        'abstract_coverage_pct': round(abstract_pct * 100, 1) if abstract_pct else 0,
        'dedup_rate_pct': round(dedup_rate * 100, 1) if dedup_rate else 0,
        'source_distribution': source_dist or {},
        'warnings': [],
        'verdict': 'good',
    }
    warnings = quality['warnings']

    if thresholds['warn_small_corpus']:
        warnings.append(f"Small corpus ({thresholds['n_papers']} papers). Recommend >= 50.")
    if thresholds['warn_narrow_span']:
        warnings.append(f"Narrow year span ({thresholds['year_span']}yr). Recommend >= 5.")
    if thresholds['warn_sparse_network']:
        warnings.append(f"Low keyword count ({thresholds['n_keywords']}). Consider enabling enhanced keyword extraction.")
    if abstract_pct is not None and abstract_pct < 0.5:
        warnings.append(f"Low abstract coverage ({abstract_pct*100:.0f}%). Gap/frontier analysis may be noisy.")
    if dedup_rate > 0.3:
        warnings.append(f"High dedup rate ({dedup_rate*100:.0f}%). Review input for duplicates.")
    if source_dist:
        dominant = max(source_dist, key=source_dist.get)
        if source_dist[dominant] / max(1, sum(source_dist.values())) > 0.9:
            warnings.append(f"Single-source dominance ({dominant}: {source_dist[dominant]/max(1,sum(source_dist.values()))*100:.0f}%). May have coverage bias.")

    quality['verdict'] = 'poor' if len(warnings) >= 3 else ('fair' if len(warnings) >= 1 else 'good')

    if warnings:
        log('info', 'Data Quality Report:')
        for w in warnings:
            warn(w)
        log('info', f'Verdict: {quality["verdict"].upper()} ({len(warnings)} warnings)')
    else:
        log('ok', f'Data quality looks good: {thresholds["n_papers"]} papers, {thresholds["year_span"]}yr span, {thresholds["n_keywords"]} keywords, {quality["abstract_coverage_pct"]:.0f}% abstracts')

    return warnings, quality


# ──────────────────────────────────────────────────────────
#  Simple TF-IDF keyword extraction from text
# ──────────────────────────────────────────────────────────

def extract_keywords_keybert(texts, top_n=5):
    """Extract keywords using KeyBERT (pip install keybert).
    Falls back to None if unavailable — caller should use TF-IDF then."""
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        results = []
        for text in texts:
            if not text or len(text) < 50:
                results.append([])
                continue
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2),
                stop_words='english', top_n=top_n, use_mmr=True, diversity=0.5
            )
            results.append([kw for kw, score in keywords if score > 0.3])
        return results
    except ImportError:
        return None


def clean_text_for_keywords(text):
    """Strip XML tags, URLs, DOIs, and non-content artifacts from text."""
    import re
    text = str(text) if text else ''
    # Remove XML/HTML tags
    text = re.sub(r'</?[a-z]+[^>]*>', ' ', text)
    # Remove DOIs
    text = re.sub(r'\b10\.\d{4,}/[^\s]+', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove pure numbers
    text = re.sub(r'\b\d+\b', ' ', text)
    # Remove single characters
    text = re.sub(r'\b[a-z]\b', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


_ACADEMIC_NOISE = {
    # XML/formatting artifacts
    'jats', 'italic', 'xml', 'http', 'https', 'doi', 'org', 'www', 'tex',
    'latex', 'bibtex', 'pdf', 'html', 'css', 'div', 'span', 'href',
    # Figure/table references
    'fig', 'figure', 'table', 'tableau', 'scheme', 'chart',
    # Latin abbreviations
    'et', 'al', 'ie', 'eg', 'etc', 'via', 'per', 'vs', 'cf', 'ibid',
    # Generic academic filler verbs
    'like', 'use', 'used', 'uses', 'using', 'make', 'makes', 'making',
    'take', 'takes', 'taking', 'give', 'gives', 'giving', 'get', 'gets',
    'show', 'shows', 'shown', 'showcasing', 'demonstrate', 'demonstrates',
    'demonstrated', 'indicate', 'indicates', 'indicated', 'suggest',
    'suggests', 'suggested', 'reveal', 'reveals', 'revealed',
    'present', 'presents', 'presented', 'propose', 'proposes', 'proposed',
    'introduce', 'introduces', 'introduced', 'describe', 'describes',
    'described', 'discuss', 'discusses', 'discussed', 'explore', 'explores',
    'explored', 'examine', 'examines', 'examined', 'investigate',
    'investigates', 'investigated', 'report', 'reports', 'reported',
    'find', 'finds', 'found', 'observe', 'observes', 'observed',
    'note', 'notes', 'noted', 'highlight', 'highlights', 'highlighted',
    'address', 'addresses', 'addressed', 'consider', 'considers',
    'considered', 'apply', 'applies', 'applied', 'require', 'requires',
    'required', 'need', 'needs', 'needed', 'allow', 'allows', 'allowed',
    'enable', 'enables', 'enabled', 'provide', 'provides', 'provided',
    'include', 'includes', 'included', 'involve', 'involves', 'involved',
    'offer', 'offers', 'offered', 'achieve', 'achieves', 'achieved',
    'produce', 'produces', 'produced', 'generate', 'generates', 'generated',
    'develop', 'develops', 'developed', 'design', 'designs', 'designed',
    'implement', 'implements', 'implemented', 'perform', 'performs',
    'performed', 'conduct', 'conducts', 'conducted',
    # Generic academic filler nouns
    'role', 'roles', 'task', 'tasks', 'way', 'ways', 'case', 'cases',
    'example', 'examples', 'approach', 'approaches', 'method', 'methods',
    'methodology', 'technique', 'techniques', 'strategy', 'strategies',
    'framework', 'frameworks', 'result', 'results', 'finding', 'findings',
    'conclusion', 'conclusions', 'outcome', 'outcomes', 'effect', 'effects',
    'impact', 'impacts', 'influence', 'influences', 'implication',
    'implications', 'challenge', 'challenges', 'opportunity', 'opportunities',
    'limitation', 'limitations', 'issue', 'issues', 'problem', 'problems',
    'application', 'applications', 'context', 'contexts', 'setting',
    'settings', 'scenario', 'scenarios', 'aspect', 'aspects', 'factor',
    'factors', 'feature', 'features', 'property', 'properties',
    'characteristic', 'characteristics', 'parameter', 'parameters',
    'component', 'components', 'element', 'elements', 'part', 'parts',
    'process', 'processes', 'mechanism', 'mechanisms', 'function',
    'functions', 'structure', 'structures', 'system', 'systems', 'model',
    'models', 'type', 'types', 'kind', 'kinds', 'form', 'forms',
    # Academic boilerplate
    'paper', 'papers', 'article', 'articles', 'study', 'studies',
    'research', 'researches', 'review', 'reviews', 'survey', 'surveys',
    'analysis', 'analyses', 'overview', 'overviews', 'summary', 'summaries',
    'introduction', 'intro', 'background', 'discussion', 'discussions',
    'conclusion', 'conclusions', 'future', 'work', 'related', 'reference',
    'references', 'appendix', 'appendixes', 'supplement', 'supplementary',
    'author', 'authors', 'journal', 'journals', 'publisher', 'publishers',
    'volume', 'issue', 'page', 'pages', 'copyright', 'license', 'rights',
    'reserved', 'abstract', 'keyword', 'keywords', 'index', 'term', 'terms',
    'topic', 'topics', 'subject', 'subjects', 'field', 'fields', 'area',
    'areas', 'domain', 'domains', 'theme', 'themes',
    # Generic qualitative adjectives
    'new', 'novel', 'old', 'well', 'good', 'bad', 'better', 'worse',
    'best', 'worst', 'also', 'may', 'might', 'could', 'would', 'should',
    'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
    'early', 'late', 'earlier', 'later', 'many', 'much', 'few', 'less',
    'more', 'most', 'some', 'any', 'all', 'no', 'none', 'each', 'every',
    'both', 'either', 'neither', 'other', 'another', 'such', 'only',
    'same', 'own', 'very', 'much', 'very', 'quite', 'rather',
    'different', 'various', 'several', 'certain', 'specific', 'particular',
    'possible', 'potential', 'likely', 'unlikely', 'common', 'rare',
    'important', 'critical', 'crucial', 'essential', 'key', 'main', 'major',
    'minor', 'primary', 'secondary', 'recent', 'current', 'previous',
    'prior', 'subsequent', 'future', 'high', 'higher', 'low', 'lower',
    'large', 'larger', 'small', 'smaller', 'significant', 'insignificant',
    'sufficient', 'insufficient', 'adequate', 'inadequate',
    'deep', 'deeper', 'wide', 'wider', 'broad', 'broader', 'narrow',
    # Transition/connector words (often picked up by TF-IDF)
    'however', 'therefore', 'thus', 'although', 'though', 'whereas',
    'while', 'despite', 'nevertheless', 'nonetheless', 'furthermore',
    'moreover', 'additionally', 'besides', 'consequently', 'hence',
    'accordingly', 'thereby', 'whereby', 'hereby', 'indeed', 'rather',
    'instead', 'otherwise', 'meanwhile', 'overall', 'particularly',
    'especially', 'specifically', 'namely', 'respectively',
    'directly', 'indirectly', 'typically', 'generally', 'usually',
    'often', 'sometimes', 'rarely', 'always', 'never',
    'well', 'widely', 'extensively', 'increasingly', 'commonly',
    # Domain-level generics (common in CS/bio/ALife but too broad alone)
    'artificial', 'life', 'living', 'time', 'system', 'model',
    'development', 'toy', 'real', 'world', 'human',
    'simple', 'complex', 'complexity', 'evolution',
    'fitness', 'reproduction', 'growth', 'death', 'survival',
    'behavior', 'interaction', 'environment', 'population',
    'selection', 'adaptation', 'organization', 'information',
    'computation', 'simulation', 'experiment', 'data',
    'analysis', 'design', 'implementation', 'evaluation',
    # Overly generic nouns (common in TF-IDF but meaningless as gaps)
    'image', 'images', 'imaging', 'video', 'audio', 'text', 'language',
    'content', 'user', 'network', 'graph', 'tree', 'path', 'space',
    'object', 'target', 'source', 'signal', 'noise', 'pattern',
    'surface', 'material', 'device', 'tool', 'platform', 'infrastructure',
    # Broad OpenAlex L0/L1 concepts (too generic for keywords)
    'computer science', 'biology', 'medicine', 'physics', 'chemistry',
    'mathematics', 'engineering', 'psychology', 'sociology', 'philosophy',
    'materials science', 'geography', 'geology', 'economics', 'history',
    'political science', 'law', 'business', 'art', 'environmental science',
    'data science', 'cognitive science', 'neuroscience', 'linguistics',
    'education', 'archaeology', 'anthropology', 'theology',
    'algorithm', 'programming', 'software', 'hardware',
    'nanotechnology', 'biotechnology', 'bioinformatics',
    # L0/L1 concepts that slip through if filter fails (defense in depth)
    'chemistry', 'biology', 'physics', 'mathematics', 'engineering',
    'computer science', 'medicine', 'materials science', 'food science',
    'environmental science', 'business', 'economics', 'psychology',
    'sociology', 'political science', 'geography', 'geology', 'history',
    'philosophy', 'law', 'art', 'linguistics', 'education',
    'chemical engineering', 'electrical engineering', 'mechanical engineering',
    'biochemistry', 'molecular biology', 'cell biology', 'genetics',
    'analytical chemistry', 'organic chemistry', 'physical chemistry',
    'biochemical engineering', 'quality',
}


def extract_keywords_tfidf(texts, top_n=5, min_df=2):
    """
    Extract keywords from a list of texts using TF-IDF.
    Precleans text (strips XML/URLs/DOIs).
    Falls back gracefully if sklearn not available.
    texts: list of strings (abstracts)
    Returns: list of lists of keywords, one per text
    """
    # Preclean all texts
    texts = [clean_text_for_keywords(t) for t in texts]
    if not texts or all(not t for t in texts):
        return [[] for _ in texts]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _simple_keyword_extract(texts, top_n)

    try:
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=min_df,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        results = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-top_n * 2:][::-1]  # get more, then filter
            keywords = []
            for j in top_indices:
                if len(keywords) >= top_n:
                    break
                kw = feature_names[j]
                if row[j] <= 0:
                    continue
                # Filter noise: must be >= 3 chars, not in academic noise list
                if len(kw) < 3 or kw.lower() in _ACADEMIC_NOISE:
                    continue
                # Filter: no pure numeric tokens
                if all(c in '0123456789. ' for c in kw):
                    continue
                keywords.append(kw)
            results.append(keywords)
        return results
    except Exception:
        return _simple_keyword_extract(texts, top_n)


def _simple_keyword_extract(texts, top_n):
    """Fallback keyword extractor using word frequency with noise filtering."""
    import re
    from collections import Counter

    results = []
    for text in texts:
        if not text:
            results.append([])
            continue
        words = re.findall(r'\b[a-z]{4,}\b', str(text).lower())
        # Remove noise words
        words = [w for w in words if w not in _ACADEMIC_NOISE]
        freq = Counter(words)
        results.append([w for w, _ in freq.most_common(top_n)])
    return results


# ──────────────────────────────────────────────────────────
#  Progress feedback
# ──────────────────────────────────────────────────────────

def progress(current, total, label=""):
    """Print a progress bar."""
    pct = current / max(1, total) * 100
    bar_len = 20
    filled = int(bar_len * current / max(1, total))
    bar = '#' * filled + '-' * (bar_len - filled)
    print(f"\r  [{bar}] {pct:.0f}% {label}", end='', flush=True)
    if current >= total:
        print()


# ── Hash-based caching ──

def file_hash(filepath):
    """SHA256 of a file. Returns None if file doesn't exist."""
    import hashlib
    p = Path(filepath) if not isinstance(filepath, Path) else filepath
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cache(cache_dir):
    """Load pipeline cache from disk."""
    import json
    cache_dir = Path(cache_dir) if not isinstance(cache_dir, Path) else cache_dir
    cache_file = cache_dir / 'pipeline_cache.json'
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache_dir, cache_data):
    """Save pipeline cache to disk."""
    import json
    cache_dir = Path(cache_dir) if not isinstance(cache_dir, Path) else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'pipeline_cache.json'
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2, default=str)


def phase_cache_key(phase_num, input_files, cli_args):
    """Generate a cache key for a phase based on inputs and args."""
    import hashlib, json
    h = hashlib.sha256()
    h.update(str(phase_num).encode())
    for f in input_files:
        fh = file_hash(f)
        h.update((fh or 'missing').encode())
    h.update(json.dumps(sorted(cli_args), default=str).encode())
    return f"phase_{phase_num}_{h.hexdigest()[:16]}"
