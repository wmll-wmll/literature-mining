#!/usr/bin/env python3
"""
Phase 1: Literature Ingestion
==============================
Ingest literature from multiple sources and output normalized JSONL.

Sources:
  - bibtex   : Parse .bib files into structured metadata
  - ris      : Parse .ris (EndNote/Zotero export) files
  - pdf      : Extract metadata from PDF files in a directory
  - search   : Query Semantic Scholar / Crossref APIs
  - csv      : Read a CSV with DOI/title columns, enrich via Crossref
  - paste    : (handled by the orchestrating skill, not this script)

Output: raw_corpus.jsonl — one JSON object per line, each a paper metadata dict.

Usage:
  python ingest.py --mode bibtex --input nanodelivery.bib --output outputs/raw_corpus.jsonl
  python ingest.py --mode pdf --input ./papers/ --output outputs/raw_corpus.jsonl
  python ingest.py --mode search --query "mRNA lipid nanoparticle delivery" --max 100 --output outputs/raw_corpus.jsonl
  python ingest.py --mode ris --input export.ris --output outputs/raw_corpus.jsonl
  python ingest.py --mode csv --input papers.csv --output outputs/raw_corpus.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, progress, DEFAULTS


# ----------------------------------------------------------
#  Helper utilities
# ----------------------------------------------------------

def safe_get(d, *keys):
    """Return first non-empty value from dict for given keys."""
    for k in keys:
        v = d.get(k, "")
        if v and str(v).strip():
            return str(v).strip()
    return ""


def normalize_author(name):
    """Normalize a single author name to 'Last, First' format."""
    name = re.sub(r'\s+', ' ', name.strip().strip(','))
    # Already "Last, First"
    if ',' in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) >= 2:
            return f"{parts[0]}, {' '.join(parts[1:])}"
        return name
    # "First Last" -> "Last, First"
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    # "First Middle Last" -> "Last, First Middle"
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def extract_doi(text):
    """Extract DOI from arbitrary text using regex."""
    if not text:
        return ""
    doi_pattern = r'\b(10\.\d{4,}/[^\s<>"\'\[\]]+)\b'
    match = re.search(doi_pattern, text, re.IGNORECASE)
    return match.group(1).rstrip('.') if match else ""


def extract_year(text):
    """Extract a 4-digit year (1900-2099) from text."""
    if not text:
        return ""
    match = re.search(r'\b(19|20)\d{2}\b', str(text))
    return match.group(0) if match else ""


# ----------------------------------------------------------
#  BibTeX parser (pure Python, no external dependency)
# ----------------------------------------------------------

def parse_bibtex(filepath):
    """
    Parse a .bib file into a list of entry dicts.
    Handles @article, @inproceedings, @book, @incollection, @phdthesis, @misc.
    Returns list of dicts with keys: id, entry_type, title, authors, year,
      journal, booktitle, volume, pages, doi, abstract, keywords, publisher.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Remove comments (lines starting with % outside entries)
    content = re.sub(r'(?m)^\s*%.*$', '', content)

    entries = []

    # Find all @entry_type{... blocks with proper brace-depth tracking
    # Match @type{entry_id,
    entry_starts = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+)\s*,', content))

    for idx, m in enumerate(entry_starts):
        entry_type = m.group(1).lower()
        entry_id = m.group(2).strip()

        # Find the matching closing brace for this entry
        body_start = m.end()
        brace_depth = 1
        i = body_start
        while i < len(content) and brace_depth > 0:
            if content[i] == '{':
                brace_depth += 1
            elif content[i] == '}':
                brace_depth -= 1
            i += 1

        body = content[body_start:i - 1]  # exclude the closing brace

        fields = parse_bibtex_fields(body)
        fields['id'] = entry_id
        fields['entry_type'] = entry_type
        entries.append(fields)

    return entries


def parse_bibtex_fields(body):
    """Parse the field=value pairs inside a BibTeX entry body.
    Handles nested braces in values like {Blood {Brain} Barrier}.
    """
    fields = {}

    # Match field_name = value blocks
    # Pattern: fieldname = {value_with_nested_braces} or fieldname = "value"
    # We need to track brace depth for {} values
    pos = 0
    while pos < len(body):
        # Look for next field name
        m = re.match(r'\s*(\w+)\s*=\s*', body[pos:])
        if not m:
            pos += 1
            continue

        key = m.group(1).lower()
        pos += m.end()

        # Read the value: either {nested} or "quoted"
        if pos >= len(body):
            break

        if body[pos] == '{':
            # Track brace depth to find matching }
            val_start = pos + 1
            depth = 1
            pos = val_start
            while pos < len(body) and depth > 0:
                if body[pos] == '{':
                    depth += 1
                elif body[pos] == '}':
                    depth -= 1
                pos += 1
            value = body[val_start:pos - 1].strip()
            # Clean up: collapse whitespace, remove trailing comma
            value = re.sub(r'\s+', ' ', value).strip().rstrip(',').strip()
            fields[key] = value
        elif body[pos] == '"':
            # Quoted value — find matching "
            val_start = pos + 1
            pos = body.find('"', val_start)
            if pos < 0:
                break
            value = body[val_start:pos].strip()
            pos += 1
            value = re.sub(r'\s+', ' ', value).strip().rstrip(',').strip()
            fields[key] = value
        else:
            # Numeric or bare value
            m2 = re.match(r'(\d+)\s*,?', body[pos:])
            if m2:
                fields[key] = m2.group(1)
                pos += m2.end()
            else:
                pos += 1
            continue

        # Skip trailing comma and whitespace
        if pos < len(body) and body[pos] == ',':
            pos += 1

    # Normalize known fields
    result = {
        'title': safe_get(fields, 'title'),
        'authors': safe_get(fields, 'author'),
        'year': safe_get(fields, 'year'),
        'journal': safe_get(fields, 'journal', 'journaltitle'),
        'booktitle': safe_get(fields, 'booktitle'),
        'volume': safe_get(fields, 'volume'),
        'number': safe_get(fields, 'number'),
        'pages': safe_get(fields, 'pages'),
        'doi': safe_get(fields, 'doi'),
        'abstract': safe_get(fields, 'abstract'),
        'keywords': safe_get(fields, 'keywords', 'keyword'),
        'publisher': safe_get(fields, 'publisher'),
        'url': safe_get(fields, 'url'),
        'raw': fields,  # keep original for debugging
    }

    # Parse author list
    if result['authors']:
        authors = [normalize_author(a.strip()) for a in result['authors'].split(' and ')]
        result['authors'] = authors
    else:
        result['authors'] = []

    # Try to extract year from other fields
    if not result['year']:
        result['year'] = extract_year(body)

    # Try to extract DOI from URL or other fields
    if not result['doi']:
        result['doi'] = extract_doi(fields.get('url', ''))
        if not result['doi']:
            result['doi'] = extract_doi(body)

    # Keywords as list
    if result['keywords']:
        result['keywords'] = [k.strip().lower() for k in re.split(r'[;,]\s*', result['keywords']) if k.strip()]
    else:
        result['keywords'] = []

    return result


# ----------------------------------------------------------
#  RIS parser
# ----------------------------------------------------------

RIS_TYPE_MAP = {
    'TY  - JOUR': 'article',
    'TY  - CONF': 'inproceedings',
    'TY  - BOOK': 'book',
    'TY  - CHAP': 'incollection',
    'TY  - THES': 'phdthesis',
    'TY  - GEN': 'misc',
}

RIS_FIELD_MAP = {
    'T1': 'title',
    'TI': 'title',
    'AU': 'author',
    'A1': 'author',
    'PY': 'year',
    'Y1': 'year',
    'JO': 'journal',
    'JF': 'journal',
    'JA': 'journal_abbrev',
    'VL': 'volume',
    'IS': 'number',
    'SP': 'pages_start',
    'EP': 'pages_end',
    'DO': 'doi',
    'AB': 'abstract',
    'N2': 'abstract',
    'KW': 'keywords',
    'PB': 'publisher',
    'UR': 'url',
}


def parse_ris(filepath):
    """Parse a .ris file into a list of entry dicts."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    entries = []
    current = {}
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('ER  -'):
            # End of record
            if current:
                entries.append(_normalize_ris(current))
            current = {}
            continue
        if len(line) >= 6 and line[2:4] == '  ':
            tag = line[:2]
            value = line[6:].strip()
            if tag in RIS_FIELD_MAP:
                key = RIS_FIELD_MAP[tag]
                if key in current:
                    if isinstance(current[key], list):
                        current[key].append(value)
                    else:
                        current[key] = [current[key], value]
                else:
                    current[key] = value
            # Store entry type
            if line.startswith('TY  -'):
                current['entry_type'] = RIS_TYPE_MAP.get(line, 'misc')

    if current:
        entries.append(_normalize_ris(current))

    return entries


def _normalize_ris(raw):
    """Normalize a RIS record to standard format."""
    authors = raw.get('author', [])
    if isinstance(authors, str):
        authors = [authors]
    authors = [normalize_author(a) for a in authors]

    pages = raw.get('pages_start', '')
    if raw.get('pages_end'):
        pages += f"-{raw['pages_end']}"

    keywords = raw.get('keywords', [])
    if isinstance(keywords, str):
        keywords = [k.strip().lower() for k in re.split(r'[;,]\s*', keywords) if k.strip()]

    return {
        'id': raw.get('id', f"ris_{hash(raw.get('title','')+raw.get('doi',''))}"),
        'entry_type': raw.get('entry_type', 'article'),
        'title': raw.get('title', ''),
        'authors': authors,
        'year': raw.get('year', ''),
        'journal': raw.get('journal', ''),
        'volume': raw.get('volume', ''),
        'number': raw.get('number', ''),
        'pages': pages,
        'doi': raw.get('doi', ''),
        'abstract': raw.get('abstract', ''),
        'keywords': keywords,
        'publisher': raw.get('publisher', ''),
        'url': raw.get('url', ''),
        'source': 'ris',
    }


# ----------------------------------------------------------
#  PDF metadata extractor
# ----------------------------------------------------------

def extract_pdf_metadata(filepath):
    """
    Extract metadata from a single PDF file.
    Uses pypdf for metadata + pdfplumber for full-text abstract fallback.
    Caches extracted text to outputs/pdf_cache/.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return {'error': 'pypdf/PyPDF2 not installed', 'file': str(filepath)}

    try:
        reader = PdfReader(filepath)
        info = reader.metadata or {}
        first_page_text = ""
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text() or ""

        # Full-text abstract: try pdfplumber for first 2 pages
        full_text = ""
        try:
            import pdfplumber
            cache_dir = Path('outputs/pdf_cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{Path(filepath).stem}.txt"
            if cache_file.exists():
                full_text = cache_file.read_text(encoding='utf-8', errors='replace')[:5000]
            else:
                with pdfplumber.open(filepath) as pdf:
                    pages_text = []
                    for page in pdf.pages[:2]:
                        t = page.extract_text()
                        if t:
                            pages_text.append(t)
                    full_text = '\n'.join(pages_text)[:5000]
                cache_file.write_text(full_text, encoding='utf-8', errors='replace')
        except ImportError:
            full_text = first_page_text[:2000]
        except Exception:
            full_text = first_page_text[:2000]

        # Use pdfplumber text as abstract if available, otherwise metadata
        abstract = full_text if len(full_text) > 100 else first_page_text[:2000]

        # Extract DOI from metadata or full text
        doi = extract_doi(str(info.get('/doi', '')))
        if not doi:
            doi = extract_doi(full_text) or extract_doi(first_page_text)

        # Title from metadata or first substantive line
        title = str(info.get('/title', '')).strip()
        if not title:
            lines = [l.strip() for l in (full_text or first_page_text).split('\n') if len(l.strip()) > 20]
            title = lines[0][:300] if lines else ''

        year = extract_year(str(info.get('/ModDate', ''))) or extract_year(full_text)
        author_str = str(info.get('/author', '')).strip()
        authors = [normalize_author(a) for a in re.split(r'[;,]', author_str) if a.strip()] if author_str else []

        return {
            'id': f"pdf_{hash(filepath)}",
            'entry_type': 'article',
            'title': title[:500],
            'authors': authors,
            'year': year,
            'journal': '',
            'volume': '', 'pages': '',
            'doi': doi,
            'abstract': abstract,
            'keywords': [],
            'source': 'pdf',
            'file': str(filepath),
            'extraction_quality': 'high' if (doi or (title and len(title) > 20)) else 'low',
        }
    except Exception as e:
        return {'error': str(e), 'file': str(filepath), 'extraction_quality': 'failed'}


def ingest_pdf_folder(folder_path):
    """Process all PDFs in a folder."""
    folder = Path(folder_path)
    pdf_files = list(folder.glob('**/*.pdf'))
    results = []
    for i, pdf_file in enumerate(pdf_files):
        print(f"  [{i+1}/{len(pdf_files)}] Extracting: {pdf_file.name}")
        meta = extract_pdf_metadata(pdf_file)
        results.append(meta)
        time.sleep(0.1)  # slight delay to not hammer disk
    return results


# ----------------------------------------------------------
#  Semantic Scholar API search
# ----------------------------------------------------------

def search_semantic_scholar(query, max_results=100, year_start=None, year_end=None):
    """
    Search Semantic Scholar API (no key required) with retry.
    year_start/year_end pushed to query string (API has no native year filter).
    """
    import requests

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    results = []
    limit = min(max_results, 100)
    offset = 0
    fields = "title,authors,year,venue,abstract,externalIds,citationCount,publicationTypes"

    while len(results) < max_results:
        params = {'query': query, 'limit': limit, 'offset': offset, 'fields': fields}

        def _fetch():
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data

        data, err = retry_api(_fetch, max_retries=3, base_delay=2, rate_limit_delay=15)
        if data is None:
            if err:
                warn(f"Semantic Scholar API failed after retries: {err}")
            break

        papers = data.get('data', [])
        if not papers:
            break

        for paper in papers:
            authors = []
            for a in paper.get('authors', []):
                authors.append(normalize_author(a.get('name', '')))

            external_ids = paper.get('externalIds', {}) or {}
            results.append({
                'id': paper.get('paperId', ''),
                'entry_type': 'article',
                'title': paper.get('title', ''),
                'authors': authors,
                'year': int(paper.get('year')) if paper.get('year') else None,
                'journal': paper['venue'].get('name', '') if isinstance(paper.get('venue'), dict) else (paper.get('venue', '') or ''),
                'volume': '', 'pages': '',
                'doi': external_ids.get('DOI', ''),
                'abstract': paper.get('abstract', ''),
                'keywords': [],
                'citation_count': paper.get('citationCount', 0),
                'source': 'semantic_scholar',
                'paper_id': paper.get('paperId', ''),
            })

        offset += limit
        if offset >= data.get('total', 0):
            break
        time.sleep(1)

    return results


def search_crossref(query, max_results=100, year_start=None, year_end=None):
    """
    Search Crossref API (no key required, generous rate limits).
    Now the PRIMARY search API (Semantic Scholar is fallback).
    """
    import requests

    base_url = "https://api.crossref.org/works"
    results = []
    rows = min(max_results, 100)
    offset = 0

    # Build filter string
    filters = ['type:journal-article']
    if year_start:
        filters.append(f'from-pub-date:{year_start}-01-01')
    if year_end:
        filters.append(f'until-pub-date:{year_end}-12-31')
    filter_str = ','.join(filters)

    while len(results) < max_results:
        params = {
            'query': query,
            'rows': rows,
            'offset': offset,
            'filter': filter_str,
        }

        def _fetch():
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data, err = retry_api(_fetch, max_retries=3, base_delay=2, rate_limit_delay=20)
        if data is None:
            break

        items = data.get('message', {}).get('items', [])
        if not items:
            break

        for item in items:
            authors = []
            for a in item.get('author', []):
                family = a.get('family', '')
                given = a.get('given', '')
                authors.append(f"{family}, {given}" if family else given)

            date_parts = item.get('issued', {}).get('date-parts', [[]])[0]
            year = int(date_parts[0]) if date_parts else None

            results.append({
                'id': f"crossref_{item.get('DOI', '')}",
                'entry_type': 'article',
                'title': ' '.join(item.get('title', [''])),
                'authors': authors,
                'year': year,
                'journal': ' '.join(item.get('container-title', [''])),
                'volume': '', 'pages': '',
                'doi': item.get('DOI', ''),
                'abstract': item.get('abstract', ''),
                'keywords': item.get('subject', []),
                'citation_count': 0,
                'source': 'crossref_search',
            })

        offset += rows
        total = data.get('message', {}).get('total-results', 0)
        if offset >= total:
            break
        time.sleep(0.5)

    return results


def search_multi_api(query, max_results=100, lang='en', year_start=None, year_end=None):
    """
    Multi-API search: OpenAlex (primary, best abstract coverage) +
    Crossref (fallback, good metadata). Semantic Scholar last (best quality, heavy rate limits).

    For Chinese queries (lang='zh'), translates and searches.
    """
    log('info', f'Searching: "{query}" (lang={lang})')

    # Chinese → English translation stub
    if lang == 'zh':
        query_en = _translate_zh_en(query)
        log('info', f'Translated query: "{query_en}"')
        query = query_en

    # OpenAlex first (best abstract coverage, full-text indexed)
    log('info', 'Trying OpenAlex (primary — best abstract coverage)...')
    results = _search_openalex(query, max_results, year_start=year_start, year_end=year_end)
    n_abstracts = sum(1 for r in results if r.get('abstract'))
    if results:
        log('ok', f'OpenAlex returned {len(results)} results ({n_abstracts} with abstracts)')
        return results
    warn('OpenAlex returned no results')

    # Crossref as fallback (good metadata, year filter pushdown)
    log('info', 'Trying Crossref (fallback)...')
    results = search_crossref(query, max_results, year_start=year_start, year_end=year_end)
    n_abstracts = sum(1 for r in results if r.get('abstract'))
    if results:
        log('ok', f'Crossref returned {len(results)} results ({n_abstracts} with abstracts)')
        return results
    warn('Crossref returned no results')

    # Semantic Scholar last resort
    log('info', 'Trying Semantic Scholar (last resort)...')
    results = search_semantic_scholar(query, max_results, year_start=year_start, year_end=year_end)
    n_abstracts = sum(1 for r in results if r.get('abstract'))
    if results:
        log('ok', f'Semantic Scholar returned {len(results)} results ({n_abstracts} with abstracts)')
        return results
    # PubMed last resort (biomedical focus)
    log('info', 'Trying PubMed (last resort)...')
    results = search_pubmed(query, max_results, year_start=year_start, year_end=year_end)
    n_abstracts = sum(1 for r in results if r.get('abstract'))
    if results:
        log('ok', f'PubMed returned {len(results)} results ({n_abstracts} with abstracts)')
        return results

    warn('All APIs exhausted')

def _search_openalex(query, max_results=100, year_start=None, year_end=None):
    """
    Search OpenAlex API (no key, best abstract coverage, full-text indexed).
    Returns list of paper metadata dicts.
    """
    import requests

    base_url = "https://api.openalex.org/works"
    results = []
    per_page = min(max_results, 50)
    page = 1

    # Build filter: OpenAlex uses range format for years
    if year_start and year_end:
        year_filter = f'publication_year:{year_start}-{year_end}'
        filter_str = year_filter
    elif year_start:
        filter_str = f'publication_year:{year_start}'
    elif year_end:
        filter_str = f'publication_year:{year_end}'
    else:
        filter_str = None

    while len(results) < max_results:
        params = {
            'search': query,
            'per_page': per_page,
            'page': page,
        }
        if filter_str:
            params['filter'] = filter_str

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            warn(f'OpenAlex request failed: {e}')
            break

        items = data.get('results', [])
        if not items:
            break

        for item in items:
            if not isinstance(item, dict):
                continue
            authors = []
            for a in (item.get('authorships') or []):
                auth = a.get('author') if isinstance(a, dict) else {}
                name = auth.get('display_name', '') if isinstance(auth, dict) else ''
                authors.append(normalize_author(name) if name else '')

            # Safe nested dict access
            primary = item.get('primary_location') or {}
            source = (primary.get('source') or {}) if isinstance(primary, dict) else {}
            journal = source.get('display_name', '') if isinstance(source, dict) else ''

            ids = item.get('ids') or {}
            doi = ids.get('doi', '') if isinstance(ids, dict) else ''

            abstract = ''
            inv = item.get('abstract_inverted_index')
            if inv and isinstance(inv, dict):
                abstract = _reconstruct_inverted_abstract(inv)

            # Author keywords (primary — what the paper is actually about)
            # Filter out L0/L1 noise that OpenAlex sometimes puts in the keywords field
            _BROAD_KW = {
                'chemistry', 'biology', 'physics', 'mathematics', 'engineering',
                'computer science', 'medicine', 'materials science', 'food science',
                'environmental science', 'business', 'economics', 'sociology',
                'psychology', 'philosophy', 'geography', 'geology', 'history',
                'political science', 'law', 'art', 'education', 'linguistics',
                'nanotechnology', 'biotechnology', 'bioinformatics',
                'chemical engineering', 'mechanical engineering', 'electrical engineering',
                'biochemistry', 'molecular biology', 'cell biology', 'genetics',
                'analytical chemistry', 'organic chemistry', 'physical chemistry',
                'science', 'technology', 'data science', 'machine learning',  # ML is too broad alone
            }
            author_kw = []
            for kw_obj in (item.get('keywords') or []):
                if isinstance(kw_obj, dict):
                    kw = (kw_obj.get('keyword') or kw_obj.get('display_name') or '').strip().lower()
                else:
                    kw = str(kw_obj).strip().lower()
                if kw and kw not in _BROAD_KW and len(kw) > 2:
                    author_kw.append(kw)

            # Concepts (supplement — OpenAlex auto-tags, L2+ only, deduped vs author KWs)
            # L0 = broad discipline ("Chemistry"), L1 = sub-discipline ("Food science")
            # L2+ = specific technique/topic ("Near-infrared spectroscopy", "Chemometrics")
            author_set = set(author_kw)
            concept_kw = []
            for c in (item.get('concepts') or []):
                if isinstance(c, dict) and c.get('level', 0) >= 2:
                    kw = c.get('display_name', '').strip().lower()
                    if kw and kw not in author_set:
                        concept_kw.append(kw)
                        author_set.add(kw)
                if len(concept_kw) >= 5:
                    break

            # Merge: author KWs first, concepts fill up to 8 total
            keywords = author_kw + concept_kw
            keywords = keywords[:max(len(author_kw), min(8, len(author_kw) + 3))]

            results.append({
                'id': item.get('id', '').split('/')[-1] if item.get('id') else '',
                'entry_type': 'article',
                'title': item.get('title', '') or '',
                'authors': [a for a in authors if a],
                'year': item.get('publication_year'),
                'journal': journal,
                'volume': '', 'pages': '',
                'doi': doi,
                'abstract': abstract,
                'keywords': keywords,
                'citation_count': item.get('cited_by_count', 0),
                'source': 'openalex',
                'paper_id': item.get('id', ''),
            })

        page += 1
        if page > 4:  # Max 4 pages (200 results) to avoid long waits
            break
        time.sleep(0.5)

    return results


def _reconstruct_inverted_abstract(inv):
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inv:
        return ''
    # Build (position, word) pairs
    positions = []
    for word, pos_list in inv.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return ' '.join(w for _, w in positions)

    return []


def search_pubmed(query, max_results=100, year_start=None, year_end=None):
    """
    Search PubMed via Entrez API (no key, biomedical focus).
    Returns list of paper metadata dicts.
    """
    import requests
    import xml.etree.ElementTree as ET

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    results = []

    # Build year range query
    if year_start and year_end:
        query = f"({query}) AND ({year_start}:{year_end}[dp])"
    elif year_start:
        query = f"({query}) AND ({year_start}:3000[dp])"

    # Step 1: Search for IDs
    try:
        search_params = {
            'db': 'pubmed', 'term': query, 'retmax': min(max_results, 100),
            'retmode': 'json', 'sort': 'relevance',
        }
        resp = requests.get(f"{base_url}/esearch.fcgi", params=search_params, timeout=15)
        resp.raise_for_status()
        id_list = resp.json().get('esearchresult', {}).get('idlist', [])
    except Exception as e:
        warn(f'PubMed search failed: {e}')
        return results

    if not id_list:
        return results

    # Step 2: Fetch details
    try:
        fetch_params = {
            'db': 'pubmed', 'id': ','.join(id_list[:max_results]),
            'retmode': 'xml', 'rettype': 'abstract',
        }
        resp = requests.get(f"{base_url}/efetch.fcgi", params=fetch_params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        warn(f'PubMed fetch failed: {e}')
        return results

    for article in root.findall('.//PubmedArticle'):
        try:
            title = ''
            title_el = article.find('.//ArticleTitle')
            if title_el is not None:
                title = title_el.text or ''

            abstract = ''
            abstract_el = article.find('.//Abstract/AbstractText')
            if abstract_el is not None:
                abstract = abstract_el.text or ''

            authors = []
            for author_el in article.findall('.//Author'):
                last = author_el.findtext('LastName', '')
                fore = author_el.findtext('ForeName', '')
                if last:
                    authors.append(f"{last}, {fore}" if fore else last)

            year = ''
            date_el = article.find('.//PubDate/Year')
            if date_el is not None:
                year = date_el.text or ''

            journal = ''
            journal_el = article.find('.//Journal/Title')
            if journal_el is not None:
                journal = journal_el.text or ''

            doi = ''
            for eid in article.findall('.//ELocationID'):
                if eid.get('EIdType') == 'doi' and eid.text:
                    doi = eid.text

            keywords = []
            for kw_el in article.findall('.//Keyword'):
                if kw_el.text:
                    keywords.append(kw_el.text.lower())

            pmid = article.findtext('.//PMID', '')

            results.append({
                'id': f"pubmed_{pmid}",
                'entry_type': 'article',
                'title': title[:500],
                'authors': authors,
                'year': int(year) if year and year.isdigit() else None,
                'journal': journal,
                'volume': '', 'pages': '',
                'doi': doi,
                'abstract': abstract,
                'keywords': keywords,
                'citation_count': 0,
                'source': 'pubmed',
                'paper_id': pmid,
            })
        except Exception:
            continue

    log('ok', f'PubMed returned {len(results)} results')
    return results


def _translate_zh_en(text):
    """Simple Chinese->English translation for search queries."""
    # Domain-specific translations
    domain_map = {
        '人工生命': 'artificial life',
        '数字生命': 'digital life',
        '合成生物学': 'synthetic biology',
        '合成细胞': 'synthetic cell',
        '原细胞': 'protocell',
        '最小基因组': 'minimal genome',
        '开放式进化': 'open-ended evolution',
        '细胞自动机': 'cellular automata',
        '纳米递送': 'nanoparticle delivery',
        '药物递送': 'drug delivery',
        '基因编辑': 'gene editing',
        '基因治疗': 'gene therapy',
        '机器学习': 'machine learning',
        '深度学习': 'deep learning',
        '人工智能': 'artificial intelligence',
        '蛋白冠': 'protein corona',
        '细胞外囊泡': 'extracellular vesicle',
        '肿瘤微环境': 'tumor microenvironment',
        '血脑屏障': 'blood brain barrier',
    }
    for zh, en in domain_map.items():
        if zh in text:
            text = text.replace(zh, en)
    return text


# ----------------------------------------------------------
#  Crossref API enrichment (for CSV/Paste inputs)
# ----------------------------------------------------------

def enrich_via_crossref_batch(doi_list):
    """
    Batch lookup multiple DOIs via Crossref API in one request.
    doi_list: list of DOIs (max 50 per batch recommended).
    Returns dict {doi: metadata_dict or None}.
    """
    import requests

    if not doi_list:
        return {}

    # Filter valid DOIs
    valid_dois = [d for d in doi_list if d and d.lower() != 'nan']
    if not valid_dois:
        return {}

    result = {}
    # Batch in groups of 50 (Crossref limit)
    for batch_start in range(0, len(valid_dois), 50):
        batch = valid_dois[batch_start:batch_start + 50]
        filter_dois = ','.join(f'doi:{d}' for d in batch)
        url = f"https://api.crossref.org/works?filter={filter_dois}&rows={len(batch)}"

        def _fetch():
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data, err = retry_api(_fetch, max_retries=2, base_delay=1, rate_limit_delay=10)
        if data is None:
            for doi in batch:
                result[doi] = None
            continue

        items = data.get('message', {}).get('items', [])
        # Build DOI → item map
        found = {}
        for item in items:
            item_doi = item.get('DOI', '').lower()
            if item_doi:
                found[item_doi] = item

        for doi in batch:
            item = found.get(doi.lower())
            if not item:
                result[doi] = None
                continue
            authors = []
            for a in item.get('author', []):
                family = a.get('family', '')
                given = a.get('given', '')
                authors.append(f"{family}, {given}" if family else given)
            date_parts = item.get('published-print', {}).get('date-parts', [[]])[0]
            if not date_parts:
                date_parts = item.get('created', {}).get('date-parts', [[]])[0]
            year = int(date_parts[0]) if date_parts else None
            result[doi] = {
                'id': f"crossref_{item.get('DOI', '')}",
                'entry_type': item.get('type', 'article'),
                'title': ' '.join(item.get('title', [''])),
                'authors': authors,
                'year': year,
                'journal': ' '.join(item.get('container-title', [''])),
                'volume': item.get('volume', ''),
                'pages': item.get('page', ''),
                'doi': item.get('DOI', ''),
                'abstract': item.get('abstract', ''),
                'keywords': item.get('subject', []),
                'publisher': item.get('publisher', ''),
                'url': item.get('URL', ''),
                'source': 'crossref',
            }

        if batch_start + 50 < len(valid_dois):
            time.sleep(0.3)

    return result


def enrich_via_crossref(doi=None, title=None):
    """
    Look up a paper on Crossref API by DOI or title.
    Returns a metadata dict or None.
    """
    import requests

    if doi:
        url = f"https://api.crossref.org/works/{doi}"
    elif title:
        url = f"https://api.crossref.org/works?query.title={requests.utils.quote(title)}&rows=1"
    else:
        return None

    def _fetch():
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()

    data, err = retry_api(_fetch, max_retries=2, base_delay=1, rate_limit_delay=10)
    if data is None:
        if err:
            warn(f"Crossref lookup failed: {err[:100]}")
        return None

    try:
        if doi:
            item = data.get('message', {})
        else:
            items = data.get('message', {}).get('items', [])
            item = items[0] if items else {}

        if not item:
            return None

        authors = []
        for a in item.get('author', []):
            family = a.get('family', '')
            given = a.get('given', '')
            authors.append(f"{family}, {given}" if family else given)

        date_parts = item.get('published-print', {}).get('date-parts', [[]])[0]
        if not date_parts:
            date_parts = item.get('created', {}).get('date-parts', [[]])[0]
        year = str(date_parts[0]) if date_parts else ''

        return {
            'id': f"crossref_{item.get('DOI', '')}",
            'entry_type': item.get('type', 'article'),
            'title': ' '.join(item.get('title', [''])),
            'authors': authors,
            'year': int(year) if year else None,
            'journal': ' '.join(item.get('container-title', [''])),
            'journal': ' '.join(item.get('container-title', [''])),
            'volume': item.get('volume', ''),
            'pages': item.get('page', ''),
            'doi': item.get('DOI', ''),
            'abstract': item.get('abstract', ''),
            'keywords': item.get('subject', []),
            'publisher': item.get('publisher', ''),
            'url': item.get('URL', ''),
            'source': 'crossref',
        }
    except Exception as e:
        warn(f"Crossref parse error: {e}")
        return None


def ingest_csv(filepath):
    """
    Read CSV file with paper metadata. Batch-enriches via Crossref for rows
    that lack abstracts. Rows with good data skip the API entirely.
    Expected columns: doi, title, authors, year, journal, abstract, keywords.
    """
    import csv
    try:
        import pandas as pd
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    except ImportError:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            df = list(reader)

    results = []
    if hasattr(df, 'to_dict'):
        records = df.to_dict('records')
    else:
        records = df

    # Phase 1: build base metadata, split into "rich" (skip API) and "needs enrichment"
    base_entries = []   # (index, base_meta, needs_enrichment)
    dois_to_enrich = []

    for i, row in enumerate(records):
        doi = str(row.get('doi') or row.get('DOI') or '').strip()
        title = str(row.get('title') or row.get('Title') or row.get('TITLE') or '').strip()
        authors_str = str(row.get('authors') or row.get('author') or '').strip()
        year = str(row.get('year') or row.get('Year') or '').strip()
        journal = str(row.get('journal') or row.get('Journal') or '').strip()
        abstract = str(row.get('abstract') or row.get('Abstract') or '').strip()
        keywords = str(row.get('keywords') or row.get('Keywords') or '').strip()

        base_meta = {
            'id': f"csv_{i}",
            'entry_type': 'article',
            'title': title,
            'authors': [normalize_author(a.strip()) for a in re.split(r'[;,]', authors_str) if a.strip()] if authors_str else [],
            'year': int(year) if year and year.isdigit() else None,
            'journal': journal,
            'volume': '', 'pages': '',
            'doi': doi,
            'abstract': abstract,
            'keywords': [k.strip().lower() for k in re.split(r'[;,]\s*', keywords) if k.strip()] if keywords else [],
            'source': 'csv',
        }

        # Skip API if CSV already has good data
        has_abstract = abstract and len(abstract) > 100
        has_keywords = bool(base_meta['keywords'])
        has_doi = doi and doi.lower() != 'nan'

        if has_abstract and has_keywords:
            base_entries.append((i, base_meta, False))
        elif has_doi:
            base_entries.append((i, base_meta, True))
            dois_to_enrich.append(doi)
        else:
            # No DOI and no good data — keep as-is
            base_entries.append((i, base_meta, False))

    # Phase 2: batch-enrich only the DOIs that need it
    n_skip = sum(1 for _, _, needs in base_entries if not needs)
    n_enrich = len(dois_to_enrich)
    print(f"  CSV: {len(records)} rows, {n_skip} skip API, {n_enrich} need enrichment")

    enriched_map = {}
    if dois_to_enrich:
        print(f"  Batch-enriching {n_enrich} DOIs via Crossref...")
        enriched_map = enrich_via_crossref_batch(dois_to_enrich)
        n_got = sum(1 for v in enriched_map.values() if v is not None)
        print(f"  Batch complete: {n_got}/{n_enrich} resolved")

    # Phase 3: assemble results
    for i, base_meta, needs in base_entries:
        if not needs:
            results.append(base_meta)
        else:
            doi = base_meta['doi']
            enriched = enriched_map.get(doi)
            if enriched:
                enriched['keywords'] = base_meta['keywords'] if base_meta['keywords'] else enriched.get('keywords', [])
                if not enriched.get('abstract') and base_meta['abstract']:
                    enriched['abstract'] = base_meta['abstract']
                if not enriched.get('title') and base_meta['title']:
                    enriched['title'] = base_meta['title']
                results.append(enriched)
            else:
                results.append(base_meta)

    return results


# ----------------------------------------------------------
#  Citation chasing / corpus expansion
# ----------------------------------------------------------

def expand_corpus(seed_jsonl, max_expand=50):
    """
    Expand a corpus by citation chasing from seed papers.
    Uses Semantic Scholar API to find references and citations.
    """
    import requests

    # Load seed papers
    with open(seed_jsonl, 'r', encoding='utf-8') as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    # Collect seed paper IDs
    paper_ids = [s.get('paper_id') or s.get('id') for s in seeds if s.get('paper_id') or s.get('id')]
    dois = [s.get('doi') for s in seeds if s.get('doi') and s.get('doi').lower() != 'nan']

    log('info', f'Expanding from {len(seeds)} seed papers')

    expanded = []
    seen_ids = set()
    seen_dois = set()

    # Mark seeds as seen
    for pid in paper_ids:
        if pid and not pid.startswith('csv_') and not pid.startswith('pdf_'):
            seen_ids.add(pid)
    for doi in dois:
        seen_dois.add(doi.lower())

    fields = "title,authors,year,venue,abstract,externalIds,citationCount"

    for i, seed in enumerate(seeds):
        pid = seed.get('paper_id') or seed.get('id', '')
        if not pid or pid.startswith('csv_') or pid.startswith('pdf_'):
            continue

        progress(i + 1, len(seeds), f'chasing {pid[:20]}')
        if len(expanded) >= max_expand:
            break

        # Get citations (papers that cite this seed)
        for direction, endpoint in [('citations', 'citations'), ('references', 'references')]:
            if len(expanded) >= max_expand:
                break

            url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{endpoint}"
            params = {'limit': 20, 'fields': fields}

            def _fetch():
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()

            data, err = retry_api(_fetch, max_retries=2, base_delay=1, rate_limit_delay=10)
            if data is None:
                continue

            for paper in data.get('data', []):
                pid2 = paper.get('paperId', '')
                doi2 = (paper.get('externalIds') or {}).get('DOI', '')
                if pid2 in seen_ids or (doi2 and doi2.lower() in seen_dois):
                    continue
                if len(expanded) >= max_expand:
                    break

                seen_ids.add(pid2)
                if doi2:
                    seen_dois.add(doi2.lower())

                authors = [normalize_author(a.get('name', '')) for a in paper.get('authors', [])]
                venue = paper.get('venue', {}) or {}

                expanded.append({
                    'id': pid2,
                    'entry_type': 'article',
                    'title': paper.get('title', ''),
                    'authors': authors,
                    'year': int(paper.get('year')) if paper.get('year') else None,
                    'journal': venue.get('name', ''),
                    'volume': '', 'pages': '',
                    'doi': doi2,
                    'abstract': paper.get('abstract', ''),
                    'keywords': [],
                    'citation_count': paper.get('citationCount', 0),
                    'source': f'expand_{direction}',
                    'paper_id': pid2,
                    'seed_paper_id': pid,
                })
            time.sleep(0.5)

    print()  # newline after progress bar
    log('ok', f'Expanded corpus: {len(expanded)} new papers added')
    return expanded


# ----------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------

def main():
    init()  # Windows UTF-8 fix

    parser = argparse.ArgumentParser(description='Literature Ingestion — Phase 1')
    parser.add_argument('--mode', required=True,
                        choices=['bibtex', 'ris', 'pdf', 'search', 'csv', 'expand'],
                        help='Input source type')
    parser.add_argument('--input', '-i', help='Input file or folder path')
    parser.add_argument('--query', '-q', help='Search query (for search mode)')
    parser.add_argument('--max', type=int, default=100, help='Max results for search mode')
    parser.add_argument('--max-expand', type=int, default=50, help='Max papers for expand mode')
    parser.add_argument('--lang', default='en', choices=['en', 'zh'], help='Query language (default: en)')
    parser.add_argument('--years', help='Year range filter pushed to API, e.g. 2020-2026')
    parser.add_argument('--output', '-o', default='outputs/raw_corpus.jsonl',
                        help='Output JSONL file path')
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []

    print("=" * 60)
    print("[INGEST] Phase 1: Literature Ingestion")
    print(f"   Mode: {args.mode}")
    print("=" * 60)

    if args.mode == 'bibtex':
        if not args.input:
            print("[ERROR] BibTeX mode requires --input <file>")
            sys.exit(1)
        print(f"   Parsing: {args.input}")
        entries = parse_bibtex(args.input)
        for e in entries:
            e['source'] = 'bibtex'

    elif args.mode == 'ris':
        if not args.input:
            print("[ERROR] RIS mode requires --input <file>")
            sys.exit(1)
        print(f"   Parsing: {args.input}")
        entries = parse_ris(args.input)

    elif args.mode == 'pdf':
        if not args.input:
            print("[ERROR] PDF mode requires --input <folder>")
            sys.exit(1)
        print(f"   Scanning folder: {args.input}")
        entries = ingest_pdf_folder(args.input)

    elif args.mode == 'search':
        if not args.query:
            print("[ERROR] Search mode requires --query <search terms>")
            sys.exit(1)
        print(f"   Searching Semantic Scholar for: \"{args.query}\"")
        print(f"   Max results: {args.max}")
        yr_start, yr_end = None, None
        if args.years:
            m = re.match(r'(\d{4})\s*[-–]\s*(\d{4})', args.years)
            if m:
                yr_start, yr_end = int(m.group(1)), int(m.group(2))
        entries = search_multi_api(args.query, args.max, lang=args.lang, year_start=yr_start, year_end=yr_end)

    elif args.mode == 'csv':
        if not args.input:
            print("[ERROR] CSV mode requires --input <file>")
            sys.exit(1)
        print(f"   Reading CSV: {args.input}")
        entries = ingest_csv(args.input)

    elif args.mode == 'expand':
        if not args.input:
            print("[ERROR] Expand mode requires --input <seed_jsonl>")
            sys.exit(1)
        max_expand = getattr(args, 'max_expand', 50)
        entries = expand_corpus(args.input, max_expand)

    # Normalize year to int across all entries (Crossref returns int, CSV may be str)
    for e in entries:
        yr = e.get('year')
        if yr is not None and not isinstance(yr, int):
            try:
                e['year'] = int(yr)
            except (ValueError, TypeError):
                e['year'] = None
        if e.get('citation_count') is not None and not isinstance(e.get('citation_count'), int):
            try:
                e['citation_count'] = int(e['citation_count'])
            except (ValueError, TypeError):
                e['citation_count'] = 0

    # Filter out failed extractions (but keep them for logging)
    failed = [e for e in entries if e.get('error') or e.get('extraction_quality') == 'failed']
    valid = [e for e in entries if not e.get('error') and e.get('extraction_quality') != 'failed']

    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Summary
    n_abstract = sum(1 for e in valid if e.get('abstract'))
    n_keywords = sum(1 for e in valid if e.get('keywords'))
    n_doi = sum(1 for e in valid if e.get('doi'))
    years = [e['year'] for e in valid if isinstance(e.get('year'), int)]

    print(f"\n{'-' * 50}")
    print(f"[INGEST] Ingestion complete")
    print(f"   Output: {output_path}")
    print(f"   Total entries:      {len(entries)}")
    print(f"   Valid entries:      {len(valid)}")
    print(f"   Failed:             {len(failed)}")
    if valid:
        print(f"   With abstract:      {n_abstract} ({n_abstract/len(valid)*100:.0f}%)")
        print(f"   With keywords:      {n_keywords} ({n_keywords/len(valid)*100:.0f}%)")
        print(f"   With DOI:           {n_doi} ({n_doi/len(valid)*100:.0f}%)")
        if years:
            print(f"   Year range:         {min(years)}-{max(years)}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
