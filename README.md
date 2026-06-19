# literature-mining

Batch literature ingestion → bibliometric analysis → research frontier detection → research gap identification.

**One command** to go from a search query or BibTeX file to a structured report of what's hot and what's missing in your field.

## Quick Start

```bash
pip install -r scripts/requirements.txt
python scripts/run_pipeline.py --mode search --query "your research topic" --years 2020-2026
```

This searches OpenAlex, Crossref, and PubMed for up to 200 papers, then runs the full 6-phase analysis pipeline. Results land in `outputs/<timestamp>/report/`.

## Input Modes

| Mode | Command | Best For |
|------|---------|----------|
| Search | `--mode search --query "..."` | Quick exploration of a topic |
| BibTeX | `--mode bibtex --input file.bib` | WoS/Scopus/PubMed exports |
| CSV | `--mode csv --input file.csv` | Custom literature lists with DOIs |
| PDF folder | `--mode pdf --input ./papers/` | Local PDF collections |

## What You Get

```
outputs/<name>/report/
├── summary.txt             # Human-readable overview with confidence estimates
├── report_context.json     # Full structured data for downstream analysis
├── report.md               # Long-form Markdown report
├── vis_data.json           # Keyword co-occurrence network (vis-network compatible)
└── trend_data.json         # Yearly keyword trend data
```

**summary.txt** tells you:
- **Frontiers**: ranked research clusters with scores and keywords (e.g., "chemometrics / FTIR spectroscopy / food authentication")
- **Gaps**: bridge gaps, temporal gaps, and semantic gaps between frontiers — where the unexplored opportunities are
- **Confidence**: per-finding confidence tags (`[HIGH]` / `[MEDIUM]` / `[LOW]`) based on data quality

## How It Works

```
Phase 1: Ingestion    → raw_corpus.jsonl     (API search / BibTeX parse / CSV read)
Phase 2: Cleaning     → clean_corpus.csv      (dedup, stem, keyword extraction)
Phase 3: Analysis     → cooccurrence + trends (keyword frequency, co-occurrence matrix)
Phase 4: Frontiers    → frontiers.json        (Kleinberg burst detection + Louvain clustering)
Phase 5: Gaps         → gaps.json             (density / bridge / temporal / semantic)
Phase 6: Report       → summary.txt + context (structured output)
```

## Pipeline Options

```bash
# Resume after a crash
python scripts/run_pipeline.py --resume-from 4 ...

# Check data quality without full run
python scripts/run_pipeline.py --mode csv --input papers.csv --check-only

# Use enhanced keyword extraction (requires: pip install keybert)
python scripts/run_pipeline.py --mode csv --input papers.csv --enhanced-kw

# Chinese search
python scripts/run_pipeline.py --mode search --query "人工生命" --lang zh
```

## Requirements

- Python 3.9+
- Core: `pandas numpy networkx scikit-learn nltk requests pypdf`
- Recommended: `python-louvain` (better community detection)
- Optional: `keybert` (enhanced keywords), `sentence-transformers` (embedding dedup), `pdfplumber` (PDF text extraction)

## Limitations

- Search mode uses public APIs (OpenAlex/Crossref/PubMed). No API key needed, but rate limits apply.
- Density gap detection requires ≥50 papers for reliable results.
- Best results with ≥50 papers and ≥5-year time span.
