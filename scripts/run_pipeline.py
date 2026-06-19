#!/usr/bin/env python3
"""
Literature Mining — Unified Pipeline Runner
=============================================
One-command entry point that runs all 6 phases in sequence.

Usage:
  python run_pipeline.py --input papers.bib --mode bibtex
  python run_pipeline.py --input papers.csv --mode csv
  python run_pipeline.py --search "artificial life" --max 50
  python run_pipeline.py --input corpus.jsonl --mode expand --max-expand 50
  python run_pipeline.py --input new_papers.csv --mode incremental --output-dir outputs/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from utils import init, log, warn, error, banner, sep, quality_report, adaptive_thresholds, DEFAULTS, load_cache, save_cache, file_hash, phase_cache_key

# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

PHASE_SCRIPTS = {
    1: 'ingest.py',
    2: 'clean.py',
    3: 'analyze.py',
    4: 'frontier.py',
    5: 'gaps.py',
    6: 'report.py',
}

PHASE_NAMES = {
    1: 'ingest',
    2: 'clean',
    3: 'analyze',
    4: 'frontier',
    5: 'gaps',
    6: 'report',
}


def run_phase(phase_num, args_list, cwd='.'):
    """Run a phase script and return (success, output)."""
    script = SCRIPTS_DIR / PHASE_SCRIPTS[phase_num]
    if not script.exists():
        return False, f"Script not found: {script}"

    cmd = [sys.executable, str(script)] + args_list
    banner(PHASE_NAMES[phase_num], f'Phase {phase_num}: {PHASE_NAMES[phase_num].title()}')

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace',
                                timeout=DEFAULTS['phase_timeout'])
        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)
        if result.returncode != 0:
            warn(f'Phase {phase_num} exited with code {result.returncode}')
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        warn(f'Phase {phase_num} timed out (>{DEFAULTS["phase_timeout"]}s)')
        return False, 'TIMEOUT'
    except Exception as e:
        warn(f'Phase {phase_num} failed: {e}')
        return False, str(e)


def run_phase_cached(phase_num, args_list, input_files, cache_data, output_dir, cwd='.'):
    """Run a phase with caching: skip if inputs unchanged."""
    import json

    # Build cache key from input files + args
    cache_key = phase_cache_key(phase_num, input_files, args_list)

    # Check if this phase was already run with same inputs
    if phase_num in cache_data.get('phases', {}):
        stored = cache_data['phases'][phase_num]
        if stored.get('key') == cache_key:
            # Verify outputs still exist
            outputs_ok = True
            for out_path in stored.get('outputs', []):
                if not Path(out_path).exists():
                    outputs_ok = False
                    break
            if outputs_ok:
                log('info', f'Phase {phase_num} cached (inputs unchanged), skipping')
                return True, '(cached)'

    # Run the phase
    ok, output = run_phase(phase_num, args_list, cwd)

    # Store cache info
    if ok:
        if 'phases' not in cache_data:
            cache_data['phases'] = {}
        cache_data['phases'][phase_num] = {
            'key': cache_key,
            'outputs': [str(Path(output_dir) / f) for f in _phase_outputs(phase_num, output_dir)],
        }
        save_cache(Path(output_dir) / '.cache', cache_data)

    return ok, output


def _phase_outputs(phase_num, output_dir):
    """Return list of expected output filenames for a phase."""
    outputs = {
        1: ['raw_corpus.jsonl'],
        2: ['clean_corpus.csv', 'stats_summary.json'],
        3: ['cooccurrence.csv', 'keyword_freq.csv', 'trends.json', 'stats_summary.json'],
        4: ['frontiers.json'],
        5: ['gaps.json'],
        6: ['report/report_context.json', 'report/report.md'],
    }
    return outputs.get(phase_num, [])


def check_output(filepath, label):
    """Check if an output file exists and is non-empty."""
    p = Path(filepath)
    if p.exists() and p.stat().st_size > 10:
        log('ok', f'{label}: {p} ({p.stat().st_size/1024:.0f} KB)')
        return True
    else:
        warn(f'{label} missing or empty: {p}')
        return False


def run_pipeline(args):
    """Run the full 6-phase pipeline with hash-based caching."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cache
    cache_dir = output_dir / '.cache'
    cache_data = load_cache(cache_dir)

    # File paths
    raw_jsonl = str(output_dir / 'raw_corpus.jsonl')
    clean_csv = str(output_dir / 'clean_corpus.csv')
    cooc_csv = str(output_dir / 'cooccurrence.csv')
    trends_json = str(output_dir / 'trends.json')
    frontiers_json = str(output_dir / 'frontiers.json')
    gaps_json = str(output_dir / 'gaps.json')
    report_dir = str(output_dir / 'report')

    # ── Phase 1: Ingestion ──
    banner('ingest', 'Phase 1: Literature Ingestion')

    ingest_args = ['--mode', args.mode, '--output', raw_jsonl]
    if args.mode == 'search':
        if not args.query:
            error("Search mode requires --query")
            return False
        ingest_args += ['--query', args.query, '--max', str(args.max_results)]
        if args.years:
            ingest_args += ['--years', args.years]
    elif args.mode == 'expand':
        if not args.input:
            error("Expand mode requires --input (seed corpus JSONL)")
            return False
        ingest_args += ['--input', args.input, '--max-expand', str(args.max_expand)]
    else:
        if not args.input:
            error(f"Mode '{args.mode}' requires --input")
            return False
        ingest_args += ['--input', args.input]

    if not args.skip_ingest:
        ok, out = run_phase(1, ingest_args, args.cwd)
        if not ok:
            error('Phase 1 failed, stopping')
            return False
        if not check_output(raw_jsonl, 'Raw corpus'):
            error('Phase 1 produced no output')
            return False
    else:
        log('info', 'Skipping Phase 1 (--skip-1)')

    # ── Diagnostic check after Phase 1 ──
    log('info', 'Running data quality diagnostic...')
    with open(raw_jsonl, 'r', encoding='utf-8') as f:
        papers = [json.loads(line) for line in f if line.strip()]
    n_papers = len(papers)
    years = [int(p['year']) for p in papers if p.get('year') and isinstance(p.get('year'), int)]
    year_span = max(years) - min(years) + 1 if years else 0
    n_keywords = sum(len(p.get('keywords', [])) for p in papers)
    n_with_abstract = sum(1 for p in papers if p.get('abstract') and len(str(p.get('abstract', ''))) > 50)
    abstract_pct = n_with_abstract / max(1, n_papers)

    thresholds = adaptive_thresholds(n_papers, n_keywords, year_span)
    warnings, quality = quality_report(thresholds, abstract_pct=abstract_pct)

    if thresholds.get('recommend_abstract_kw'):
        log('info', 'Keyword density low — TF-IDF extraction from abstracts will be enabled in Phase 2')

    if args.check_only:
        log('info', '--check-only: stopping after diagnostic')
        return True

    # ── Phase 2: Cleaning ──
    clean_args = ['--input', raw_jsonl, '--output', clean_csv]
    if args.years:
        clean_args += ['--years', args.years]
    if not args.skip_clean:
        ok, out = run_phase(2, clean_args, args.cwd)
        if not ok:
            error('Phase 2 failed, stopping')
            return False
        if not check_output(clean_csv, 'Clean corpus'):
            error('Phase 2 produced no output')
            return False
    else:
        log('info', 'Skipping Phase 2 (--skip-2)')

    # ── Phase 3: Analysis ──
    analyze_args = ['--input', clean_csv, '--output-dir', str(output_dir)]
    if not args.skip_analyze:
        ok, out = run_phase(3, analyze_args, args.cwd)
        if not ok:
            error('Phase 3 failed, stopping')
            return False
        if not check_output(cooc_csv, 'Co-occurrence'):
            warn('Co-occurrence file missing — Phase 4 may fail')
    else:
        log('info', 'Skipping Phase 3 (--skip-3)')

    # ── Phase 4: Frontiers ──
    frontier_args = [
        '--input', clean_csv,
        '--cooccurrence', cooc_csv,
        '--trends', trends_json,
        '--output', frontiers_json,
    ]
    if not args.skip_frontier:
        ok, out = run_phase(4, frontier_args, args.cwd)
        if not ok:
            error('Phase 4 failed, stopping')
            return False
        if not check_output(frontiers_json, 'Frontiers'):
            warn('Frontiers file missing — Phase 5 may fail')
    else:
        log('info', 'Skipping Phase 4 (--skip-4)')

    # ── Phase 5: Gaps ──
    gap_args = [
        '--input', clean_csv,
        '--cooccurrence', cooc_csv,
        '--frontiers', frontiers_json,
        '--trends', trends_json,
        '--output', gaps_json,
    ]
    if not args.skip_gaps:
        ok, out = run_phase(5, gap_args, args.cwd)
        if not ok:
            warn('Phase 5 had errors but continuing to report')
        check_output(gaps_json, 'Gaps')
    else:
        log('info', 'Skipping Phase 5 (--skip-5)')

    # ── Phase 6: Report ──
    report_args = [
        '--corpus', clean_csv,
        '--frontiers', frontiers_json,
        '--gaps', gaps_json,
        '--trends', trends_json,
        '--cooccurrence', cooc_csv,
        '--output-dir', report_dir,
    ]
    if not args.skip_report:
        ok, out = run_phase(6, report_args, args.cwd)
        if not ok:
            warn('Phase 6 had errors')
    else:
        log('info', 'Skipping Phase 6 (--skip-6)')

    # ── Final Summary ──
    sep()
    # Save cache
    save_cache(cache_dir, cache_data)

    log('ok', 'Pipeline complete!')
    log('info', f'Outputs: {output_dir}/')
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            log('info', f'  {"[FILE]" if f.suffix != ".html" else "[HTML]"} {f.name}')
    if Path(report_dir).exists():
        log('info', f'  Report: {report_dir}/')
        log('info', f'    report.md — Full Markdown report')
        log('info', f'    index.html — Interactive visualization (open in browser)')
    sep()

    return True


# ──────────────────────────────────────────────────────────
#  Incremental mode
# ──────────────────────────────────────────────────────────

def run_incremental(args):
    """
    Incremental update: add new papers to an existing corpus.
    1. Ingest new papers into a temporary JSONL
    2. Append to existing raw_corpus.jsonl
    3. Re-run Phases 2-6 on the merged corpus
    """
    output_dir = Path(args.output_dir)
    raw_jsonl = str(output_dir / 'raw_corpus.jsonl')

    if not Path(raw_jsonl).exists():
        error('No existing corpus found. Run full pipeline first.')
        error(f'Expected: {raw_jsonl}')
        return False

    # Phase 1: ingest new papers only
    tmp_jsonl = str(output_dir / 'raw_corpus_new.jsonl')
    ingest_args = ['--mode', args.mode, '--output', tmp_jsonl]
    if args.mode == 'search':
        ingest_args += ['--query', args.query, '--max', str(args.max_results)]
    else:
        ingest_args += ['--input', args.input]

    ok, out = run_phase(1, ingest_args, args.cwd)
    if not ok:
        return False

    # Append to existing corpus
    with open(raw_jsonl, 'a', encoding='utf-8') as out_f:
        with open(tmp_jsonl, 'r', encoding='utf-8') as in_f:
            out_f.write(in_f.read())

    # Count
    with open(raw_jsonl, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)
    log('info', f'Merged corpus: {total} total papers')

    # Now run Phases 2-6 on merged corpus
    # Skip Phase 1 by passing --skip-ingest
    args.skip_ingest = True
    args.input = None  # Use existing raw_corpus.jsonl
    return run_pipeline(args)


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    init()  # UTF-8 encoding fix

    parser = argparse.ArgumentParser(
        description='Literature Mining — Unified Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --input papers.bib --mode bibtex
  python run_pipeline.py --input papers.csv --mode csv
  python run_pipeline.py --search "artificial life" --max 50
  python run_pipeline.py --input seed.jsonl --mode expand --max-expand 50
  python run_pipeline.py --input new.csv --mode incremental
  python run_pipeline.py --input papers.bib --mode bibtex --check-only
        """
    )

    # Input options
    parser.add_argument('--input', '-i', help='Input file path (bibtex/csv/ris/pdf folder)')
    parser.add_argument('--mode', '-m', required=True,
                        choices=['bibtex', 'ris', 'pdf', 'search', 'csv', 'expand', 'incremental'],
                        help='Input source type')
    parser.add_argument('--query', '-q', help='Search query (for search mode)')
    parser.add_argument('--max-results', '--max', type=int, default=100,
                        help='Max results for search (default: 100)')
    parser.add_argument('--max-expand', type=int, default=50,
                        help='Max papers to add in expand mode (default: 50)')

    # Output
    parser.add_argument('--output-dir', '-o', default='outputs', help='Output directory (default: outputs/)')
    parser.add_argument('--years', help='Year range filter, e.g. 2016-2026')
    parser.add_argument('--cwd', default='.', help='Working directory')

    # Phase toggles
    parser.add_argument('--skip-1', '--skip-ingest', dest='skip_ingest', action='store_true',
                        help='Skip Phase 1 (use existing raw_corpus.jsonl)')
    parser.add_argument('--skip-2', '--skip-clean', dest='skip_clean', action='store_true',
                        help='Skip Phase 2')
    parser.add_argument('--skip-3', '--skip-analyze', dest='skip_analyze', action='store_true',
                        help='Skip Phase 3')
    parser.add_argument('--skip-4', '--skip-frontier', dest='skip_frontier', action='store_true',
                        help='Skip Phase 4')
    parser.add_argument('--skip-5', '--skip-gaps', dest='skip_gaps', action='store_true',
                        help='Skip Phase 5')
    parser.add_argument('--skip-6', '--skip-report', dest='skip_report', action='store_true',
                        help='Skip Phase 6')

    # Diagnostic only
    parser.add_argument('--check-only', action='store_true',
                        help='Stop after Phase 1 diagnostic (no full pipeline)')

    # Resume from phase
    parser.add_argument('--resume-from', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Resume from a specific phase (1-6). Skips all prior phases.')

    args = parser.parse_args()

    # Handle --resume-from: skip prior phases
    _skip_attrs = {1: 'skip_ingest', 2: 'skip_clean', 3: 'skip_analyze',
                   4: 'skip_frontier', 5: 'skip_gaps', 6: 'skip_report'}
    if args.resume_from:
        for p in range(1, args.resume_from):
            if p in _skip_attrs:
                setattr(args, _skip_attrs[p], True)
        log('info', f'Resuming from Phase {args.resume_from} (skipping phases 1-{args.resume_from-1})')

    # Route to incremental mode
    if args.mode == 'incremental':
        success = run_incremental(args)
    else:
        success = run_pipeline(args)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
