#!/usr/bin/env python3
"""
Phase 3: Bibliometric Analysis
================================
Compute:
  - Keyword frequency distributions (overall + yearly)
  - Keyword co-occurrence matrix
  - Yearly trend data for each keyword
  - Author collaboration network (optional)
  - Citation analysis (if data available)

Outputs:
  - keyword_freq.csv       : keyword | total | per-year counts | trend | cagr_3yr
  - cooccurrence.csv       : keyword_a | keyword_b | weight | year_range
  - trends.json            : full trend data for top keywords
  - author_network.csv     : author_a | author_b | weight (optional)
  - stats_summary.json     : (updated from Phase 2, enriched)

Usage:
  python analyze.py --input outputs/clean_corpus.csv --output-dir outputs/
"""

import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, adaptive_thresholds, quality_report, DEFAULTS

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_corpus(filepath):
    """Load cleaned corpus CSV."""
    if HAS_PANDAS:
        return pd.read_csv(filepath)
    else:
        import csv
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            return list(reader)


def parse_keywords(kw_str):
    """Parse semicolon-separated keyword string to list."""
    if not kw_str or (isinstance(kw_str, float) and math.isnan(kw_str)):
        return []
    return [k.strip().lower() for k in str(kw_str).split(';') if k.strip()]


def compute_keyword_freq(df, is_dict_list=False):
    """Compute overall and yearly keyword frequencies."""
    if is_dict_list:
        # Manual iteration
        freq = Counter()
        yearly = defaultdict(Counter)
        years_set = set()
        n_papers = len(df)

        for row in df:
            kws = parse_keywords(row.get('keywords_normalized', ''))
            year = str(row.get('year', '')).strip()
            for kw in kws:
                freq[kw] += 1
                if year and year.isdigit():
                    yearly[kw][int(year)] += 1
                    years_set.add(int(year))

        years_sorted = sorted(years_set)
        n_years = len(years_sorted) if years_sorted else 1

        # Build result rows
        result = []
        for kw, total in freq.most_common():
            row = {'keyword': kw, 'total_count': total}
            for y in years_sorted:
                row[str(y)] = yearly[kw].get(y, 0)

            # Trend direction
            # Compute papers per year for normalization
            papers_per_year = {y: 0 for y in years_sorted}
            for row_data in df:
                yr_val = row_data.get('year', '')
                yr = int(yr_val) if isinstance(yr_val, (int, float)) and not (isinstance(yr_val, float) and math.isnan(yr_val)) else (int(str(yr_val).strip()) if str(yr_val).strip().isdigit() else None)
                if yr in papers_per_year:
                    papers_per_year[yr] += 1

            if n_years >= 4:
                early_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[:2])
                late_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[-2:])
                prev_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[-4:-2])
                early_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[:2])
                late_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[-2:])
                prev_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[-4:-2])
                # Normalize: keyword_freq_per_paper
                early = early_raw / max(1, early_papers)
                late = late_raw / max(1, late_papers)
                prev_last = prev_raw / max(1, prev_papers)
                if late > early * 1.3:
                    row['trend'] = 'rising'
                elif late < early * 0.7:
                    row['trend'] = 'declining'
                else:
                    row['trend'] = 'stable'
                if prev_last > 0:
                    row['cagr_3yr'] = round((late / prev_last) ** (1/2) - 1, 3) if late > 0 else -1
                else:
                    row['cagr_3yr'] = float('inf') if late > 0 else 0
            else:
                row['trend'] = 'insufficient_data'
                row['cagr_3yr'] = None

            result.append(row)

        return result, years_sorted, freq
    else:
        # pandas DataFrame
        freq = Counter()
        yearly = defaultdict(Counter)
        years_set = set()

        for _, row in df.iterrows():
            kws = parse_keywords(row.get('keywords_normalized', ''))
            year = row.get('year')
            for kw in kws:
                freq[kw] += 1
                if pd.notna(year):
                    try:
                        yearly[kw][int(year)] += 1
                        years_set.add(int(year))
                    except (ValueError, TypeError):
                        pass

        years_sorted = sorted(years_set)
        n_years = len(years_sorted) if years_sorted else 1

        # Papers per year for normalization
        papers_per_year = {y: 0 for y in years_sorted}
        for _, row_data in df.iterrows():
            yr = row_data.get('year')
            try:
                yr = int(yr) if not (isinstance(yr, float) and math.isnan(yr)) else None
                if yr in papers_per_year:
                    papers_per_year[yr] += 1
            except (ValueError, TypeError):
                pass

        result = []
        for kw, total in freq.most_common():
            row = {'keyword': kw, 'total_count': total}
            for y in years_sorted:
                row[str(y)] = yearly[kw].get(y, 0)

            if n_years >= 4:
                early_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[:2])
                late_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[-2:])
                prev_papers = sum(papers_per_year.get(y, 0) for y in years_sorted[-4:-2])
                early_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[:2])
                late_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[-2:])
                prev_raw = sum(yearly[kw].get(y, 0) for y in years_sorted[-4:-2])
                early = early_raw / max(1, early_papers)
                late = late_raw / max(1, late_papers)
                prev_last = prev_raw / max(1, prev_papers)
                row['trend'] = 'rising' if late > early * 1.3 else ('declining' if late < early * 0.7 else 'stable')
                row['cagr_3yr'] = round((late / prev_last) ** (1/2) - 1, 3) if prev_last > 0 and late > 0 else (-1 if late == 0 else float('inf'))
            else:
                row['trend'] = 'insufficient_data'
                row['cagr_3yr'] = None

            result.append(row)

        return result, years_sorted, freq


def compute_cooccurrence(df, is_dict_list=False, top_k=200):
    """
    Compute keyword co-occurrence matrix.
    Two keywords co-occur if they appear in the same paper.
    Returns list of {keyword_a, keyword_b, weight, year_range}.
    """
    cooc = Counter()
    kw_years = defaultdict(set)

    def process_row(kws, year):
        # Only consider pairs within top_k
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                a, b = sorted([kws[i], kws[j]])
                cooc[(a, b)] += 1
                if year:
                    kw_years[(a, b)].add(year)

    if is_dict_list:
        for row in df:
            kws = parse_keywords(row.get('keywords_normalized', ''))
            yr_val = row.get('year', '')
            if isinstance(yr_val, int):
                year = yr_val
            elif isinstance(yr_val, str) and yr_val.strip().isdigit():
                year = int(yr_val.strip())
            else:
                year = None
            process_row(kws, year)
    else:
        for _, row in df.iterrows():
            kws = parse_keywords(row.get('keywords_normalized', ''))
            year = row.get('year')
            try:
                year = int(year) if not (isinstance(year, float) and math.isnan(year)) else None
            except (ValueError, TypeError):
                year = None
            process_row(kws, year)

    # Filter to only top keywords
    # We need the top keyword set — compute separately
    kw_freq = Counter()
    if is_dict_list:
        for row in df:
            for kw in parse_keywords(row.get('keywords_normalized', '')):
                kw_freq[kw] += 1
    else:
        for _, row in df.iterrows():
            for kw in parse_keywords(row.get('keywords_normalized', '')):
                kw_freq[kw] += 1

    top_kw_set = {kw for kw, _ in kw_freq.most_common(top_k)}

    result = []
    for (a, b), weight in cooc.most_common():
        if a in top_kw_set and b in top_kw_set and weight >= 2:
            yrs = sorted(kw_years.get((a, b), set()))
            result.append({
                'keyword_a': a,
                'keyword_b': b,
                'weight': weight,
                'year_range_start': min(yrs) if yrs else None,
                'year_range_end': max(yrs) if yrs else None,
            })

    return result


def compute_author_network(df, is_dict_list=False):
    """Compute author collaboration network (co-authorship)."""
    collab = Counter()

    def process_authors(authors_str):
        if not authors_str:
            return
        authors = [a.strip() for a in str(authors_str).split(';') if a.strip()]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = sorted([authors[i], authors[j]])
                collab[(a, b)] += 1

    if is_dict_list:
        for row in df:
            process_authors(row.get('authors', ''))
    else:
        for _, row in df.iterrows():
            process_authors(row.get('authors', ''))

    result = []
    for (a, b), weight in collab.most_common():
        if weight >= 2:
            result.append({'author_a': a, 'author_b': b, 'weight': weight})

    return result


def main():
    init()  # Windows UTF-8 encoding fix
    parser = argparse.ArgumentParser(description='Bibliometric Analysis — Phase 3')
    parser.add_argument('--input', '-i', required=True, help='Cleaned corpus CSV')
    parser.add_argument('--output-dir', '-o', default='outputs', help='Output directory')
    parser.add_argument('--top-k', type=int, default=200, help='Top N keywords for co-occurrence')
    args = parser.parse_args()

    print("=" * 60)
    print("[ANALYZE] Phase 3: Bibliometric Analysis")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if HAS_PANDAS:
        df = pd.read_csv(args.input)
        is_dict_list = False
        print(f"   Loaded {len(df)} papers (pandas)")
    else:
        df = load_corpus(args.input)
        is_dict_list = True
        print(f"   Loaded {len(df)} papers (csv fallback)")

    # Keyword frequency
    print("\n   Computing keyword frequencies...")
    kw_freq_data, years_sorted, kw_counter = compute_keyword_freq(df, is_dict_list)
    print(f"   Unique keywords: {len(kw_freq_data)}")
    print(f"   Year range: {years_sorted[0]}-{years_sorted[-1]}" if years_sorted else "   Year range: N/A")

    # Save keyword_freq.csv
    kw_path = output_dir / 'keyword_freq.csv'
    if HAS_PANDAS:
        kw_df = pd.DataFrame(kw_freq_data)
        kw_df.to_csv(kw_path, index=False, encoding='utf-8-sig')
    else:
        import csv
        with open(kw_path, 'w', encoding='utf-8-sig', newline='') as f:
            if kw_freq_data:
                writer = csv.DictWriter(f, fieldnames=kw_freq_data[0].keys())
                writer.writeheader()
                writer.writerows(kw_freq_data)
    print(f"   -> {kw_path}")

    # Co-occurrence
    print("\n   Computing co-occurrence matrix...")
    cooc_data = compute_cooccurrence(df, is_dict_list, args.top_k)
    print(f"   Co-occurrence pairs: {len(cooc_data)}")

    cooc_path = output_dir / 'cooccurrence.csv'
    if cooc_data:
        if HAS_PANDAS:
            pd.DataFrame(cooc_data).to_csv(cooc_path, index=False, encoding='utf-8-sig')
        else:
            import csv
            with open(cooc_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cooc_data[0].keys())
                writer.writeheader()
                writer.writerows(cooc_data)
    print(f"   -> {cooc_path}")

    # Trends JSON
    print("\n   Computing trends...")
    trends = {}
    for row in kw_freq_data[:100]:  # top 100
        kw = row['keyword']
        yearly_data = {}
        for y in years_sorted:
            yearly_data[y] = row.get(str(y), 0)
        trends[kw] = {
            'total': row['total_count'],
            'yearly': yearly_data,
            'trend': row.get('trend'),
            'cagr_3yr': row.get('cagr_3yr'),
        }

    trends_path = output_dir / 'trends.json'
    with open(trends_path, 'w', encoding='utf-8') as f:
        json.dump({
            'years': years_sorted,
            'keywords': trends,
        }, f, ensure_ascii=False, indent=2)
    print(f"   -> {trends_path}")

    # Author network
    print("\n   Computing author collaboration network...")
    author_data = compute_author_network(df, is_dict_list)
    print(f"   Author collaboration pairs: {len(author_data)}")

    if author_data:
        author_path = output_dir / 'author_network.csv'
        if HAS_PANDAS:
            pd.DataFrame(author_data).to_csv(author_path, index=False, encoding='utf-8-sig')
        else:
            import csv
            with open(author_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=author_data[0].keys())
                writer.writeheader()
                writer.writerows(author_data)
        print(f"   -> {author_path}")

    # Summary stats
    summary_path = output_dir / 'stats_summary.json'
    summary = {}
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    summary['unique_keywords'] = len(kw_freq_data)
    summary['cooccurrence_pairs'] = len(cooc_data)
    summary['author_collab_pairs'] = len(author_data) if author_data else 0
    summary['year_range'] = [years_sorted[0], years_sorted[-1]] if years_sorted else []
    summary['top_keywords_20'] = [
        {'keyword': kw, 'count': c, 'trend': row.get('trend', '')}
        for kw, c in kw_counter.most_common(20)
        for row in kw_freq_data if row['keyword'] == kw
    ][:20]
    summary['rising_keywords'] = [row['keyword'] for row in kw_freq_data if row.get('trend') == 'rising'][:20]
    summary['declining_keywords'] = [row['keyword'] for row in kw_freq_data if row.get('trend') == 'declining'][:20]

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   -> {summary_path} (updated)")

    print(f"\n{'-' * 50}")
    print(f"[ANALYZE] Analysis complete")
    print(f"   Keywords: {len(kw_freq_data)} unique | {len(cooc_data)} co-occurrence pairs")
    print(f"   Rising:   {len(summary.get('rising_keywords', []))} keywords")
    print(f"   Declining: {len(summary.get('declining_keywords', []))} keywords")
    if years_sorted:
        print(f"   Years:    {years_sorted[0]}-{years_sorted[-1]}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
