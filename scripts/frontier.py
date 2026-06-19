#!/usr/bin/env python3
"""
Phase 4: Research Frontier Detection
======================================
Detect research frontiers via:
  1. Kleinberg-style burst detection on keyword time series
  2. Louvain community detection on keyword co-occurrence network
  3. Frontier scoring: recency + burst + growth + interdisciplinarity

Output: frontiers.json — ranked list of frontier topics with metadata.

Usage:
  python frontier.py --input outputs/clean_corpus.csv \
    --cooccurrence outputs/cooccurrence.csv \
    --trends outputs/trends.json \
    --output outputs/frontiers.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, adaptive_thresholds, quality_report, DEFAULTS


# ----------------------------------------------------------
#  Burst Detection (Kleinberg 2002, simplified)
# ----------------------------------------------------------

def detect_bursts(yearly_counts, years, gamma=1.0):
    """
    Detect bursts in a keyword's yearly frequency time series.
    Simplified Kleinberg: compare each year's rate to the baseline rate.
    Returns list of burst intervals [(start_year, end_year, intensity), ...].
    """
    n = len(yearly_counts)
    if n < 3:
        return []

    total = sum(yearly_counts)
    if total == 0:
        return []

    baseline = max(total / n, 0.5)  # events per year, minimum 0.5 to avoid false bursts
    bursts = []

    i = 0
    while i < n:
        count = yearly_counts[i]
        # A burst year: count significantly above baseline
        if count >= baseline * gamma and count >= 2:
            start = years[i]
            end = years[i]
            peak = count
            j = i + 1
            while j < n and yearly_counts[j] >= baseline * gamma * 0.5:
                if yearly_counts[j] > peak:
                    peak = yearly_counts[j]
                end = years[j]
                j += 1
            # Intensity: peak / baseline
            intensity = peak / baseline if baseline > 0 else 0
            bursts.append((start, end, round(intensity, 2)))
            i = j
        else:
            i += 1

    return bursts


def compute_all_bursts(trends_data, years):
    """Compute bursts for all keywords in trends data."""
    keywords_data = trends_data.get('keywords', {})
    all_bursts = {}

    for kw, data in keywords_data.items():
        yearly = data.get('yearly', {})
        # Build ordered yearly counts
        counts = [yearly.get(str(y), 0) for y in years]
        bursts = detect_bursts(counts, years)
        if bursts:
            all_bursts[kw] = {
                'total': data.get('total', 0),
                'bursts': bursts,
                'cagr_3yr': data.get('cagr_3yr', 0),
            }

    return all_bursts


# ----------------------------------------------------------
#  Community Detection (Louvain, simplified greedy modularity)
# ----------------------------------------------------------

def build_network_from_cooccurrence(cooc_data, top_n=200):
    """
    Build a networkx graph from co-occurrence data.
    Nodes: keywords. Edges: weighted by co-occurrence count.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[ERROR] networkx is required for network analysis")
        sys.exit(1)

    G = nx.Graph()

    for row in cooc_data:
        a = row.get('keyword_a', row.get('source', ''))
        b = row.get('keyword_b', row.get('target', ''))
        w = float(row.get('weight', 1))

        if a and b:
            G.add_edge(a, b, weight=w)

    # Keep only the largest connected component if too large
    if len(G) > top_n * 3:
        components = list(nx.connected_components(G))
        # Sort by size and keep nodes from largest components
        components.sort(key=len, reverse=True)
        keep_nodes = set()
        for comp in components[:5]:
            keep_nodes.update(comp)
        # Also keep nodes with highest degree
        degrees = dict(G.degree())
        top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        keep_nodes.update(n for n, _ in top_degree)
        G = G.subgraph(keep_nodes).copy()

    return G


def detect_communities(G, resolution=1.0):
    """
    Detect communities using Louvain (python-louvain) or greedy modularity.
    Returns dict {node: community_id}.
    """
    try:
        from community import best_partition
        return best_partition(G, weight='weight', random_state=42, resolution=resolution)
    except ImportError:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G, weight='weight')
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            return partition
        except ImportError:
            # Fallback: label propagation
            try:
                from networkx.algorithms.community import label_propagation_communities
                communities = label_propagation_communities(G)
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
                return partition
            except ImportError:
                print("[WARN]  No community detection available, using connected components")
                components = list(nx.connected_components(G) if hasattr(nx, 'connected_components') else [])
                partition = {}
                for i, comp in enumerate(components):
                    for node in comp:
                        partition[node] = i
                return partition


# ----------------------------------------------------------
#  Frontier Scoring
# ----------------------------------------------------------

def score_frontiers(communities, G, burst_data, trends_data, paper_years, min_cluster_size=3):
    """
    Score each community as a research frontier.
    Returns list of frontier dicts, sorted by frontier_score descending.
    """
    keywords_data = trends_data.get('keywords', {})

    # Invert partition: community_id -> [nodes]
    comm_nodes = defaultdict(list)
    for node, cid in communities.items():
        comm_nodes[cid].append(node)

    frontiers = []

    for cid, nodes in comm_nodes.items():
        if len(nodes) < min_cluster_size:
            continue

        # Compute community-level metrics
        avg_year = 0
        growth_rates = []
        burst_scores = []
        n_with_data = 0

        for node in nodes:
            kd = keywords_data.get(node, {})
            yearly = kd.get('yearly', {})

            # Average year (weighted by count)
            total_yearly = 0
            weighted_year = 0
            for yr_str, count in yearly.items():
                yr = int(yr_str)
                count = int(count) if count else 0
                total_yearly += count
                weighted_year += yr * count
            if total_yearly > 0:
                avg_year += weighted_year / total_yearly
                n_with_data += 1

            # Growth rate (3yr CAGR)
            cagr = kd.get('cagr_3yr')
            if cagr is not None and cagr != float('inf') and cagr >= 0:
                growth_rates.append(cagr)

            # Burst intensity
            if node in burst_data:
                for start, end, intensity in burst_data[node].get('bursts', []):
                    burst_scores.append(intensity)

        if n_with_data == 0:
            continue

        avg_year /= n_with_data
        avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
        max_burst = max(burst_scores) if burst_scores else 0
        mean_burst = sum(burst_scores) / len(burst_scores) if burst_scores else 0

        # Interdisciplinarity: edges connecting this community to others
        cross_edges = 0
        for node in nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in nodes:
                    cross_edges += 1

        # Normalize components (0-1)
        # Recency: avg_year mapped to 0-1 across full year range
        all_years = list(paper_years) if paper_years else [2010, 2026]
        year_span = max(all_years) - min(all_years) if all_years else 10
        recency_norm = min(1.0, max(0.0, (avg_year - min(all_years)) / year_span)) if year_span > 0 else 0.5
        growth_norm = min(1.0, max(0.0, avg_growth))
        burst_norm = min(1.0, max(0.0, (max_burst * 0.7 + mean_burst * 0.3) / 10))
        cross_norm = min(1.0, max(0.0, cross_edges / max(1, len(nodes)) / 5))

        frontier_score = (
            DEFAULTS['frontier_score_burst_weight'] * burst_norm +
            DEFAULTS['frontier_score_recency_weight'] * recency_norm +
            DEFAULTS['frontier_score_growth_weight'] * growth_norm +
            DEFAULTS['frontier_score_cross_weight'] * cross_norm
        )

        # Cluster size penalty: very large clusters are usually noise (generic concepts)
        nc = len(nodes)
        if nc > 60:
            frontier_score *= 0.5
        elif nc > 30:
            frontier_score *= 0.8

        # Community label: top keywords by degree within community
        subgraph = G.subgraph(nodes)
        degrees = dict(subgraph.degree(weight='weight'))
        top_keywords = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        label_keywords = [kw for kw, _ in top_keywords]

        # Burst keywords in this community
        burst_kws = [kw for kw in nodes if kw in burst_data and burst_data[kw].get('bursts')]

        # Compute a human-readable label
        label = ' / '.join(label_keywords[:3])

        frontiers.append({
            'community_id': cid,
            'label': label,
            'top_keywords': label_keywords,
            'burst_keywords': burst_kws,
            'node_count': len(nodes),
            'avg_year': round(avg_year, 1),
            'avg_growth_rate': round(avg_growth, 3),
            'max_burst_intensity': round(max_burst, 2),
            'cross_cluster_edges': cross_edges,
            'recency_norm': round(recency_norm, 3),
            'growth_norm': round(growth_norm, 3),
            'burst_norm': round(burst_norm, 3),
            'cross_norm': round(cross_norm, 3),
            'frontier_score': round(frontier_score, 3),
        })

    frontiers.sort(key=lambda x: x['frontier_score'], reverse=True)
    return frontiers


def main():
    init()  # Windows UTF-8 encoding fix
    parser = argparse.ArgumentParser(description='Frontier Detection — Phase 4')
    parser.add_argument('--input', '-i', help='Cleaned corpus CSV (for paper year data)')
    parser.add_argument('--cooccurrence', '-c', required=True, help='Co-occurrence CSV')
    parser.add_argument('--trends', '-t', required=True, help='Trends JSON from Phase 3')
    parser.add_argument('--output', '-o', required=True, help='Output frontiers.json')
    parser.add_argument('--top-n', type=int, default=100, help='Top N keywords for network')
    parser.add_argument('--burst-gamma', type=float, default=DEFAULTS['burst_gamma'],
                        help=f'Kleinberg burst detection gamma (default: {DEFAULTS["burst_gamma"]})')
    parser.add_argument('--louvain-resolution', type=float, default=DEFAULTS['louvain_resolution'],
                        help=f'Louvain community resolution (default: {DEFAULTS["louvain_resolution"]})')
    args = parser.parse_args()

    print("=" * 60)
    print("[FRONTIER] Phase 4: Research Frontier Detection")
    print("=" * 60)

    # Load co-occurrence
    cooc_path = Path(args.cooccurrence)
    if cooc_path.suffix == '.csv':
        try:
            import pandas as pd
            cooc_data = pd.read_csv(cooc_path).to_dict('records')
        except ImportError:
            import csv
            with open(cooc_path, 'r', encoding='utf-8-sig') as f:
                cooc_data = list(csv.DictReader(f))
    else:
        print("[ERROR] Co-occurrence file must be CSV")
        sys.exit(1)

    print(f"   Co-occurrence pairs loaded: {len(cooc_data)}")

    # Load trends
    with open(args.trends, 'r', encoding='utf-8') as f:
        trends_data = json.load(f)
    years = trends_data.get('years', [])
    print(f"   Years: {years[0]}-{years[-1]}" if years else "   Years: N/A")

    # Load paper years from corpus (if available)
    paper_years = set()
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(input_path)
                for y in df['year'].dropna():
                    try:
                        paper_years.add(int(y))
                    except (ValueError, TypeError):
                        pass
            except ImportError:
                import csv
                with open(input_path, 'r', encoding='utf-8-sig') as f:
                    for row in csv.DictReader(f):
                        yr = row.get('year', '')
                        if yr is not None:
                            paper_years.add(int(yr) if isinstance(yr, str) else yr)
    if not paper_years:
        paper_years = set(years)
    year_span = max(paper_years) - min(paper_years) + 1 if paper_years else 0
    print(f"   Paper years: {min(paper_years)}-{max(paper_years)} (span={year_span}yr)" if paper_years else "")

    # Adaptive thresholds based on corpus size
    # Paper count from input CSV (not cooc_data length — that's edges, not papers!)
    n_papers = len(df) if 'df' in dir() else len(cooc_data)
    n_keywords = len(G.nodes()) if 'G' in locals() else 0
    n_keywords = max(n_keywords, len(trends_data.get('keywords', {})))
    thresholds = adaptive_thresholds(n_papers, n_keywords, year_span)
    quality_report(thresholds)

    # Burst detection
    print("\n   Detecting bursts...")
    burst_data = compute_all_bursts(trends_data, years)
    n_burst = len(burst_data)
    recent_burst = 0
    if len(years) >= 2:
        recent_burst = sum(
            1 for kw, bd in burst_data.items()
            if bd.get('bursts') and bd['bursts'] and bd['bursts'][-1][1] >= years[-2]
        )
    else:
        recent_burst = n_burst
    print(f"   Keywords with bursts: {n_burst} ({recent_burst} active in recent 2 years)")

    # Build network + detect communities
    print("\n   Building co-occurrence network...")
    G = build_network_from_cooccurrence(cooc_data, args.top_n)
    print(f"   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Density: {G.number_of_edges() / max(1, G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.4f}")

    print("\n   Detecting communities...")
    communities = detect_communities(G, resolution=args.louvain_resolution)
    n_comm = len(set(communities.values()))
    print(f"   Communities found: {n_comm}")

    # Score frontiers
    print("\n   Scoring frontiers...")
    frontiers = score_frontiers(communities, G, burst_data, trends_data, paper_years,
                                 min_cluster_size=thresholds['min_cluster_size'])

    # Build output
    output = {
        'frontiers': frontiers,
        'burst_keywords': [
            {'keyword': kw, 'total': bd['total'], 'bursts': bd['bursts'], 'cagr_3yr': bd.get('cagr_3yr')}
            for kw, bd in sorted(burst_data.items(),
                                 key=lambda x: x[1].get('bursts', [(0,0,0)])[0][2],
                                 reverse=True)
        ],
        'topology': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_communities': n_comm,
            'density': round(G.number_of_edges() / max(1, G.number_of_nodes() * (G.number_of_nodes() - 1) / 2), 4),
        },
        'parameters': {
            'burst_gamma': 1.0,
            'top_n_keywords': args.top_n,
        },
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'-' * 50}")
    print(f"[FRONTIER] Frontier detection complete")
    print(f"   Communities: {n_comm}")
    print(f"   Frontiers scored: {len(frontiers)}")
    if frontiers:
        print(f"\n   Top-5 Frontiers:")
        for i, fr in enumerate(frontiers[:5]):
            print(f"   {i+1}. {fr['label']}")
            print(f"      Score={fr['frontier_score']:.3f} | Keywords={fr['top_keywords'][:3]} | avg_year={fr['avg_year']}")
    print(f"   Output: {output_path}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
