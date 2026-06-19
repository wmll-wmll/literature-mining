#!/usr/bin/env python3
"""
Phase 5: Research Gap Identification
=======================================
Identify research gaps via three strategies:
  1. Density Gap  : Low-paper-count keywords near high-density frontier clusters
  2. Bridge Gap   : Semantically related but unconnected active clusters
  3. Temporal Gap : Declining keywords with rising external relevance

Output: gaps.json — ranked list of research gaps with opportunity scores.

Usage:
  python gaps.py --input outputs/clean_corpus.csv \
    --cooccurrence outputs/cooccurrence.csv \
    --frontiers outputs/frontiers.json \
    --trends outputs/trends.json \
    --output outputs/gaps.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, adaptive_thresholds, quality_report, DEFAULTS


def load_corpus(filepath):
    try:
        import pandas as pd
        return pd.read_csv(filepath)
    except ImportError:
        import csv
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))


def parse_keywords(kw_str):
    if not kw_str or (isinstance(kw_str, float) and math.isnan(kw_str)):
        return []
    return [k.strip().lower() for k in str(kw_str).split(';') if k.strip()]


# Semantic quality gate for gap keywords
_GENERIC_SINGLE_WORDS = {
    'possible', 'impossible', 'introduce', 'introduced', 'use', 'used',
    'show', 'shown', 'find', 'found', 'give', 'give', 'take', 'make',
    'like', 'role', 'task', 'way', 'case', 'need', 'allow', 'enable',
    'key', 'main', 'major', 'support', 'result', 'change', 'develop',
    'different', 'various', 'potential', 'new', 'novel', 'high', 'low',
    'large', 'small', 'complex', 'simple', 'basic', 'general', 'specific',
    'real', 'true', 'full', 'part', 'form', 'type', 'kind', 'field',
    'system', 'model', 'process', 'method', 'approach', 'technique',
    'time', 'year', 'day', 'week', 'month', 'life', 'work', 'world',
    'human', 'nature', 'science', 'technology', 'engineering',
    'present', 'recent', 'current', 'future', 'early', 'late',
    'level', 'number', 'amount', 'rate', 'range', 'scale', 'size',
    'state', 'status', 'view', 'point', 'order', 'set', 'group',
}


def _valid_gap_keyword(kw):
    """Return True if keyword is semantically meaningful for a research gap.
    Relaxed: single words >= 6 chars pass if not in generic list.
    Multi-word terms (2-gram+) always pass."""
    kw = kw.strip().lower()
    if not kw or len(kw) < 4:
        return False
    if kw in _GENERIC_SINGLE_WORDS:
        return False
    words = kw.split()
    if len(words) >= 2:
        return True
    # Single word: pass if >= 6 chars (catches "crispr", "lenia", "mrna", "genome", etc.)
    if len(kw) >= 6:
        return True
    return False



def build_network(cooc_data):
    """Build a networkx graph from co-occurrence data."""
    import networkx as nx
    G = nx.Graph()
    for row in cooc_data:
        a = row.get('keyword_a', row.get('source', ''))
        b = row.get('keyword_b', row.get('target', ''))
        w = float(row.get('weight', 1))
        if a and b:
            G.add_edge(a, b, weight=w)
    return G


# ----------------------------------------------------------
#  Gap Type 1: Density Gaps
# ----------------------------------------------------------

def find_density_gaps(G, frontiers, trends_data, paper_count):
    """
    Find keywords that are:
    - Close (in network distance) to a high-score frontier cluster
    - Have low paper counts (< median)
    - Have high edge weight to frontier nodes
    """
    keywords_data = trends_data.get('keywords', {})

    # Build frontier keyword sets
    frontier_kw_sets = []
    for fr in frontiers:
        frontier_kw_sets.append(set(fr.get('top_keywords', [])))

    gaps = []

    # Compute node degrees for normalization
    node_degrees = dict(G.degree(weight='weight'))

    for node in G.nodes():
        # Skip if node is in a frontier cluster
        in_frontier = any(node in fset for fset in frontier_kw_sets)
        if in_frontier:
            continue

        kd = keywords_data.get(node, {})
        total = kd.get('total', 0)
        # Skip keywords with 0 frequency (TF-IDF terms not in author keywords)
        if total == 0:
            continue
        if total > paper_count * 0.1:
            continue

        node_degree = node_degrees.get(node, 1)

        # Compute proximity to each frontier
        best_proximity = 0
        best_frontier_label = ""
        for i, fr in enumerate(frontiers):
            fset = frontier_kw_sets[i]
            edges_to_fr = []
            for fn in fset:
                if fn in G and G.has_edge(node, fn):
                    edges_to_fr.append(G[node][fn].get('weight', 1))
            if edges_to_fr:
                # Proximity = avg edge weight / (keyword_freq * node_degree)
                # Normalized to [0,1]: higher = closer to frontier relative to overall connectivity
                raw = sum(edges_to_fr) / len(edges_to_fr)
                proximity = raw / max(1, total) / max(1, node_degree)
                # Clamp to [0, 1]
                proximity = min(1.0, proximity)
                if proximity > best_proximity:
                    best_proximity = proximity
                    best_frontier_label = fr.get('label', '')

        if best_proximity > 0.5 and _valid_gap_keyword(node):  # threshold + semantic gate
            density = total / max(1, paper_count)
            gaps.append({
                'keyword': node,
                'gap_type': 'density',
                'paper_count': total,
                'density': round(density, 4),
                'proximity_to_frontier': round(best_proximity, 3),
                'nearby_frontier': best_frontier_label,
                'trend': kd.get('trend', ''),
                'cagr_3yr': kd.get('cagr_3yr'),
            })

    return gaps


# ----------------------------------------------------------
#  Gap Type 2: Bridge Gaps
# ----------------------------------------------------------

def find_bridge_gaps(G, frontiers, trends_data):
    """
    Find pairs of frontier communities that are:
    - Both active (high frontier score / growth)
    - NOT connected by co-occurrence edges between their keywords
    - Semantically related (keyword overlap or text similarity)

    Uses a simple heuristic: shared neighbors in the co-occurrence network.
    """
    import networkx as nx

    # Build frontier community sets — use all top_keywords for bridge detection
    frontier_sets = []
    for fr in frontiers:
        all_kws = set(fr.get('top_keywords', []))
        frontier_sets.append({
            'label': fr.get('label', ''),
            'keywords': all_kws,
            'score': fr.get('frontier_score', 0),
            'growth': fr.get('avg_growth_rate', 0),
        })

    gaps = []

    for i in range(len(frontier_sets)):
        for j in range(i + 1, len(frontier_sets)):
            f1 = frontier_sets[i]
            f2 = frontier_sets[j]

            # Check direct edges between clusters
            direct_edges = 0
            for kw1 in f1['keywords']:
                for kw2 in f2['keywords']:
                    if kw1 in G and kw2 in G and G.has_edge(kw1, kw2):
                        direct_edges += 1

            # Check shared neighbors (keywords that co-occur with both)
            shared_neighbors = set()
            for kw1 in f1['keywords']:
                if kw1 in G:
                    neighbors1 = set(G.neighbors(kw1))
                    for kw2 in f2['keywords']:
                        if kw2 in G:
                            neighbors2 = set(G.neighbors(kw2))
                            shared_neighbors.update(neighbors1 & neighbors2)

            # Bridge potential: few direct edges + many shared neighbors + both active
            f1_size = len(f1['keywords'])
            f2_size = len(f2['keywords'])
            max_possible_edges = f1_size * f2_size
            edge_ratio = direct_edges / max(1, max_possible_edges)

            shared_overlap = len(shared_neighbors) / max(1, f1_size + f2_size)

            # High bridge = low edge_ratio + high shared_overlap + both high score
            # Adaptive edge_ratio: denser networks need looser threshold
            edge_threshold = 0.35 if max_possible_edges > 30 else 0.25
            if edge_ratio < edge_threshold and shared_overlap > 0:
                bridge_score = shared_overlap * (1 - edge_ratio)
                combined_growth = (f1['growth'] + f2['growth']) / 2

                gaps.append({
                    'gap_type': 'bridge',
                    'bridging_clusters': [f1['label'], f2['label']],
                    'cluster1_score': f1['score'],
                    'cluster2_score': f2['score'],
                    'direct_edges': direct_edges,
                    'shared_neighbors': len(shared_neighbors),
                    'shared_neighbor_examples': sorted(shared_neighbors)[:10],
                    'edge_ratio': round(edge_ratio, 3),
                    'shared_overlap': round(shared_overlap, 3),
                    'combined_growth': round(combined_growth, 3),
                    'bridge_potential': round(shared_overlap * (1 - edge_ratio), 3),
                })

    return gaps


# ----------------------------------------------------------
#  Gap Type 3: Temporal Gaps
# ----------------------------------------------------------

def find_semantic_gaps(G, frontiers, trends_data, paper_count):
    """
    Find semantic gaps: frontier clusters that are semantically close
    (by centroid distance in co-occurrence network) but have no direct edges.
    Uses network shortest-path distance as a proxy for semantic similarity.
    """
    import networkx as nx
    gaps = []

    if len(frontiers) < 2:
        return gaps

    # Build frontier centroids: set of all keywords in each frontier
    frontier_sets = []
    for fr in frontiers:
        f_kws = set(fr.get('top_keywords', [])[:10])
        # Filter to nodes actually in the graph
        f_kws = {kw for kw in f_kws if kw in G}
        if f_kws:
            frontier_sets.append({
                'label': fr.get('label', ''),
                'keywords': f_kws,
                'score': fr.get('frontier_score', 0),
            })

    for i in range(len(frontier_sets)):
        for j in range(i + 1, len(frontier_sets)):
            f1 = frontier_sets[i]
            f2 = frontier_sets[j]

            # Check direct edges
            has_direct = False
            for kw1 in f1['keywords']:
                for kw2 in f2['keywords']:
                    if G.has_edge(kw1, kw2):
                        has_direct = True
                        break
                if has_direct:
                    break

            if has_direct:
                continue  # already connected

            # Compute min network distance between clusters
            min_dist = float('inf')
            for kw1 in f1['keywords']:
                for kw2 in f2['keywords']:
                    try:
                        d = nx.shortest_path_length(G, kw1, kw2, weight='weight')
                        min_dist = min(min_dist, d)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

            if min_dist < float('inf') and min_dist <= 4:
                # Semantic proximity = 1 / distance (closer = more related)
                semantic_score = 1.0 / max(1, min_dist)
                gap_score = semantic_score * (f1['score'] + f2['score']) / 2

                if gap_score > 0.2:
                    gaps.append({
                        'gap_type': 'semantic',
                        'label': f'Semantic gap: {f1["label"]} <-> {f2["label"]}',
                        'description': f'Frontiers "{f1["label"]}" and "{f2["label"]}" are '
                                       f'{min_dist} steps apart in the co-occurrence network '
                                       f'with no direct connections — semantically close but isolated.',
                        'gap_score': round(min(0.85, gap_score), 3),
                        'opportunity': '高——语义接近但无共现' if min_dist <= 2 else '中——需要验证',
                        'suggested_approach': f'Explore methods from "{f1["label"]}" applied to "{f2["label"]}" problems.',
                        'distance': min_dist,
                        'semantic_proximity': round(semantic_score, 3),
                    })

    gaps.sort(key=lambda x: x['gap_score'], reverse=True)
    return gaps[:10]


def find_temporal_gaps(trends_data, years, burst_data, paper_count=100):
    """
    Find keywords that:
    - Had activity in early years but declined sharply
    - Have high external relevance (e.g., cited by rising keywords)
    """
    keywords_data = trends_data.get('keywords', {})

    gaps = []
    if len(years) < 6:
        return gaps  # Need enough time span

    early_window = years[:3]
    late_window = years[-3:]

    for kw, data in keywords_data.items():
        yearly = data.get('yearly', {})

        early_count = sum(yearly.get(str(y), 0) for y in early_window)
        late_count = sum(yearly.get(str(y), 0) for y in late_window)

        # Early activity -> sharp decline
        min_papers = max(10, int(paper_count * DEFAULTS['gap_temporal_min_papers_factor']))
        if early_count >= min_papers and late_count <= early_count * DEFAULTS['gap_temporal_decline_ratio']:
            decline_ratio = late_count / max(1, early_count)
            years_since_peak = 0
            peak_year = None
            peak_count = 0
            for yr_str, cnt in yearly.items():
                cnt = int(cnt) if cnt else 0
                if cnt > peak_count:
                    peak_count = cnt
                    peak_year = int(yr_str)
            if peak_year:
                years_since_peak = years[-1] - peak_year

            gaps.append({
                'keyword': kw,
                'gap_type': 'temporal',
                'early_count': early_count,
                'late_count': late_count,
                'decline_ratio': round(decline_ratio, 3),
                'peak_year': peak_year,
                'peak_count': peak_count,
                'years_since_peak': years_since_peak,
                'trend': 'declining',
                'total_papers': data.get('total', 0),
            })

    return gaps


# ----------------------------------------------------------
#  Qualitative Gap Inference (fallback for small corpora)
# ----------------------------------------------------------

def infer_gaps_qualitative(G, frontiers, trends_data, paper_count):
    """
    When quantitative gap detection returns nothing (common with small corpora),
    infer potential gaps from the frontier community structure.
    Uses network topology to generate hypotheses.
    """
    import networkx as nx
    gaps = []
    gap_id = 0

    # Strategy 1: Keywords that sit between frontier clusters (broker nodes)
    frontier_kw_sets = [set(fr.get('top_keywords', [])) for fr in frontiers]
    all_frontier_kws = set().union(*frontier_kw_sets) if frontier_kw_sets else set()

    for node in G.nodes():
        if node in all_frontier_kws:
            continue
        # Check if this node connects to multiple frontier clusters
        connections = []
        for i, fset in enumerate(frontier_kw_sets):
            for fkw in fset:
                if fkw in G and G.has_edge(node, fkw):
                    connections.append(i)
                    break
        n_connections = len(set(connections))
        if n_connections >= 2:
            gap_id += 1
            connected_frontiers = [frontiers[i]['label'] for i in set(connections)]

            # Gradient scoring based on:
            # - number of frontier clusters bridged (2→0.5, 3→0.6, 4+→0.7)
            # - node degree / betweenness centrality proxy
            degree = G.degree(node, weight='weight') if node in G else 1
            degree_norm = min(1.0, degree / max(1, max(dict(G.degree(weight='weight')).values())))

            connection_bonus = min(0.25, (n_connections - 2) * 0.08)
            score = round(0.40 + connection_bonus + degree_norm * 0.20, 3)
            score = min(0.85, score)  # cap

            opportunity = '高——多前沿桥接' if n_connections >= 3 else ('中——双前沿桥接' if n_connections == 2 else '低')
            gaps.append({
                'rank': gap_id,
                'label': f'Broker: {node}',
                'gap_type': 'density',
                'description': f'Keyword "{node}" bridges {n_connections} frontiers ({", ".join(connected_frontiers)}). '
                               f'It connects multiple active research communities but has low standalone presence.',
                'keyword': node,
                'gap_score': score,
                'opportunity': opportunity,
                'suggested_approach': f'Investigate how "{node}" can serve as a bridge between '
                                      f'{connected_frontiers[0]} and {connected_frontiers[-1]}.',
                'nearby_frontier': connected_frontiers[0] if connected_frontiers else '',
                'score_components': {
                    'bridge_potential': round(0.5 + connection_bonus, 3),
                    'density': 0.5,
                    'centrality': round(degree_norm, 3),
                    'feasibility': 0.5,
                },
            })

    # Strategy 2: Unconnected frontier clusters (potential bridges)
    for i in range(len(frontiers)):
        for j in range(i + 1, len(frontiers)):
            f1_kws = set(frontiers[i].get('top_keywords', []))
            f2_kws = set(frontiers[j].get('top_keywords', []))
            # Check if any edges exist between these clusters
            has_edge = False
            for kw1 in f1_kws:
                for kw2 in f2_kws:
                    if kw1 in G and kw2 in G and G.has_edge(kw1, kw2):
                        has_edge = True
                        break
                if has_edge:
                    break
            if not has_edge:
                gap_id += 1
                # Compute shortest path distance
                min_dist = float('inf')
                for kw1 in f1_kws:
                    for kw2 in f2_kws:
                        if kw1 in G and kw2 in G:
                            try:
                                dist = nx.shortest_path_length(G, kw1, kw2)
                                min_dist = min(min_dist, dist)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                pass
                if min_dist < float('inf') and min_dist <= 3:
                    gaps.append({
                        'rank': gap_id,
                        'label': f'Bridge: {frontiers[i]["label"]} <-> {frontiers[j]["label"]}',
                        'gap_type': 'bridge',
                        'description': f'Frontiers "{frontiers[i]["label"]}" and "{frontiers[j]["label"]}" '
                                       f'are not directly connected but are {min_dist} steps apart in the network. '
                                       f'This suggests a potential cross-field research opportunity.',
                        'bridging_clusters': [frontiers[i]['label'], frontiers[j]['label']],
                        'gap_score': 0.50 + (1.0 / max(1, min_dist)) * 0.15,
                        'opportunity': '中——网络距离近但需要确认语义相关性',
                        'suggested_approach': f'Explore connections between {frontiers[i]["label"]} and '
                                              f'{frontiers[j]["label"]} — methods from one may apply to the other.',
                        'shared_neighbors': [],
                        'score_components': {'bridge_potential': 0.6, 'edge_sparsity': 0.9, 'feasibility': 0.4},
                    })

    gaps.sort(key=lambda x: x['gap_score'], reverse=True)
    for i, g in enumerate(gaps):
        g['rank'] = i + 1
    return gaps[:20]


# ----------------------------------------------------------
#  Gap Scoring (unified)
# ----------------------------------------------------------

def score_all_gaps(density_gaps, bridge_gaps, temporal_gaps, frontiers, paper_count, trends_data):
    """
    Unify and score all gaps. Returns sorted list of scored gaps.
    """
    all_gaps = []

    # Score density gaps
    for g in density_gaps:
        density_norm = 1 - min(1, g['density'] * 5)  # lower density = higher score
        proximity_norm = min(1, g['proximity_to_frontier'])
        growth = g.get('cagr_3yr')
        if growth is None or growth == float('inf') or growth < -1:
            growth_norm = 0.3
        else:
            growth_norm = min(1, max(0, growth))

        score = (
            0.35 * density_norm +
            0.30 * proximity_norm +
            0.20 * growth_norm +
            0.15 * 0.5  # methodology readiness default
        )

        label = f"{g['keyword']} (near {g['nearby_frontier']})"

        all_gaps.append({
            'rank': 0,
            'label': label,
            'gap_type': 'density',
            'description': (
                f"关键词「{g['keyword']}」在语料中仅出现 {g['paper_count']} 次，"
                f"但与前沿「{g['nearby_frontier']}」高度相关（proximity={g['proximity_to_frontier']:.2f}）。"
                f"该领域论文密度极低，可能是被忽视的研究机会。"
            ),
            'density': g['density'],
            'proximity_to_frontier': g['proximity_to_frontier'],
            'nearby_frontier': g['nearby_frontier'],
            'gap_score': round(score, 3),
            'opportunity': '高——切入成本低' if score > 0.6 else '中——需要更多验证',
            'suggested_approach': f"将前沿「{g['nearby_frontier']}」的方法/视角应用到「{g['keyword']}」问题上",
            'keyword': g['keyword'],
            'score_components': {
                'density': round(density_norm, 3),
                'proximity': round(proximity_norm, 3),
                'growth': round(growth_norm, 3),
                'feasibility': 0.5,
            }
        })

    # Score bridge gaps
    for g in bridge_gaps:
        bridge_norm = g['bridge_potential']
        growth_norm = min(1, g['combined_growth'])
        cluster_avg_score = (g['cluster1_score'] + g['cluster2_score']) / 2

        score = (
            0.30 * (1 - min(1, g['edge_ratio'] * 3)) +
            0.25 * cluster_avg_score +
            0.20 * bridge_norm +
            0.15 * growth_norm +
            0.10 * 0.5
        )

        label = f"Bridge: {g['bridging_clusters'][0]} <-> {g['bridging_clusters'][1]}"

        all_gaps.append({
            'rank': 0,
            'label': label,
            'gap_type': 'bridge',
            'description': (
                f"两个活跃前沿「{g['bridging_clusters'][0]}」和「{g['bridging_clusters'][1]}」"
                f"各自快速增长，但互相之间仅 {g['direct_edges']} 条共现边（edge_ratio={g['edge_ratio']:.3f}）。"
                f"它们有 {g['shared_neighbors']} 个共同邻居关键词，表明存在交叉研究机会。"
            ),
            'bridging_clusters': g['bridging_clusters'],
            'shared_neighbors': g['shared_neighbor_examples'][:10],
            'cluster_scores': [g['cluster1_score'], g['cluster2_score']],
            'gap_score': round(score, 3),
            'opportunity': '很高——跨学科桥接' if score > 0.6 else '中——需要确认两端可行性',
            'suggested_approach': (
                f"寻找「{g['bridging_clusters'][0]}」的方法在「{g['bridging_clusters'][1]}」场景中的应用，"
                f"或反之；共用中间关键词：{', '.join(g['shared_neighbor_examples'][:5])}"
            ),
            'score_components': {
                'edge_sparsity': round(1 - min(1, g['edge_ratio'] * 3), 3),
                'frontier_proximity': round(cluster_avg_score, 3),
                'bridge_potential': round(bridge_norm, 3),
                'growth': round(growth_norm, 3),
                'feasibility': 0.5,
            }
        })

    # Score temporal gaps
    for g in temporal_gaps:
        decline_norm = 1 - min(1, g['decline_ratio'] * 2)
        recency_norm = min(1, g['years_since_peak'] / 10)
        score = (
            0.35 * decline_norm +
            0.25 * recency_norm +
            0.20 * 0.3 +  # default interest
            0.20 * 0.5    # default methodology
        )

        label = f"Revival: {g['keyword']} (peak {g['peak_year']})"

        all_gaps.append({
            'rank': 0,
            'label': label,
            'gap_type': 'temporal',
            'description': (
                f"关键词「{g['keyword']}」在 {g['peak_year']} 年达到峰值（{g['peak_count']} 篇）后急剧下降，"
                f"近 3 年仅 {g['late_count']} 篇。若外部技术条件发生变化，该领域可能具备复苏潜力。"
            ),
            'peak_year': g['peak_year'],
            'peak_count': g['peak_count'],
            'decline_ratio': g['decline_ratio'],
            'gap_score': round(score, 3),
            'opportunity': '中——需要评估技术可行性' if score > 0.5 else '低——可能已被替代',
            'suggested_approach': f"调研 {g['peak_year']} 年后的技术进展，判断{'' if 'revival' in label else ''}是否有新的使能技术",
            'keyword': g['keyword'],
            'score_components': {
                'decline': round(decline_norm, 3),
                'recency': round(recency_norm, 3),
                'interest': 0.3,
                'feasibility': 0.5,
            }
        })

    # Sort by gap_score
    all_gaps.sort(key=lambda x: x['gap_score'], reverse=True)

    # Assign ranks
    for i, g in enumerate(all_gaps):
        g['rank'] = i + 1

    return all_gaps


def main():
    init()  # Windows UTF-8 encoding fix
    parser = argparse.ArgumentParser(description='Gap Analysis — Phase 5')
    parser.add_argument('--input', '-i', help='Cleaned corpus CSV')
    parser.add_argument('--cooccurrence', '-c', required=True, help='Co-occurrence CSV')
    parser.add_argument('--frontiers', '-f', required=True, help='Frontiers JSON from Phase 4')
    parser.add_argument('--trends', '-t', required=True, help='Trends JSON from Phase 3')
    parser.add_argument('--output', '-o', required=True, help='Output gaps.json')
    parser.add_argument('--top-n', type=int, default=20, help='Max gaps to output')
    args = parser.parse_args()

    print("=" * 60)
    print("[GAPS]  Phase 5: Research Gap Identification")
    print("=" * 60)

    # Load data
    if args.cooccurrence.endswith('.csv'):
        try:
            import pandas as pd
            cooc_data = pd.read_csv(args.cooccurrence).to_dict('records')
        except ImportError:
            import csv
            with open(args.cooccurrence, 'r', encoding='utf-8-sig') as f:
                cooc_data = list(csv.DictReader(f))
    else:
        print("[ERROR] Co-occurrence must be CSV")
        sys.exit(1)

    with open(args.frontiers, 'r', encoding='utf-8') as f:
        frontiers_data = json.load(f)
    frontiers = frontiers_data.get('frontiers', [])

    with open(args.trends, 'r', encoding='utf-8') as f:
        trends_data = json.load(f)
    years = trends_data.get('years', [])

    # Paper count
    paper_count = 100
    if args.input:
        corpus = load_corpus(args.input)
        if hasattr(corpus, '__len__'):
            paper_count = len(corpus)
        else:
            paper_count = len(list(corpus))
    print(f"   Paper count: {paper_count}")
    print(f"   Frontiers loaded: {len(frontiers)}")
    print(f"   Years: {years[0]}-{years[-1]}" if years else "   Years: N/A")

    # Build network for gap analysis
    G = build_network(cooc_data)
    print(f"   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Burst data (for temporal gap assessment)
    burst_data = {b['keyword']: b for b in frontiers_data.get('burst_keywords', [])}

    # Find gaps of each type
    print("\n   Finding density gaps...")
    density_gaps = find_density_gaps(G, frontiers, trends_data, paper_count)
    print(f"   Density gaps found: {len(density_gaps)}")

    print("   Finding bridge gaps...")
    bridge_gaps = find_bridge_gaps(G, frontiers, trends_data)
    print(f"   Bridge gaps found: {len(bridge_gaps)}")

    print("   Finding temporal gaps...")
    temporal_gaps = find_temporal_gaps(trends_data, years, burst_data, paper_count)
    print(f"   Temporal gaps found: {len(temporal_gaps)}")

    print("   Finding semantic gaps...")
    semantic_gaps = find_semantic_gaps(G, frontiers, trends_data, paper_count)
    print(f"   Semantic gaps found: {len(semantic_gaps)}")

    # Score all gaps
    all_scored = score_all_gaps(density_gaps, bridge_gaps, temporal_gaps, frontiers, paper_count, trends_data)
    # Merge semantic gaps (already scored)
    all_scored.extend(semantic_gaps)
    all_scored.sort(key=lambda x: x['gap_score'], reverse=True)
    # Re-rank
    for i, g in enumerate(all_scored):
        g['rank'] = i + 1
    print("\n   Scoring gaps...")
    all_gaps = score_all_gaps(density_gaps, bridge_gaps, temporal_gaps, frontiers, paper_count, trends_data)

    # Qualitative fallback: when quantitative methods find nothing, infer gaps
    # from frontier community structure
    if not all_gaps and frontiers and G.number_of_nodes() >= 4:
        log('info', 'Quantitative gap detection found nothing — running qualitative inference...')
        inferred = infer_gaps_qualitative(G, frontiers, trends_data, paper_count)
        if inferred:
            all_gaps = inferred
            log('ok', f'Qualitative inference produced {len(all_gaps)} gap hypotheses')

    # Limit to top-n
    top_gaps = all_gaps[:args.top_n]

    # Build output
    output = {
        'gaps': top_gaps,
        'gap_summary': {
            'density_gaps': len(density_gaps),
            'bridge_gaps': len(bridge_gaps),
            'temporal_gaps': len(temporal_gaps),
            'semantic_gaps': len(semantic_gaps),
            'total_scored': len(all_gaps),
            'top_n': len(top_gaps),
        },
        'parameters': {
            'density_threshold': 0.5,
            'bridge_edge_ratio_max': 0.2,
            'temporal_years_needed': 6,
        }
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'-' * 50}")
    print(f"[GAPS]  Gap analysis complete")
    print(f"   Gaps identified: {len(all_gaps)} total")
    print(f"     - Density:  {len(density_gaps)}")
    print(f"     - Bridge:   {len(bridge_gaps)}")
    print(f"     - Temporal: {len(temporal_gaps)}")
    if top_gaps:
        print(f"\n   Top-5 Research Gaps:")
        for i, g in enumerate(top_gaps[:5]):
            gtype = {'density': '[GAPS] ', 'bridge': '[BRIDGE] ', 'temporal': '[TEMPORAL] '}.get(g['gap_type'], '')
            print(f"   {gtype}{i+1}. {g['label']} (Score={g['gap_score']:.3f})")
    print(f"   Output: {output_path}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
