#!/usr/bin/env python3
"""
Phase 6: Report Generation
============================
Generate the final outputs:
  - report.md          : Full Markdown report
  - vis_data.json      : vis-network compatible node/edge data
  - trend_data.json    : Trend line chart data
  - frontier_heatmap.json : Frontier-Gap matrix
  - index.html         : Standalone interactive visualization page (optional)

Usage:
  python report.py --corpus outputs/clean_corpus.csv \
    --frontiers outputs/frontiers.json \
    --gaps outputs/gaps.json \
    --trends outputs/trends.json \
    --cooccurrence outputs/cooccurrence.csv \
    --output-dir outputs/report/
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import init, log, warn, banner, sep, retry_api, adaptive_thresholds, quality_report


# ----------------------------------------------------------
#  Data loading
# ----------------------------------------------------------

def load_cooccurrence(filepath):
    try:
        import pandas as pd
        return pd.read_csv(filepath).to_dict('records')
    except ImportError:
        import csv
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))


def load_corpus(filepath):
    try:
        import pandas as pd
        return pd.read_csv(filepath)
    except ImportError:
        import csv
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))


# ----------------------------------------------------------
#  vis-network data builder
# ----------------------------------------------------------

def build_vis_data(cooc_data, frontiers, gaps, top_n=200):
    """
    Build vis-network JSON with:
      - nodes: keywords sized by frequency, colored by community
      - edges: co-occurrence relationships
      - metadata: frontier/gap annotations
    """
    # Build community color map
    import random
    random.seed(42)
    community_colors = {}
    frontier_keywords = set()

    for i, fr in enumerate(frontiers):
        color = f"hsl({(i * 137) % 360}, 70%, 55%)"
        for kw in fr.get('top_keywords', []):
            if kw not in community_colors:
                community_colors[kw] = {'color': color, 'community': fr.get('label', f'cluster_{i}')}
            frontier_keywords.add(kw)

    # Build node list
    node_ids = set()
    edges_for_vis = []
    edge_weights = []

    for row in cooc_data:
        a = row.get('keyword_a', row.get('source', ''))
        b = row.get('keyword_b', row.get('target', ''))
        w = float(row.get('weight', 1))
        if a and b and w >= 2:
            node_ids.add(a)
            node_ids.add(b)
            edges_for_vis.append({'from': a, 'to': b, 'value': w, 'title': f"共现 {w} 次"})
            edge_weights.append(w)

    # Normalize edge weights
    max_w = max(edge_weights) if edge_weights else 1
    for e in edges_for_vis:
        e['value'] = round(e['value'] / max_w * 5, 2)  # scale to 0-5

    # Build nodes
    # Compute keyword frequencies from co-occurrence
    kw_freq = Counter()
    for row in cooc_data:
        a = row.get('keyword_a', row.get('source', ''))
        b = row.get('keyword_b', row.get('target', ''))
        kw_freq[a] += int(row.get('weight', 1))
        kw_freq[b] += int(row.get('weight', 1))

    gap_keywords = set()
    for g in gaps:
        if g.get('gap_type') == 'density':
            gap_keywords.add(g.get('keyword', ''))

    nodes_for_vis = []
    for node in node_ids:
        freq = kw_freq.get(node, 1)
        size = max(5, min(40, 5 + math.log(freq + 1) * 8))

        comm_info = community_colors.get(node, {})
        color = comm_info.get('color', '#97C2FC')  # default blue

        # Adjust for gap keywords
        border_color = '#2B7CE9'
        border_width = 1
        if node in gap_keywords:
            border_color = '#FF6B6B'
            border_width = 3

        if node in frontier_keywords:
            border_width = max(border_width, 2)

        nodes_for_vis.append({
            'id': node,
            'label': node,
            'value': round(size, 1),
            'group': comm_info.get('community', 'other'),
            'color': {
                'background': color,
                'border': border_color,
            },
            'borderWidth': border_width,
            'title': f"<b>{node}</b><br>频率: {freq}<br>社区: {comm_info.get('community', '未分类')}",
        })

    return {
        'nodes': nodes_for_vis,
        'edges': edges_for_vis,
        'metadata': {
            'n_nodes': len(nodes_for_vis),
            'n_edges': len(edges_for_vis),
            'frontiers': [
                {'label': fr['label'], 'keywords': fr['top_keywords'][:5], 'score': fr['frontier_score']}
                for fr in frontiers[:10]
            ],
            'gaps': [
                {'label': g['label'], 'type': g['gap_type'], 'score': g['gap_score']}
                for g in gaps[:10]
            ],
        },
        'options': {
            'physics': {'solver': 'forceAtlas2Based', 'forceAtlas2Based': {'gravitationalConstant': -50}},
            'edges': {'smooth': False},
        },
    }


# ----------------------------------------------------------
#  Trend data for charts
# ----------------------------------------------------------

def build_trend_chart_data(trends_data, frontiers, gaps):
    """Extract trend data for chart visualization (top keywords)."""
    keywords_data = trends_data.get('keywords', {})
    years = trends_data.get('years', [])

    # Collect frontier and gap keywords
    highlight_kws = set()
    for fr in frontiers:
        for kw in fr.get('top_keywords', [])[:5]:
            highlight_kws.add(kw)
    for g in gaps:
        if g.get('keyword'):
            highlight_kws.add(g['keyword'])
        for kw in g.get('shared_neighbors', [])[:5]:
            highlight_kws.add(kw)

    series = []
    for kw in highlight_kws:
        kd = keywords_data.get(kw, {})
        yearly = kd.get('yearly', {})
        series.append({
            'keyword': kw,
            'total': kd.get('total', 0),
            'trend': kd.get('trend', ''),
            'data': [yearly.get(str(y), 0) for y in years],
        })

    series.sort(key=lambda x: x['total'], reverse=True)

    return {
        'years': years,
        'series': series,
    }


# ----------------------------------------------------------
#  Frontier-Gap heatmap data
# ----------------------------------------------------------

def build_frontier_gap_matrix(frontiers, gaps):
    """Build a matrix showing which gaps relate to which frontiers."""
    frontier_labels = [fr.get('label', f"F{i}") for i, fr in enumerate(frontiers[:10])]
    gap_labels = [g.get('label', f"G{i}") for i, g in enumerate(gaps[:10])]

    matrix = []
    for gi, g in enumerate(gaps[:10]):
        row = []
        for fi, fr in enumerate(frontiers[:10]):
            # Compute relevance: gap keyword near frontier?
            relevance = 0.0
            gap_kw = g.get('keyword', '')
            if gap_kw:
                f_kws = set(fr.get('top_keywords', []))
                if gap_kw in f_kws:
                    relevance = 0.8
                # Check shared neighbors
                if g.get('shared_neighbors'):
                    shared = set(g.get('shared_neighbors', []))
                    f_kw_set = set(fr.get('top_keywords', []))
                    overlap = shared & f_kw_set
                    relevance = max(relevance, min(1.0, len(overlap) / max(1, len(f_kw_set))))

            row.append({
                'frontier': frontier_labels[fi],
                'gap': gap_labels[gi],
                'relevance': round(relevance, 3) if relevance > 0 else 0,
            })
        matrix.append(row)

    return {
        'frontiers': frontier_labels,
        'gaps': gap_labels,
        'matrix': matrix,
    }


# ----------------------------------------------------------
#  Markdown report generator
# ----------------------------------------------------------

def generate_markdown_report(corpus, frontiers, gaps, cooc_data, trends_data, summary, args):
    """Generate the full Markdown report."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    n_papers = len(corpus) if hasattr(corpus, '__len__') else '?'
    years = trends_data.get('years', [])
    year_range = f"{years[0]}-{years[-1]}" if years else "N/A"

    lines = []
    lines.append(f"# 文献计量分析报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {now} | **文献数**: {n_papers} | **时间跨度**: {year_range}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # -- Executive Summary --
    lines.append(f"## 摘要")
    lines.append(f"")
    n_frontiers = len(frontiers)

    # Determine if gaps is a list or wrapped dict
    if isinstance(gaps, dict):
        gap_list = gaps.get('gaps', [])
        n_gaps = sum(gaps.get('gap_summary', {}).values()) if 'gap_summary' in gaps else len(gap_list)
    elif isinstance(gaps, list):
        gap_list = gaps
        n_gaps = len(gaps)
    else:
        gap_list = []
        n_gaps = 0

    lines.append(f"本报告对 **{n_papers}** 篇文献（{year_range}）进行了系统的文献计量分析。")
    lines.append(f"")
    lines.append(f"- 识别到 **{n_frontiers}** 个研究主题簇，其中 {len(frontiers)} 个被评为前沿方向")
    lines.append(f"- 识别到 **{len(gap_list)}** 个潜在研究空白（密度空白 + 桥接空白 + 时间空白）")
    lines.append(f"")
    if frontiers:
        lines.append(f"**[FRONTIER] 最热前沿**: {frontiers[0].get('label', 'N/A')} (FrontierScore={frontiers[0].get('frontier_score', 0)})")
    if gap_list:
        lines.append(f"**[GAPS] 最大空白**: {gap_list[0].get('label', 'N/A')} (GapScore={gap_list[0].get('gap_score', 0)})")
    lines.append(f"")

    # -- Research Landscape --
    lines.append(f"## 1. 研究版图概览")
    lines.append(f"")

    # Keyword stats
    if summary:
        lines.append(f"### 1.1 关键词统计")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 唯一关键词数 | {summary.get('unique_keywords', 'N/A')} |")
        lines.append(f"| 共现关系对数 | {summary.get('cooccurrence_pairs', 'N/A')} |")
        lines.append(f"| 作者合作对数 | {summary.get('author_collab_pairs', 'N/A')} |")
        lines.append(f"")

        top_kws = summary.get('top_keywords_20', [])
        if top_kws:
            lines.append(f"### 1.2 Top-20 关键词")
            lines.append(f"")
            lines.append(f"| # | 关键词 | 频次 | 趋势 |")
            lines.append(f"|---|--------|------|------|")
            for i, kw in enumerate(top_kws[:20]):
                if isinstance(kw, dict):
                    lines.append(f"| {i+1} | {kw.get('keyword', kw.get('keyword', ''))} | {kw.get('count', kw.get('total_count', ''))} | {kw.get('trend', '')} |")
                elif isinstance(kw, (tuple, list)):
                    lines.append(f"| {i+1} | {kw[0]} | {kw[1]} | |")
            lines.append(f"")

    # Year distribution
    if hasattr(corpus, 'year') and years:
        lines.append(f"### 1.3 年份分布")
        lines.append(f"")
        lines.append(f"文献时间跨度为 **{years[0]}** 至 **{years[-1]}**。")
        lines.append(f"")

    # -- Research Frontiers --
    lines.append(f"## 2. 研究前沿")
    lines.append(f"")

    for i, fr in enumerate(frontiers[:10]):
        lines.append(f"### 2.{i+1} {fr.get('label', f'Frontier {i+1}')}")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 前沿评分 | **{fr.get('frontier_score', 0):.3f}** |")
        lines.append(f"| 论文数 | {fr.get('node_count', fr.get('paper_count', 'N/A'))} |")
        lines.append(f"| 平均年份 | {fr.get('avg_year', 'N/A')} |")
        lines.append(f"| 3年增长率 | {fr.get('avg_growth_rate', 0):.1%} |")
        lines.append(f"| 突发强度 | {fr.get('max_burst_intensity', 0):.2f} |")
        lines.append(f"| 核心关键词 | {', '.join(fr.get('top_keywords', [])[:5])} |")
        if fr.get('burst_keywords'):
            lines.append(f"| 突发关键词 | {', '.join(fr.get('burst_keywords', [])[:5])} |")
        lines.append(f"")
        lines.append(f"**解读**: {fr.get('label', '')} 是当前领域的前沿方向，")
        lines.append(f"由 {', '.join(fr.get('top_keywords', [])[:3])} 等关键词驱动。")
        lines.append(f"")

    # -- Research Gaps --
    lines.append(f"## 3. 研究空白")
    lines.append(f"")

    for i, g in enumerate(gap_list[:10]):
        gtype_emoji = {'density': '[GAPS]', 'bridge': '[BRIDGE]', 'temporal': '[TEMPORAL]'}.get(g.get('gap_type', ''), '')
        lines.append(f"### 3.{i+1} {gtype_emoji} {g.get('label', f'Gap {i+1}')}")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 空白评分 | **{g.get('gap_score', 0):.3f}** |")
        lines.append(f"| 空白类型 | {g.get('gap_type', '')} |")
        lines.append(f"| 机会等级 | {g.get('opportunity', '')} |")
        if g.get('gap_type') == 'bridge':
            lines.append(f"| 桥接方向 | {' ↔ '.join(g.get('bridging_clusters', []))} |")
        if g.get('keyword'):
            lines.append(f"| 关键词 | {g['keyword']} |")
        lines.append(f"")
        lines.append(f"**描述**: {g.get('description', '')}")
        lines.append(f"")
        lines.append(f"**建议切入路径**: {g.get('suggested_approach', '')}")
        lines.append(f"")

    # -- Methodology --
    lines.append(f"## 4. 方法附录")
    lines.append(f"")
    lines.append(f"### 4.1 分析流程")
    lines.append(f"")
    lines.append(f"1. **文献摄入**: 从 BibTeX/PDF/API 提取论文元数据")
    lines.append(f"2. **数据清洗**: DOI/标题去重 + 关键词标准化（词干化 + 同义词合并）")
    lines.append(f"3. **文献计量**: 关键词频率 + 共现矩阵 + 逐年趋势")
    lines.append(f"4. **前沿检测**: Kleinberg 突发检测 + Louvain 社区发现 + 多维评分")
    lines.append(f"5. **空白识别**: 密度分析 + 桥接检测 + 时间空洞分析")
    lines.append(f"")
    lines.append(f"### 4.2 关键参数")
    lines.append(f"")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 突发检测 gamma | 1.0 |")
    lines.append(f"| 社区检测算法 | Louvain (greedy fallback) |")
    lines.append(f"| 前沿评分权重 | Burst 0.35 + Recency 0.30 + Growth 0.25 + Cross 0.10 |")
    lines.append(f"| 空白评分权重 | Density 0.35 + Proximity 0.30 + Growth 0.20 + Feasibility 0.15 |")
    lines.append(f"")
    lines.append(f"### 4.3 局限性")
    lines.append(f"")
    lines.append(f"- 关键词分析依赖于论文中标注的关键词质量——若论文关键词标注不规范，会影响共现网络精度")
    lines.append(f"- 突发检测对小样本（<20 篇）的关键词不可靠")
    lines.append(f"- 前沿/空白评分是基于文献计量信号的启发式估计，**不能替代领域专家判断**")
    lines.append(f"- 语料库的覆盖范围（来源数据库、检索策略）决定了分析的视野边界")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*报告由 literature-mining skill 自动生成 | {now}*")
    lines.append(f"")

    return '\n'.join(lines)


# ----------------------------------------------------------
#  Main
# ----------------------------------------------------------

def main():
    init()  # Windows UTF-8 encoding fix
    parser = argparse.ArgumentParser(description='Report Generation — Phase 6')
    parser.add_argument('--corpus', help='Cleaned corpus CSV')
    parser.add_argument('--frontiers', '-f', required=True, help='Frontiers JSON')
    parser.add_argument('--gaps', '-g', required=True, help='Gaps JSON')
    parser.add_argument('--trends', '-t', required=True, help='Trends JSON')
    parser.add_argument('--cooccurrence', '-c', required=True, help='Co-occurrence CSV')
    parser.add_argument('--output-dir', '-o', default='outputs/report', help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("[ANALYZE] Phase 6: Report Generation")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("\n   Loading data...")
    with open(args.frontiers, 'r', encoding='utf-8') as f:
        frontiers_data = json.load(f)
    frontiers = frontiers_data.get('frontiers', [])

    gaps_path = Path(args.gaps)
    if gaps_path.exists():
        with open(gaps_path, 'r', encoding='utf-8') as f:
            gaps_data = json.load(f)
        gaps = gaps_data
    else:
        warn(f'Gaps file not found: {args.gaps} — report will skip gap section')
        gaps_data = {'gaps': [], 'gap_summary': {'density_gaps': 0, 'bridge_gaps': 0, 'temporal_gaps': 0}}
        gaps = gaps_data

    with open(args.trends, 'r', encoding='utf-8') as f:
        trends_data = json.load(f)

    cooc_data = load_cooccurrence(args.cooccurrence)

    # Load summary if available
    summary = {}
    summary_path = Path(args.cooccurrence).parent / 'stats_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    corpus = None
    if args.corpus:
        corpus = load_corpus(args.corpus)

    if isinstance(gaps, dict):
        gap_list = gaps.get('gaps', [])
    elif isinstance(gaps, list):
        gap_list = gaps
    else:
        gap_list = []

    print(f"   Frontiers: {len(frontiers)} | Gaps: {len(gap_list)}")

    # 1. Build vis-network data
    print("\n   Building vis-network data...")
    vis_data = build_vis_data(cooc_data, frontiers, gap_list)
    vis_path = output_dir / 'vis_data.json'
    with open(vis_path, 'w', encoding='utf-8') as f:
        json.dump(vis_data, f, ensure_ascii=False, indent=2)
    print(f"   -> {vis_path} ({vis_data['metadata']['n_nodes']} nodes, {vis_data['metadata']['n_edges']} edges)")

    # 2. Build trend chart data
    print("   Building trend chart data...")
    trend_chart = build_trend_chart_data(trends_data, frontiers, gap_list)
    trend_path = output_dir / 'trend_data.json'
    with open(trend_path, 'w', encoding='utf-8') as f:
        json.dump(trend_chart, f, ensure_ascii=False, indent=2)
    print(f"   -> {trend_path} ({len(trend_chart['series'])} series)")

    # 3. Build frontier-gap matrix
    print("   Building frontier-gap matrix...")
    fg_matrix = build_frontier_gap_matrix(frontiers, gap_list)
    matrix_path = output_dir / 'frontier_heatmap.json'
    with open(matrix_path, 'w', encoding='utf-8') as f:
        json.dump(fg_matrix, f, ensure_ascii=False, indent=2)
    print(f"   -> {matrix_path}")

    # 4. Generate Markdown report
    print("   Generating Markdown report...")
    md_report = generate_markdown_report(corpus, frontiers, gaps_data, cooc_data, trends_data, summary, args)
    md_path = output_dir / 'report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"   -> {md_path}")

    # 5. Build report_context.json for AI-driven visualization
    context_path = output_dir / 'report_context.json'

    papers_for_ctx = []
    if corpus is not None:
        if hasattr(corpus, 'to_dict'):
            records = corpus.to_dict('records')
        else:
            records = corpus
        for r in records[:200]:
            papers_for_ctx.append({
                'title': str(r.get('title', ''))[:200],
                'doi': str(r.get('doi', '')),
                'year': str(r.get('year', '')),
                'authors': str(r.get('authors', ''))[:100],
                'journal': str(r.get('journal', ''))[:100],
                'keywords': str(r.get('keywords_normalized', ''))[:200],
            })

    # Read summary
    summary = {}
    summary_path = output_dir.parent / 'stats_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    context = {
        'corpus_stats': {
            'n_papers': len(papers_for_ctx),
            'year_range': [min(int(p['year']) for p in papers_for_ctx if p['year'].isdigit()),
                          max(int(p['year']) for p in papers_for_ctx if p['year'].isdigit())] if papers_for_ctx else [],
            'n_keywords': len(trend_chart.get('series', [])),
            'n_cooccurrence': len(cooc_data) if isinstance(cooc_data, list) else 0,
        },
        'frontiers': frontiers,
        'gaps': gap_list,
        'network': vis_data,
        'trends': trend_chart,
        'matrix': fg_matrix,
        'papers': papers_for_ctx,
        'top_keywords': summary.get('top_keywords_20', [])[:20],
        'rising_keywords': summary.get('rising_keywords', [])[:10],
        'declining_keywords': summary.get('declining_keywords', [])[:10],
    }

    with open(context_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, ensure_ascii=False, indent=2)
    print(f"   -> {context_path}")

    # 5b. Generate summary.txt for AI to present directly
    summary_txt = _build_summary_txt(context)
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_txt)
    print(f"   -> {summary_path}")

    # Final summary
    print(f"\n{'-' * 50}")
    print(f"[ANALYZE] Report generation complete")
    print(f"   Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   {'[FILE]' if f.suffix == '.md' else '[REPORT]' if f.suffix == '.json' else '[DATA]'} {f.name} ({size_kb:.1f} KB)")
    print(f"{'-' * 50}")

    return context


def _build_summary_txt(ctx):
    """Build a concise text summary that AI can present directly."""
    lines = []
    stats = ctx.get('corpus_stats', {})
    n = stats.get('n_papers', 0)
    yr = stats.get('year_range', [])
    frontiers = ctx.get('frontiers', [])
    gaps = ctx.get('gaps', [])
    top_kw = ctx.get('top_keywords', [])

    lines.append(f"=== Analysis Summary ===")
    lines.append(f"Papers: {n}  |  Years: {yr[0]}-{yr[-1]}" if yr else f"Papers: {n}")
    lines.append(f"Frontiers: {len(frontiers)}  |  Gaps: {len(gaps)}")
    lines.append("")

    # Data quality + overall confidence (computed inline from stats)
    yr_span = (yr[1] - yr[0] + 1) if yr and len(yr) >= 2 else 0
    n_abstracts = sum(1 for p in ctx.get('papers', []) if p.get('abstract') and len(p.get('abstract', '')) > 50)
    abstract_pct = n_abstracts / max(1, n) if n else 0
    warnings = []
    if n < 50: warnings.append(f'Small corpus ({n} papers, recommend >= 50)')
    if yr_span < 5: warnings.append(f'Narrow span ({yr_span}yr, recommend >= 5)')
    if abstract_pct < 0.5: warnings.append(f'Low abstract coverage ({abstract_pct*100:.0f}%)')
    n_warnings = len(warnings)
    verdict = 'good' if n_warnings == 0 else ('fair' if n_warnings <= 2 else 'poor')
    confidence = {'good': 'HIGH', 'fair': 'MEDIUM', 'poor': 'LOW'}[verdict]

    lines.append(f"Confidence: {confidence} ({n} papers, {yr_span}yr span, {abstract_pct*100:.0f}% abstracts)")
    for w in warnings:
        lines.append(f"  Warning: {w}")
    lines.append("")

    # Frontiers (with per-item confidence)
    lines.append("--- Frontiers ---")
    for i, f in enumerate(frontiers[:5]):
        kws = ', '.join(f.get('top_keywords', [])[:4])
        score = f.get('frontier_score', 0)
        item_conf = 'HIGH' if score >= 0.5 else 'MEDIUM'
        lines.append(f"  #{i+1} {f.get('label', '')}  (score={score:.3f}) [{item_conf}]")
        lines.append(f"     Keywords: {kws}")
    lines.append("")

    # Gaps (with per-item confidence + source tag)
    lines.append("--- Gaps ---")
    gap_types = {'density': 'Density', 'bridge': 'Bridge', 'temporal': 'Temporal', 'semantic': 'Semantic'}
    for i, g in enumerate(gaps[:5]):
        gt = gap_types.get(g.get('gap_type', ''), g.get('gap_type', ''))
        score = g.get('gap_score', 0)
        is_qualitative = g.get('gap_type') == 'density' and 'Broker' in g.get('label', '')
        if is_qualitative:
            item_conf = 'LOW (qualitative inference)'
        elif score >= 0.5:
            item_conf = 'HIGH'
        elif score >= 0.3:
            item_conf = 'MEDIUM'
        else:
            item_conf = 'LOW'
        lines.append(f"  #{i+1} [{gt}] {g.get('label', '')}  (score={score:.3f}) [{item_conf}]")
        lines.append(f"     {g.get('description', '')[:150]}")
        approach = g.get('suggested_approach', '')
        if approach:
            lines.append(f"     Approach: {approach[:150]}")
    lines.append("")

    # Top keywords
    lines.append("--- Top Keywords ---")
    for kw in top_kw[:10]:
        if isinstance(kw, dict):
            lines.append(f"  {kw.get('keyword', '')}: {kw.get('count', '')} ({kw.get('trend', '')})")
        elif isinstance(kw, (list, tuple)):
            lines.append(f"  {kw[0]}: {kw[1]}")
    lines.append("")

    # Next steps
    lines.append("--- Next Steps ---")
    lines.append("  [1] Deep-dive a specific gap or frontier")
    lines.append("  [2] Expand corpus via search or citation chasing and re-run")
    lines.append("  [3] Export data for external analysis (report_context.json)")

    return '\n'.join(lines)


def generate_html(html_path, vis_data, trend_chart, fg_matrix, papers=None):
    """Generate an enhanced standalone HTML page with full interactive visualization."""

    if papers is None:
        papers = []

    # Precompute dashboard stats
    n_nodes = vis_data['metadata']['n_nodes']
    n_edges = vis_data['metadata']['n_edges']
    n_frontiers = len(vis_data['metadata'].get('frontiers', []))
    n_gaps = len(vis_data['metadata'].get('gaps', []))
    n_papers = len(papers)
    year_range = f"{trend_chart['years'][0]}-{trend_chart['years'][-1]}" if trend_chart.get('years') else 'N/A'
    rising_kw = sum(1 for s in trend_chart.get('series', []) if s.get('trend') == 'rising')

    _C = '"""'  # quote trick

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Literature Mining — Research Landscape</title>
<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
<link href="https://unpkg.com/vis-network@9.1.2/dist/dist/vis-network.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
:root{{
  --bg:#f8fafc; --surface:#fff; --text:#0f172a; --text-muted:#64748b; --text-faint:#94a3b8;
  --border:#e2e8f0; --border-light:#f1f5f9;
  --accent:#ef4444; --accent-soft:#fef2f2; --accent-ring:#fca5a5;
  --teal:#0d9488; --teal-soft:#f0fdfa; --teal-ring:#5eead4;
  --amber:#d97706; --amber-soft:#fffbeb;
  --slate:#475569; --slate-light:#f1f5f9;
  --brand:#1e293b; --brand-light:#334155;
  --shadow-sm:0 1px 2px rgba(0,0,0,.05); --shadow:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.05);
  --shadow-lg:0 10px 15px -3px rgba(0,0,0,.08),0 4px 6px -4px rgba(0,0,0,.04);
  --radius-sm:6px; --radius:10px; --radius-lg:14px;
  --node-default:#6366f1;
}}
@media (prefers-color-scheme:dark){{
  :root{{
    --bg:#0f172a; --surface:#1e293b; --text:#e2e8f0; --text-muted:#94a3b8; --text-faint:#64748b;
    --border:#334155; --border-light:#1e293b;
    --accent-soft:#450a0a; --accent-ring:#7f1d1d;
    --teal-soft:#042f2e; --teal-ring:#134e4a;
    --amber-soft:#451a03;
    --slate-light:#1e293b;
    --shadow-sm:0 1px 2px rgba(0,0,0,.3); --shadow:0 4px 6px -1px rgba(0,0,0,.4),0 2px 4px -2px rgba(0,0,0,.3);
    --shadow-lg:0 10px 15px -3px rgba(0,0,0,.5),0 4px 6px -4px rgba(0,0,0,.3);
    --node-default:#818cf8;
  }}
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;-webkit-font-smoothing:antialiased;transition:background .3s,color .3s}}

/* ── Header ── */
.header{{background:linear-gradient(135deg,var(--brand) 0%,var(--brand-light) 100%);color:#fff;padding:18px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;position:relative;z-index:30}}
.header h1{{font-size:20px;font-weight:700;letter-spacing:-.02em}}
.header .subtitle{{color:rgba(255,255,255,.55);font-size:12px;margin-top:2px;font-weight:400}}
.header-actions{{display:flex;gap:6px}}
.header-actions button{{padding:7px 16px;border:1px solid rgba(255,255,255,.2);border-radius:var(--radius-sm);background:rgba(255,255,255,.08);color:rgba(255,255,255,.85);cursor:pointer;font-size:12px;font-weight:500;transition:all .2s;backdrop-filter:blur(4px)}}
.header-actions button:hover{{background:rgba(255,255,255,.15);border-color:rgba(255,255,255,.35)}}

/* ── Dashboard ── */
#dashboard{{position:absolute;top:80px;left:0;right:0;z-index:10;display:flex;gap:12px;padding:12px 28px;flex-wrap:wrap;pointer-events:none}}
#dashboard .stat-card{{pointer-events:auto;background:var(--surface);border-radius:var(--radius);padding:14px 18px;box-shadow:var(--shadow);min-width:110px;flex:1;max-width:170px;border:1px solid var(--border);cursor:pointer;transition:all .2s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}}
#dashboard .stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent);opacity:0;transition:opacity .2s}}
#dashboard .stat-card:hover{{transform:translateY(-3px);box-shadow:var(--shadow-lg)}}
#dashboard .stat-card:hover::before{{opacity:1}}
#dashboard .stat-card.teal::before{{background:var(--teal)}}
.stat-card .value{{font-size:32px;font-weight:800;letter-spacing:-.02em;color:var(--text);line-height:1.1}}
.stat-card .value.accent{{color:var(--accent)}}
.stat-card .value.teal{{color:var(--teal)}}
.stat-card .label{{font-size:11px;color:var(--text-muted);margin-top:3px;font-weight:500;text-transform:uppercase;letter-spacing:.05em}}
.stat-card .warn{{font-size:10px;color:var(--amber);margin-top:4px}}

/* ── Main layout ── */
.container{{display:flex;height:calc(100vh - 80px);position:relative}}
#network{{flex:1;background:var(--surface);position:relative;border-right:1px solid var(--border)}}
#network canvas{{border-radius:0}}
.sidebar{{width:400px;overflow-y:auto;padding:16px 18px;background:var(--surface);scrollbar-width:thin;scrollbar-color:var(--border) transparent}}
.sidebar::-webkit-scrollbar{{width:5px}}
.sidebar::-webkit-scrollbar-thumb{{background:var(--border);border-radius:10px}}
@media(max-width:768px){{.container{{flex-direction:column}}#network{{height:45vh;border-right:none;border-bottom:1px solid var(--border)}}.sidebar{{width:100%;height:55vh}}#dashboard{{top:56px;padding:8px 12px;gap:6px}}.stat-card{{min-width:60px;padding:10px 12px;max-width:120px}}.stat-card .value{{font-size:22px}}.header{{padding:12px 16px}}.header h1{{font-size:16px}}}}

/* ── Search ── */
.search-wrap{{position:relative;margin-bottom:14px}}
#searchBox{{width:100%;padding:10px 14px 10px 36px;border:1.5px solid var(--border);border-radius:var(--radius);font-size:13px;background:var(--bg);color:var(--text);outline:none;transition:border-color .2s,box-shadow .2s}}
#searchBox:focus{{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}}
.search-wrap::before{{content:'\\1F50D';position:absolute;left:11px;top:50%;transform:translateY(-50%);font-size:14px;opacity:.5}}
#searchResults{{font-size:11px;color:var(--text-muted);margin:4px 0 0 4px;min-height:16px}}

/* ── Year slider ── */
.year-slider{{display:flex;align-items:center;gap:10px;margin-bottom:14px;font-size:11px;font-weight:500}}
.year-slider input[type=range]{{flex:1;accent-color:var(--accent);height:4px}}
.year-slider span{{color:var(--text-muted);min-width:32px;text-align:center;font-variant-numeric:tabular-nums}}

/* ── Tabs ── */
.tab-bar{{display:flex;gap:2px;margin-bottom:16px;background:var(--slate-light);border-radius:var(--radius);padding:3px}}
.tab-bar button{{flex:1;padding:8px 6px;border:none;border-radius:var(--radius-sm);background:transparent;cursor:pointer;font-size:11px;font-weight:600;color:var(--text-muted);transition:all .2s;white-space:nowrap}}
.tab-bar button.active{{background:var(--surface);color:var(--text);box-shadow:var(--shadow-sm)}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}

/* ── Cards ── */
.frontier-item,.gap-item{{margin-bottom:10px;padding:12px 14px;border-radius:var(--radius);border:1px solid var(--border);cursor:pointer;transition:all .2s;background:var(--surface);position:relative}}
.frontier-item{{border-left:3px solid var(--accent)}}
.gap-item{{border-left:3px solid var(--teal)}}
.frontier-item:hover,.gap-item:hover{{box-shadow:var(--shadow);border-color:var(--accent-ring)}}
.gap-item:hover{{border-color:var(--teal-ring)}}
.frontier-item .name,.gap-item .name{{font-weight:600;font-size:13px;margin-bottom:3px;line-height:1.3}}
.score{{font-weight:700;font-size:12px;display:inline-flex;align-items:center;gap:4px}}
.score::before{{content:'';width:8px;height:8px;border-radius:50%;display:inline-block}}
.score.high::before{{background:var(--accent)}}
.score.high{{color:var(--accent)}}
.score.medium::before{{background:var(--amber)}}
.score.medium{{color:var(--amber)}}
.score.low::before{{background:var(--text-faint)}}
.score.low{{color:var(--text-muted)}}
.meta{{font-size:11px;color:var(--text-muted);margin-top:3px;line-height:1.4}}
.detail{{display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border-light);font-size:11px}}
.detail.open{{display:block}}
.detail .kw-tag{{display:inline-block;background:var(--accent-soft);color:var(--accent);padding:3px 8px;border-radius:20px;margin:3px 3px 3px 0;font-size:10px;font-weight:500}}
.detail .gap-tag{{background:var(--teal-soft);color:var(--teal)}}
.rank-badge{{position:absolute;top:-6px;right:12px;background:var(--text);color:var(--surface);font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;opacity:.15}}

/* ── Score breakdown ── */
.score-bars{{margin-top:8px}}
.score-bar-row{{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:10px}}
.score-bar-label{{width:50px;text-align:right;color:var(--text-muted);flex-shrink:0}}
.score-bar-track{{flex:1;height:5px;background:var(--border-light);border-radius:10px;overflow:hidden}}
.score-bar-fill{{height:100%;border-radius:10px;transition:width .6s cubic-bezier(.4,0,.2,1)}}
.score-bar-fill.red{{background:var(--accent)}}
.score-bar-fill.teal{{background:var(--teal)}}
.score-bar-val{{width:30px;font-weight:600;color:var(--text-muted);flex-shrink:0}}

/* ── Heatmap ── */
#heatmapContainer table{{border-collapse:separate;border-spacing:2px;width:100%}}
#heatmapContainer th{{font-size:9px;color:var(--text-muted);padding:2px;writing-mode:vertical-rl;text-orientation:mixed;vertical-align:bottom;max-height:80px}}
#heatmapContainer td{{padding:8px 6px;text-align:center;font-size:10px;font-weight:600;border-radius:4px;min-width:36px;color:#fff}}

/* ── Papers list ── */
#papersList .year-group{{margin-bottom:12px}}
#papersList .year-header{{font-weight:700;font-size:13px;color:var(--text);margin-bottom:4px;display:flex;align-items:center;gap:8px}}
#papersList .year-header span{{font-size:10px;color:var(--text-muted);font-weight:400;background:var(--slate-light);padding:2px 8px;border-radius:10px}}
#papersList .paper-row{{font-size:11px;padding:5px 8px;border-radius:4px;cursor:pointer;transition:background .15s;color:var(--text);border-bottom:1px solid var(--border-light);display:flex;justify-content:space-between;align-items:center}}
#papersList .paper-row:hover{{background:var(--accent-soft)}}
#papersList .paper-row .paper-doi{{font-size:9px;color:var(--text-faint);flex-shrink:0;margin-left:12px}}

/* ── Drill-down ── */
#drilldown{{position:absolute;top:20px;right:20px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;box-shadow:var(--shadow-lg);z-index:20;max-width:340px;max-height:55vh;overflow-y:auto;display:none;font-size:12px;animation:slideIn .2s ease}}
#drilldown.open{{display:block}}
@keyframes slideIn{{from{{opacity:0;transform:translateY(-8px)}}to{{opacity:1;transform:translateY(0)}}}}
#drilldown h3{{font-size:14px;margin-bottom:10px;font-weight:700}}
#drilldown .close{{position:absolute;top:10px;right:14px;cursor:pointer;font-size:18px;color:var(--text-muted);width:24px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:50%;transition:background .2s}}
#drilldown .close:hover{{background:var(--slate-light)}}
#drilldown .paper-link{{display:block;padding:5px 8px;margin:2px 0;color:var(--text);text-decoration:none;font-size:11px;border-radius:4px;transition:background .15s}}
#drilldown .paper-link:hover{{background:var(--accent-soft);color:var(--accent)}}

/* ── Loading ── */
#loading{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:30;text-align:center;display:none;background:var(--surface);padding:32px;border-radius:var(--radius-lg);box-shadow:var(--shadow-lg)}}
#loading .spinner{{width:36px;height:36px;border:3px solid var(--border);border-top:3px solid var(--accent);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 12px}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
#loading p{{color:var(--text-muted);font-size:12px;font-weight:500}}

/* ── Empty state ── */
.empty{{text-align:center;padding:32px 16px;color:var(--text-muted);font-size:12px}}
.empty .icon{{font-size:32px;margin-bottom:8px;opacity:.4}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Research Landscape</h1>
    <p class="subtitle" id="metadata">{n_nodes} keywords  ·  {n_edges} co-occurrences  ·  {n_frontiers} frontiers  ·  {n_gaps} gaps  ·  {year_range}</p>
  </div>
  <div class="header-actions">
    <button onclick="exportPNG()">Export PNG</button>
    <button onclick="exportCSV()">Export CSV</button>
  </div>
</div>

<div id="dashboard">
  <div class="stat-card" onclick="switchTab('frontiers')">
    <div class="value accent">{n_frontiers}</div><div class="label">Frontiers</div>
  </div>
  <div class="stat-card teal" onclick="switchTab('gaps')">
    <div class="value teal">{n_gaps}</div><div class="label">Research Gaps</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_nodes}</div><div class="label">Keywords</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_edges}</div><div class="label">Co-occurrences</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_papers}</div><div class="label">Papers</div>
  </div>
  <div class="stat-card">
    <div class="value">{year_range}</div><div class="label">Year Range</div>
  </div>
</div>

<div class="container">
  <div id="network">
    <div id="loading"><div class="spinner"></div><p>Rendering network...</p></div>
    <div id="drilldown"><span class="close" onclick="closeDrilldown()">&times;</span><div id="drilldown-content"></div></div>
  </div>
  <div class="sidebar">
    <div class="search-wrap">
      <input type="text" id="searchBox" placeholder="Search keywords, frontiers, gaps, papers...">
    </div>
    <div id="searchResults"></div>
    <div class="year-slider">
      <span id="yearMin">--</span>
      <input type="range" id="yearSlider" min="0" max="100" value="100" oninput="filterByYear(this.value)">
      <span id="yearMax">--</span>
    </div>
    <div class="tab-bar">
      <button class="active" onclick="switchTab('frontiers')">Frontiers</button>
      <button onclick="switchTab('gaps')">Gaps</button>
      <button onclick="switchTab('trends')">Trends</button>
      <button onclick="switchTab('heatmap')">Heatmap</button>
      <button onclick="switchTab('papers')">Papers</button>
    </div>
    <div id="frontiers-tab" class="tab-content active"></div>
    <div id="gaps-tab" class="tab-content"></div>
    <div id="trends-tab" class="tab-content"><canvas id="trendChart" width="360" height="260"></canvas></div>
    <div id="heatmap-tab" class="tab-content"><div id="heatmapContainer"></div></div>
    <div id="papers-tab" class="tab-content"><div id="papersList"></div></div>
  </div>
</div>

<script>
const VIS = {json.dumps(vis_data, ensure_ascii=False)};
const TREND = {json.dumps(trend_chart, ensure_ascii=False)};
const MATRIX = {json.dumps(fg_matrix, ensure_ascii=False)};
const PAPERS = {json.dumps(papers, ensure_ascii=False)};

// ── Init ──
const COLORS = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F'];
let network, trendChart, yearRange = TREND.years||[];
let selectedYear = yearRange.length ? yearRange[yearRange.length-1] : 2025;

document.getElementById('metadata').textContent = `Nodes: ${{VIS.metadata.n_nodes}} | Edges: ${{VIS.metadata.n_edges}} | Frontiers: ${{VIS.metadata.frontiers.length}} | Gaps: ${{VIS.metadata.gaps.length}}`;

// Year slider
if(yearRange.length){{
  const slider=document.getElementById('yearSlider');
  slider.max=yearRange.length-1; slider.value=yearRange.length-1;
  document.getElementById('yearMin').textContent=yearRange[0];
  document.getElementById('yearMax').textContent=yearRange[yearRange.length-1];
}}

// ── Network ──
function buildNetwork(){{
  document.getElementById('loading').style.display='block';
  setTimeout(()=>{{
    const container=document.getElementById('network');
    const nodesArr=VIS.nodes.map(n=>({{...n,color:{{background:n.color?.background||'#97C2FC',border:n.borderColor||'#2B7CE9',highlight:{{background:n.color?.background||'#97C2FC',border:'#FF6B6B'}}}}}}));
    const edgesArr=VIS.edges.map(e=>({{...e,color:{{color:'#cccccc',highlight:'#FF6B6B',opacity:0.6}}}}));
    const nodes=new vis.DataSet(nodesArr);
    const edges=new vis.DataSet(edgesArr);
    network=new vis.Network(container,{{nodes,edges}},{{
      physics:{{solver:'forceAtlas2Based',forceAtlas2Based:{{gravitationalConstant:-50,springLength:150}}}},
      edges:{{smooth:false}},
      nodes:{{font:{{size:11}},scaling:{{min:5,max:40}}}},
      interaction:{{hover:true,tooltipDelay:100}},
    }});
    network.on('click',function(p){{ if(p.nodes.length) showDrilldown(p.nodes[0]); }});
    document.getElementById('loading').style.display='none';
    renderSidebar();
  }},50);
}}

// ── Sidebar rendering ──
function renderSidebar(){{
  const fl=VIS.metadata.frontiers||[];
  const gl=VIS.metadata.gaps||[];
  document.getElementById('frontiers-tab').innerHTML=fl.map((f,i)=>`<div class="frontier-item" onclick="expandItem(this);focusNode('${{(f.keywords||[])[0]||f.label}}')">
    <span class="rank-badge">${{i+1}}</span>
    <div class="name">${{f.label}}</div>
    <div class="score ${{f.score>0.7?'high':f.score>0.4?'medium':'low'}}">${{f.score.toFixed(3)}}</div>
    <div class="meta">${{(f.keywords||[]).slice(0,5).join('  ·  ')}}</div>
    <div class="detail">
      <div class="score-bars">
        <div class="score-bar-row"><span class="score-bar-label">Recency</span><span class="score-bar-track"><span class="score-bar-fill red" style="width:${{Math.round((f.recency_norm||0)*100)}}%"></span></span><span class="score-bar-val">${{((f.recency_norm||0)*100).toFixed(0)}}%</span></div>
        <div class="score-bar-row"><span class="score-bar-label">Growth</span><span class="score-bar-track"><span class="score-bar-fill red" style="width:${{Math.round((f.growth_norm||0)*100)}}%"></span></span><span class="score-bar-val">${{((f.growth_norm||0)*100).toFixed(0)}}%</span></div>
        <div class="score-bar-row"><span class="score-bar-label">Burst</span><span class="score-bar-track"><span class="score-bar-fill red" style="width:${{Math.round((f.burst_norm||0)*100)}}%"></span></span><span class="score-bar-val">${{((f.burst_norm||0)*100).toFixed(0)}}%</span></div>
        <div class="score-bar-row"><span class="score-bar-label">Cross</span><span class="score-bar-track"><span class="score-bar-fill red" style="width:${{Math.round((f.cross_norm||0)*100)}}%"></span></span><span class="score-bar-val">${{((f.cross_norm||0)*100).toFixed(0)}}%</span></div>
      </div>
      <div style="margin-top:6px">${{(f.keywords||[]).map(k=>'<span class="kw-tag">'+k+'</span>').join(' ')}}</div>
    </div>
  </div>`).join('');
  document.getElementById('gaps-tab').innerHTML=gl.map((g,i)=>`<div class="gap-item" onclick="expandItem(this);focusNode('${{g.label}}')">
    <span class="rank-badge">${{i+1}}</span>
    <div class="name">${{g.label}}</div>
    <div class="score ${{g.score>0.7?'high':g.score>0.4?'medium':'low'}}">${{g.score.toFixed(3)}}</div>
    <div class="meta">${{g.type||'unknown'}} gap</div>
    <div class="detail"><div class="meta">Keywords:</div>${{(g.keywords||[]).map(k=>'<span class="kw-tag gap-tag">'+k+'</span>').join(' ')}}</div>
  </div>`).join('');
  renderHeatmap();
}}

// ── Expand item ──
function expandItem(el){{ el.querySelector('.detail').classList.toggle('open'); }}

// ── Focus node ──
function focusNode(label){{
  const found=VIS.nodes.find(n=>n.label===label||n.id===label);
  if(found&&network){{ network.selectNodes([found.id]); network.focus(found.id,{{scale:1.5,animation:true}}); }}
}}

// ── Drill-down ──
function showDrilldown(nodeId){{
  const node=VIS.nodes.find(n=>n.id===nodeId);
  if(!node) return;
  const ts=TREND.series.find(s=>s.keyword===nodeId||s.keyword===node.label);
  let html=`<h3>${{node.label||nodeId}}</h3>`;
  if(node.title) html+=`<p style="color:var(--text-secondary)">${{node.title}}</p>`;
  if(ts){{ html+=`<p>Total: ${{ts.total}} | Trend: ${{ts.trend||'N/A'}}</p>`;
    html+=`<div class="score-bar"><div class="score-bar-fill" style="width:${{Math.min(100,ts.total*2)}}%;background:var(--accent)"></div></div>`; }}
  html+=`<p style="margin-top:8px;color:var(--text-secondary)">Sample papers (from corpus):</p>`;
  const kw=node.label||nodeId;
  const related=PAPERS.filter(p=>p.title.toLowerCase().includes(kw.toLowerCase())).slice(0,5);
  if(related.length){{
    related.forEach(p=>html+=`<span class="paper-link" title="${{p.title}}">${{p.title.slice(0,80)}} (${{p.year}})</span>`);
  }}else{{
    html+='<span style="color:var(--text-secondary);font-size:11px">No matching papers in corpus</span>';
  }}
  document.getElementById('drilldown-content').innerHTML=html;
  document.getElementById('drilldown').classList.add('open');
}}
function closeDrilldown(){{ document.getElementById('drilldown').classList.remove('open'); }}

// ── Year filter ──
function filterByYear(val){{
  const idx=parseInt(val);
  if(!yearRange.length) return;
  selectedYear=yearRange[idx];
  document.getElementById('yearMin').textContent=yearRange[0];
  document.getElementById('yearMax').textContent=selectedYear;
  // Gray out nodes with edges only after selectedYear (simplified: all visible for now)
}}

// ── Search ──
let searchTimeout;
document.getElementById('searchBox').addEventListener('input',function(e){{
  clearTimeout(searchTimeout);
  searchTimeout=setTimeout(()=>{{
    const q=e.target.value.toLowerCase().trim();
    const res=document.getElementById('searchResults');
    if(!q){{ res.textContent=''; network&&network.unselectAll(); return; }}
    const matched=[];
    VIS.nodes.forEach(n=>{{ if(n.id.toLowerCase().includes(q)||(n.label||'').toLowerCase().includes(q)) matched.push(n.id); }});
    VIS.metadata.frontiers.forEach((f,i)=>{{ if(f.label.toLowerCase().includes(q)) matched.push('F'+i); }});
    VIS.metadata.gaps.forEach((g,i)=>{{ if(g.label.toLowerCase().includes(q)) matched.push('G'+i); }});
    res.textContent=`Found ${{matched.filter(m=>!m.startsWith('F')&&!m.startsWith('G')).length}} keywords, ${{matched.filter(m=>m.startsWith('F')).length}} frontiers, ${{matched.filter(m=>m.startsWith('G')).length}} gaps`;
    if(matched.length&&network){{
      const nodeIds=matched.filter(m=>!m.startsWith('F')&&!m.startsWith('G'));
      if(nodeIds.length){{ network.selectNodes(nodeIds); network.focus(nodeIds[0],{{scale:1.3,animation:true}}); }}
    }}
  }},200);
}});

// ── Tabs ──
function switchTab(tab){{
  document.querySelectorAll('.tab-bar button').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  const idx={{frontiers:0,gaps:1,trends:2,heatmap:3,papers:4}}[tab]||0;
  document.querySelectorAll('.tab-bar button')[idx].classList.add('active');
  document.getElementById(tab+'-tab').classList.add('active');
  if(tab==='trends') renderTrendChart();
  if(tab==='heatmap') renderHeatmap();
}}

// ── Trend chart ──
function renderTrendChart(){{
  if(trendChart) return;
  const ctx=document.getElementById('trendChart').getContext('2d');
  const years=TREND.years||[];
  const top=TREND.series.slice(0,6);
  trendChart=new Chart(ctx,{{
    type:'line',
    data:{{
      labels:years,
      datasets:top.map((s,i)=>({{
        label:s.keyword, data:s.data, borderColor:COLORS[i%COLORS.length],
        backgroundColor:COLORS[i%COLORS.length]+'22', fill:false, tension:.3, pointRadius:3
      }}))
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{position:'bottom',labels:{{boxWidth:10,font:{{size:10}}}}}} }},
      scales:{{ x:{{title:{{display:true,text:'Year'}}}}, y:{{title:{{display:true,text:'Papers (normalized)'}},beginAtZero:true}} }}
    }}
  }});
}}

// ── Heatmap ──
function renderHeatmap(){{
  const container=document.getElementById('heatmapContainer');
  const fl=MATRIX.frontiers||[];
  const gl=MATRIX.gaps||[];
  if(!fl.length||!gl.length){{ container.innerHTML='<p style="font-size:12px;color:var(--text-secondary)">No frontier-gap data available.</p>'; return; }}
  let h='<div style="overflow-x:auto;font-size:11px"><table style="border-collapse:collapse;width:100%"><tr><th></th>';
  fl.forEach(f=>h+=`<th style="writing-mode:vertical-rl;padding:4px;font-size:10px;color:var(--text-secondary)">${{f}}</th>`);
  h+='</tr>';
  MATRIX.matrix.forEach((row,i)=>{{
    h+=`<tr><td style="font-size:10px;color:var(--text-secondary);white-space:nowrap;max-width:80px;overflow:hidden;text-overflow:ellipsis">${{gl[i]||''}}</td>`;
    row.forEach(cell=>{{
      const v=cell.relevance||0;
      const alpha=v>.5?v>.8?.9:.7:v>.2?.4:.15;
      h+=`<td style="background:rgba(255,107,107,${{alpha}});padding:8px 4px;text-align:center;font-size:10px;font-weight:600">${{v.toFixed(2)}}</td>`;
    }});
    h+='</tr>';
  }});
  h+='</table></div>';
  container.innerHTML=h;
}}

// ── Papers list ──
function renderPapers(){{
  const container=document.getElementById('papersList');
  if(!PAPERS.length){{ container.innerHTML='<p style="font-size:12px;color:var(--text-secondary)">No paper metadata available. Use CSV/BibTeX input for best results.</p>'; return; }}
  // Group by year
  const byYear={{}};
  PAPERS.forEach(p=>{{ const y=p.year||'?'; if(!byYear[y]) byYear[y]=[]; byYear[y].push(p); }});
  const years=Object.keys(byYear).sort().reverse();
  let h='';
  years.forEach(y=>{{
    h+=`<div class="year-group"><div class="year-header">${{y}} <span>${{byYear[y].length}} papers</span></div>`;
    byYear[y].forEach(p=>{{
      const doi=p.doi&&p.doi!='nan'?p.doi:'';
      h+=`<div class="paper-row" onclick="focusNode('${{p.title.slice(0,30)}}')" title="${{p.title}}">${{p.title.slice(0,90)}}<span class="paper-doi">${{doi.slice(0,25)}}</span></div>`;
    }});
    h+=`</div>`;
  }});
  container.innerHTML=h;
}}
if(PAPERS.length){{ renderPapers(); }}

// ── Export ──
function exportPNG(){{
  if(!network) return;
  const c=document.getElementById('network');
  html2canvas(c,{{backgroundColor:'#fff'}}).then(canvas=>{{
    const a=document.createElement('a'); a.download='network.png'; a.href=canvas.toDataURL(); a.click();
  }});
}}
function exportCSV(){{
  let csv='type,label,score,keywords\\n';
  VIS.metadata.frontiers.forEach(f=>csv+=`frontier,"${{f.label}}",${{f.score}},"${{(f.keywords||[]).join(';')}}"\\n`);
  VIS.metadata.gaps.forEach(g=>csv+=`gap,"${{g.label}}",${{g.score}},`+'\\n');
  const b=new Blob([csv],{{type:'text/csv'}});
  const a=document.createElement('a'); a.download='results.csv'; a.href=URL.createObjectURL(b); a.click();
}}

// ── Start ──
buildNetwork();
</script>
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
