# 文献计量分析报告

**生成时间**: {{generated_at}} | **文献数**: {{n_papers}} | **时间跨度**: {{year_range}}

---

## 摘要

本报告对 **{{n_papers}}** 篇文献（{{year_range}}）进行了系统的文献计量分析，利用关键词共现网络、突发检测、社区发现和密度分析，自动识别研究前沿和潜在空白。

| 指标 | 数值 |
|------|------|
| 唯一关键词 | {{unique_keywords}} |
| 共现关系 | {{cooccurrence_pairs}} |
| 识别前沿 | {{n_frontiers}} |
| 识别空白 | {{n_gaps}} |

**🔥 最热前沿**: {{top_frontier_label}} (FrontierScore={{top_frontier_score}})
**🕳️ 最大空白**: {{top_gap_label}} (GapScore={{top_gap_score}})

---

## 1. 研究版图概览

### 1.1 Top-20 关键词

| # | 关键词 | 频次 | 趋势 |
|---|--------|------|------|
{{#top_keywords}}
| {{rank}} | {{keyword}} | {{count}} | {{trend}} |
{{/top_keywords}}

### 1.2 上升关键词

{{rising_keywords}}

### 1.3 下降关键词

{{declining_keywords}}

---

## 2. 研究前沿

{{#frontiers}}
### 2.{{rank}} {{label}}

| 指标 | 值 |
|------|----|
| 前沿评分 | **{{frontier_score}}** |
| 论文数 | {{node_count}} |
| 平均发表年份 | {{avg_year}} |
| 3年增长率 | {{avg_growth_rate_pct}} |
| 最大突发强度 | {{max_burst_intensity}} |
| 核心关键词 | {{top_keywords_csv}} |
| 突发关键词 | {{burst_keywords_csv}} |

**解读**: {{label}} 是当前领域的前沿方向，由 {{top_keywords_3}} 等关键词驱动。近 3 年增长率 {{avg_growth_rate_pct}}，表明该方向正处于快速上升期。

{{/frontiers}}

---

## 3. 研究空白

{{#gaps}}
### 3.{{rank}} {{gap_emoji}} {{label}}

| 指标 | 值 |
|------|----|
| 空白评分 | **{{gap_score}}** |
| 空白类型 | {{gap_type}} |
| 机会等级 | {{opportunity}} |
| 临近前沿 | {{nearby_frontier}} |

**描述**: {{description}}

**建议切入路径**: {{suggested_approach}}

{{/gaps}}

---

## 4. 方法附录

### 4.1 分析流程

1. **文献摄入**: 从 BibTeX/PDF/API 提取论文元数据
2. **数据清洗**: DOI/标题去重 + 关键词标准化（词干化 + 同义词合并）
3. **文献计量**: 关键词频率 + 共现矩阵 + 逐年趋势
4. **前沿检测**: Kleinberg 突发检测 + Louvain 社区发现 + 四维评分
5. **空白识别**: 密度分析 + 桥接检测 + 时间空洞分析

### 4.2 关键参数

| 参数 | 值 |
|------|----|
| 突发检测 gamma | 1.0 |
| 社区检测算法 | Louvain (greedy fallback) |
| 前沿评分权重 | Burst 0.35 + Recency 0.30 + Growth 0.25 + Cross 0.10 |
| 空白评分权重 | Density 0.35 + Proximity 0.30 + Growth 0.20 + Feasibility 0.15 |

### 4.3 局限性

- 关键词分析依赖于论文中标注的关键词质量——若论文关键词标注不规范，会影响共现网络精度
- 突发检测对小样本（<20 篇）的关键词不可靠
- 前沿/空白评分是基于文献计量信号的启发式估计，**不能替代领域专家判断**
- 语料库的覆盖范围（来源数据库、检索策略）决定了分析的视野边界

---

*报告由 literature-mining skill 自动生成 | {{generated_at}}*
