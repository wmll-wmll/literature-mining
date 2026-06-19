---
name: literature-mining
description: 文献挖掘——批量文献摄入 → 文献计量分析 → 研究前沿检测 → 研究空白识别 → 可视化报告。博士生开题、基金申请、综述论文、课题组方向调研。触发词："文献分析"/"研究前沿"/"找研究空白"/"literature mining"/"文献计量"/"前沿分析"。
argument-hint: "[— input: <bibtex|pdf-folder|search|paste>] [— query: <search terms>] [— path: <file or folder>] [— years: <start-end>]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, WebFetch, WebSearch
---

# /literature-mining — 文献挖掘与分析

批量分析文献，自动识别研究前沿与研究空白，生成可视化报告。

## Overview

```
[用户：我有 200 篇纳米递送的论文 BibTeX，帮我分析前沿和空白]
  │
  ▼
Phase 0: 环境检查（Python + 依赖）
  │
  ▼
Phase 1: 文献摄入（BibTeX / PDF / 搜索 / 粘贴）
  │
  ▼
Phase 2: 数据清洗（去重 + 标准化 + 关键词规范化）
  │
  ▼
Phase 3: 文献计量分析（关键词趋势 + 共现网络 + 引用分析）
  │
  ▼
Phase 4: 研究前沿检测（突发检测 + 社区发现 + 前沿评分）
  │
  ▼
Phase 5: 研究空白识别（密度分析 + 跨域桥接 + 空白评分）
  │
  ▼
Phase 6: 输出结构化上下文 + 可选可视化 HTML
```

## Constants

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_N_FRONTIERS` | 10 | 输出的前沿主题数 |
| `TOP_N_GAPS` | 10 | 输出的空白领域数 |
| `BURST_WINDOW` | 3 | 突发检测窗口（年） |
| `MIN_CLUSTER_SIZE` | 自适应 | 最小主题簇论文数（根据语料规模动态调整） |
| `SIMILARITY_THRESHOLD` | 0.85 | 标题去重相似度阈值 |
| `DEFAULT_YEAR_RANGE` | 2020-2026 | 默认年份范围 |
| `MAX_SEARCH_RESULTS` | 200 | 单次 API 搜索最大结果数 |
| `ENHANCED_KW` | 关闭 | `--enhanced-kw` 启用 KeyBERT 关键词提取 |
| `EMBEDDING_DEDUP` | 关闭 | `--embedding-dedup` 启用语义去重 |
| `LOUVAIN_RESOLUTION` | 1.0 | `--louvain-resolution` 调整社区粒度 |
| `BURST_GAMMA` | 1.0 | `--burst-gamma` 调整突发检测灵敏度 |

所有可配置参数统一在 `scripts/utils.py` 顶部的 `DEFAULTS` dict 中定义。各脚本导入 `from utils import DEFAULTS` 后引用。

## Inputs

| 输入方式 | CLI | 说明 |
|----------|-----|------|
| BibTeX 文件 | `python run_pipeline.py --mode bibtex --input <file>` | 最稳定 |
| CSV 文件 | `python run_pipeline.py --mode csv --input <file>` | DOI/title/abstract 列，Crossref 补全 |
| 搜索词 | `python run_pipeline.py --mode search --query "<terms>" [--lang zh]` | 多 API fallback：OpenAlex → Crossref → Semantic Scholar → PubMed |
| 语料库扩展 | `python run_pipeline.py --mode expand --input seed.jsonl` | 从种子论文 citation chasing 扩展 |
| 增量更新 | `python run_pipeline.py --mode incremental --input new.csv` | 追加论文 + 重跑 Phase 2-6 |
| 断点续跑 | `python run_pipeline.py --resume-from 4 ...` | 从指定 Phase 继续 |

### Unified Runner (推荐)

```bash
# 一键全流程
python scripts/run_pipeline.py --input papers.csv --mode csv

# 仅诊断
python scripts/run_pipeline.py --input papers.csv --mode csv --check-only

# 中文搜索
python scripts/run_pipeline.py --mode search --query "人工生命" --lang zh

# 断点续跑（Phase 3 完成后 Phase 4 挂了）
python scripts/run_pipeline.py --mode csv --input papers.csv --resume-from 4
```

### Phase Scripts (单独使用)

```bash
python ingest.py --mode bibtex --input file.bib --output outputs/raw_corpus.jsonl
python clean.py --input outputs/raw_corpus.jsonl --output outputs/clean_corpus.csv [--years 2016-2026]
python analyze.py --input outputs/clean_corpus.csv --output-dir outputs/
python frontier.py --input outputs/clean_corpus.csv --cooccurrence outputs/cooccurrence.csv --trends outputs/trends.json --output outputs/frontiers.json
python gaps.py --input outputs/clean_corpus.csv --cooccurrence outputs/cooccurrence.csv --frontiers outputs/frontiers.json --trends outputs/trends.json --output outputs/gaps.json
python report.py --corpus outputs/clean_corpus.csv --frontiers outputs/frontiers.json --gaps outputs/gaps.json --trends outputs/trends.json --cooccurrence outputs/cooccurrence.csv --output-dir outputs/report/
```

## Workflow

### Phase 0: 环境检查

检查 Python 3.9+ 和依赖包。缺则自动安装。

```bash
python -c "import pandas, numpy, networkx, sklearn, nltk, requests; print('OK')" 2>&1 || \
pip install pandas numpy networkx scikit-learn nltk requests pypdf bibtexparser community-detection
```

输出：
```
✅ Python 3.11.4
✅ 核心依赖就绪（pandas, numpy, networkx, sklearn, nltk, requests, pypdf, bibtexparser）
⚠️  community-detection 未安装，Phase 4 社区发现将回退到 greedy_modularity
```

### Phase 1: 文献摄入

根据 `--input` 标志选择摄入方法。

#### 1a: BibTeX 摄入（推荐）

```bash
python scripts/ingest.py --mode bibtex --input "<path>" --output outputs/raw_corpus.jsonl
```

每个 BibTeX entry 抽取：
- `id`, `title`, `authors`, `year`, `journal`, `volume`, `pages`, `doi`
- `abstract`（如果有）
- `keywords`（如果有 `keywords` 字段）

#### 1b: PDF 文件夹摄入

```bash
python scripts/ingest.py --mode pdf --input "<dir>" --output outputs/raw_corpus.jsonl
```

从每个 PDF 提取：
- DOI（从 PDF metadata 或第一页文本用正则匹配）
- 标题（从 metadata 或首段大字文本）
- 摘要（元数据或首段密集文本）
- 作者/年份（元数据或 Crossref API 补全）

**注意**：PDF 提取精度取决于 PDF 质量。若某 PDF 无法提取关键字段，在 `raw_corpus.jsonl` 中标记 `extraction_quality: "low"`，Phase 2 会尝试 API 补全。

#### 1c: 搜索词摄入（多 API）

```bash
python scripts/ingest.py --mode search --query "<terms>" --max 200 --output outputs/raw_corpus.jsonl [--lang zh] [--years 2020-2026]
```

API 优先级：**OpenAlex → Crossref → Semantic Scholar**。
- OpenAlex 优先（全文本索引，摘要覆盖率 76%，返回 concepts 关键词）
- Crossref 次选（元数据完整，`--years` 推送至 API 过滤层）
- Semantic Scholar 兜底（元数据质量最佳，但公开 API 限流极严）
- PubMed 最后（Entrez API，生物医学领域全覆盖）
- `--lang zh` 中文查询自动翻译后搜索

#### 1d: RIS / CSV

RIS：逐字段解析 `TY/ER` 块。
CSV：期望列 `doi`, `title`, `authors`, `year`, `journal`, `abstract`, `keywords`。有 DOI 的条目调 Crossref API 补全缺失字段；无 DOI 的保留 CSV 原始数据。`--years` 推送至 Crossref API filter 层（`from-pub-date` / `until-pub-date`），不浪费配额拉旧论文。

#### 摄入统计输出

```
[INGEST] Ingestion complete
  - 来源：CSV (alife_corpus.csv)
  - 总条目：18
  - 含摘要：18 (100%)
  - 含关键词：18 (100%)
  - 含 DOI：18 (100%)
  - 年份跨度：2024-2025
```

### Phase 2: 数据清洗

```bash
python scripts/clean.py --input outputs/raw_corpus.jsonl --output outputs/clean_corpus.csv
```

执行：
1. **DOI 去重**：相同 DOI → 保留摘要最完整的一条
2. **标题去重**：标题相似度 > `SIMILARITY_THRESHOLD` → 标记疑似重复，保留较新/较完整的一条
3. **作者名规范化**：统一为 `Last, First` 格式
4. **关键词标准化**：
   - 小写化
   - 轻量词干化（仅去复数 s/es + 长词尾 ing，保留可读性）
   - Lemma 字典补齐不规则形式（analyses→analysis, squares→square 等 30+ 组）
   - 同义词合并（基于 `templates/synonyms.json`，预置纳米医学 + ALife 两领域）
   - 噪声过滤（剥离 XML 标签/DOI/URL + 180+ 学术停用词黑名单）
5. **增强关键词提取**（可选）：`--enhanced-kw` 启用 KeyBERT（pip install keybert）替代 TF-IDF，使用 MMR diversity 提升关键词质量
6. **Embedding 去重**（可选）：`--embedding-dedup` 启用 sentence-transformers（all-MiniLM-L6-v2）对模糊匹配的 pair 做语义相似度判断，>0.95 自动合并
7. **字段补全**：有 DOI 但缺摘要/年份的 → 调 Crossref API 补全
8. **年份过滤**：过滤掉 `--years` 范围外的论文

输出统计：
```
🧹 清洗完成
  - 去重移除：12 篇（DOI 重复 8，标题相似 4）
  - 字段补全：18 篇通过 Crossref API 补全
  - 关键词标准化：合并 34 组同义词（如 "nanoparticle" + "nano-particle" + "NPs"）
  - 最终语料库：175 篇论文
  - 时间范围：2015-2026
```

### Phase 3: 文献计量分析

```bash
python scripts/analyze.py --input outputs/clean_corpus.csv --output-dir outputs/
```

生成以下分析产物：

#### 3a: 关键词频率分析 → `outputs/keyword_freq.csv`

| keyword | total_count | 2018 | 2019 | ... | 2026 | trend | cagr |
|---------|-------------|------|------|-----|------|-------|------|
| nanoparticle | 89 | 12 | 14 | ... | 8 | stable | +2.1% |
| mRNA delivery | 34 | 0 | 2 | ... | 15 | rising | +67% |

#### 3b: 关键词共现矩阵 → `outputs/cooccurrence.csv`

基于同一篇论文中关键词对出现的频次。

#### 3c: 逐年趋势 → `outputs/trends.json`

```json
{
  "keyword": "mRNA delivery",
  "yearly": [0, 2, 3, 5, 8, 12, 15, 16, 18],
  "cagr_3yr": 0.67,
  "burst_signal": true
}
```

#### 3d: 摘要统计 → `outputs/stats_summary.json`

```json
{
  "total_papers": 175,
  "year_range": [2015, 2026],
  "top_keywords_20": [...],
  "avg_authors_per_paper": 5.2,
  "journal_distribution": {...},
  "citation_stats": {"mean": 23.4, "median": 8, "max": 342}
}
```

Phase 3 完成后，向用户展示摘要统计（top keywords、年份分布），确认数据质量后再进入 Phase 4。

### Phase 4: 研究前沿检测

```bash
python scripts/frontier.py --cooccurrence outputs/cooccurrence.csv \
  --trends outputs/trends.json \
  --output outputs/frontiers.json \
  [--input outputs/clean_corpus.csv]
```

算法流程：

#### 4a: 突发检测

对每个关键词的逐年频次序列，使用 Kleinberg (2002) burst detection：
- 识别关键词在哪个时间段出现了统计学显著的频率爆发
- 重点关注**近 3 年内处于爆发期**的关键词
- 输出每个 burst keyword 的：爆发强度、爆发区间、当前状态

#### 4b: 主题聚类

在关键词共现网络上运行 Louvain 社区发现：
- 每个社区 = 一个研究主题
- 社区标签来源于其 top-5 关键词
- 计算每个主题的：平均发表年份、论文数、引文影响力、增长速度

#### 4c: 前沿评分

```
FrontierScore = 0.35 × BurstIntensity_norm
              + 0.30 × Recency_norm          (avg_year)
              + 0.25 × GrowthRate_norm       (3yr CAGR)
              + 0.10 × Interdisciplinarity   (cross-cluster edges)
```

输出 `outputs/frontiers.json`：

```json
{
  "frontiers": [
    {
      "rank": 1,
      "label": "mRNA-LNP 肿瘤免疫递送",
      "top_keywords": ["mRNA", "lipid nanoparticle", "tumor microenvironment", "immunotherapy"],
      "burst_keywords": ["mRNA delivery", "STING pathway", "ionizable lipid"],
      "paper_count": 23,
      "avg_year": 2024.1,
      "growth_rate_3yr": 0.45,
      "frontier_score": 0.91,
      "description": "mRNA 疫苗技术向肿瘤免疫的快速延伸，...",
      "representative_papers": ["doi:...", "doi:..."]
    }
  ],
  "burst_keywords": [...],
  "topology": {
    "n_clusters": 8,
    "modularity": 0.62
  }
}
```

向用户展示 Top-N 前沿表格，确认后进入 Phase 5。

### Phase 5: 研究空白识别

```bash
python scripts/gaps.py --cooccurrence outputs/cooccurrence.csv \
  --frontiers outputs/frontiers.json \
  --trends outputs/trends.json \
  --output outputs/gaps.json \
  [--input outputs/clean_corpus.csv]
```

三种空白类型：

#### 5a: 密度空白（Density Gap）
- 在共现网络中找到**论文数少但与高密度前沿簇高度相关**的关键词/子簇
- 信号：某技术在 A 领域很热，B 领域相关 but 几乎没有人发 → 空白

#### 5b: 桥接空白（Bridge Gap）
- 找到**两个不连通但语义相近的活跃主题簇**
- 语义相近判定：关键词向量余弦相似度 > 0.6
- 信号：两个领域各自发展，中间存在交叉研究机会

#### 5c: 时间空白（Temporal Gap）
- **早期有少量论文 but 近年断崖下降**的关键词 + 外部相关性上升
- 信号：因技术限制被搁置，新技术出现后可以复兴的领域

#### 5d: 语义空白（Semantic Gap）
- 在共现网络中计算 frontier 集群之间的最短路径距离
- 距离 ≤4 步且无直接边 → 语义接近但孤立 → 跨领域机会
- 作为已有三种空白类型的补充，独立评分后合并到结果中

#### 空白评分

```
GapScore = 0.30 × (1 - Density_norm)              # 真·空白
         + 0.25 × FrontierProximity_norm            # 离前沿近
         + 0.20 × BridgePotential_norm              # 桥接潜力
         + 0.15 × MethodologyReadiness_norm          # 方法可行性
         + 0.10 × ExternalInterest_norm              # 外部关注度上升
```

输出 `outputs/gaps.json`：

```json
{
  "gaps": [
    {
      "rank": 1,
      "label": "AI 驱动的血脑屏障递送设计",
      "gap_type": "bridge",
      "bridging_clusters": ["BBB delivery", "AI/ML drug design"],
      "description": "两端均在快速增长，但交叉区域仅 2 篇论文。ML 辅助穿越 BBB 的分子设计逻辑是明确的空白。",
      "gap_score": 0.87,
      "opportunity": "高——具备方法基础（两端成熟），可快速切入",
      "suggested_approach": "迁移 AI 分子设计方法到 BBB 穿越肽的设计空间"
    }
  ],
  "gap_summary": {
    "density_gaps": 3,
    "bridge_gaps": 4,
    "temporal_gaps": 3,
    "total": 10
  }
}
```

### Phase 6: 报告生成

```bash
python scripts/report.py --corpus outputs/clean_corpus.csv \
  --frontiers outputs/frontiers.json \
  --gaps outputs/gaps.json \
  --trends outputs/trends.json \
  --cooccurrence outputs/cooccurrence.csv \
  --output-dir outputs/report/
```

Phase 6 输出：

```
outputs/<name>/report/
├── summary.txt             # 精炼文本摘要（AI 直接读取呈现）
├── report_context.json     # 完整结构化数据（供二次分析/深入）
├── report.md               # Markdown 长文报告
├── vis_data.json           # 网络节点/边数据
├── trend_data.json         # 趋势序列
└── frontier_heatmap.json   # 前沿-空白矩阵
```

## Interactive Workflow（对话交互流程）

实际执行时，AI 执行以下交互节奏：

1. **确认输入** → "你准备怎么提供文献？"
   - A) 我有一个 `.bib` 文件
   - B) 我有一堆 PDF
   - C) 帮我搜索一个关键词
   - D) 我直接粘贴列表

2. **摄入 + 展示摘要** → "共摄入 187 篇，含摘要 142 篇。以下是 top-20 关键词：... 数据看起来对吗？"

3. **Phase 3-5 自动运行** → 无需交互

4. **展示前沿** → 展示 Top-N 前沿表格。**每条必须标注置信度**（见下方置信度规则）。数据不足时明确告知。

5. **展示空白** → 展示 Top-N 空白表格。**每条标注置信度**。若空白来自定性推断（quantitative gap detection found nothing），必须标注 `[LOW CONFIDENCE — qualitative inference]`。

6. **展示结论** → AI 读取 `report/summary.txt`，先报**整体置信度**（基于数据质量 verdict），再呈现具体发现。然后给出三个选项：

   > [1] 深入某个 gap/frontier  
   > [2] 扩展语料库重跑（搜索更多论文或 citation chasing）  
   > [3] 导出数据（`report_context.json` 可二次分析）

   用户选 1 → AI 从 `report_context.json` 提取该 gap/frontier 的详情。
   用户选 2 → 回到步骤 0，增量或全量重跑。
   用户选 3 → 告知文件路径。

### 置信度规则

**整体置信度**（基于数据质量 verdict）：

| Verdict | 置信度 | 说明 |
|---------|--------|------|
| good（≥50 papers, ≥5yr span, ≥70% abstracts） | **HIGH** | 结果可引用 |
| fair（1-2 项不达标） | **MEDIUM** | 方向正确，具体排名可能波动 |
| poor（3+ 项不达标） | **LOW** | 仅供探索参考，不建议作为决策依据 |

**单条置信度**（前沿/空白）：

| 条件 | 置信度 |
|------|--------|
| 定量检测产出 + 评分 ≥ 0.5 | **HIGH** |
| 定量检测产出 + 评分 < 0.5 | **MEDIUM** |
| 定性推断产出（qualitative inference） | **LOW** |
| 基础数据 < 10 篇论文支撑 | **LOW** |

**展示格式示例：**

```
[整体置信度: MEDIUM — 50 papers, 6yr span, 76% abstracts, 1 warning]

前沿:
  #1 ML/SVM ↔ chemometric authentication  [HIGH — score 0.894, 定量]
  #2 olive oil ↔ raman spectroscopy       [MEDIUM — score 0.677, 定量]
  ...

空白:
  #1 Bridge: ML vs chemometric            [HIGH — score 0.894, 定量 bridge]
  #2 Broker: genetic                      [LOW — qualitative inference]
  ...
```

## Key Rules

1. **渐进式输出**：每完成一个 Phase 就向用户报告摘要
2. **数据质量诚实**：PDF 提取失败、API 不可用等在输出中明确标注
3. **可复现**：所有中间产物存为 JSON/CSV
4. **评分是信号不是真理**：frontier/gap 分数是文献计量启发式估计，需领域专家判断
5. **去重优先**：Phase 2 宁可多删 (false positive) 也不少留 (false negative)
6. **离线优先**：优先本地文献（BibTeX/CSV），API 作为补充
7. **哈希缓存**：Phase 输入文件做 SHA256，未变则跳过
8. **自适应阈值**：min_cluster_size、edge_ratio 等根据语料规模动态调整
9. **质量 verdict**：Phase 1 输出数据质量报告（good / fair / poor）

## Refusals

- "跳过清洗直接分析" → 拒绝。未清洗数据让前沿/空白检测失效
- "给我 100 个前沿方向" → 拒绝。超过 20 无意义

## Integration

- `scripts/run_pipeline.py` 一键串联 6 个 Phase
- 输出 JSON/CSV 可在 Python/R 中二次分析
- 与 `/deep-research` 互补：deep-research 面向通用信息深度调研，literature-mining 面向学术文献结构化分析

## Python Dependencies

完整依赖见 `scripts/requirements.txt`。Phase 0 自动检测安装。

核心依赖：`pandas`, `numpy`, `networkx`, `scikit-learn`, `nltk`, `scipy`, `requests`, `pypdf`
