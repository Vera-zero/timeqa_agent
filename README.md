# TimeQA Agent

时间事件抽取与时间线分析系统。从文本中抽取时间事件，进行实体消歧，构建时间线，并存储为知识图谱。

---

## 📋 目录

- [安装](#安装)
- [项目结构](#项目结构)
- [流水线阶段](#流水线阶段)
- [命令行使用](#命令行使用)
- [中间文件](#中间文件)
- [Python API](#python-api)
- [检索器升级 (2026-02-07)](#检索器升级-2026-02-07)
- [配置](#配置)
- [更新日志](#更新日志)
- [环境变量](#环境变量)

---

## 安装

```bash
cd timeqa_agent
pip install -e .
```

依赖：
- requests
- numpy
- networkx

### 嵌入模型

本项目支持多种嵌入模型进行语义检索：

| 模型 | 类型 | 推荐场景 | 模型大小 |
|------|------|----------|----------|
| **Contriever-MSMARCO** | 无监督密集检索 | 通用检索（推荐） | ~438 MB |
| **DPR** | 双编码器架构 | 问答系统 | ~876 MB |
| **BGE-M3** | 多语言模型 | 多语言/已有模型 | ~2.3 GB |

下载模型后，将模型文件放置于 `models/` 目录下，并在配置文件中指定模型路径。

**快速下载 Contriever**：
```bash
python download_contriever.py
```

## 项目结构

```
timeqa_agent/
├── timeqa_agent/
│   ├── config.py              # 配置模块
│   ├── chunker.py             # 文档分块器
│   ├── event_extractor.py     # 时间事件抽取器
│   ├── event_validator.py     # 事件检查器（时间格式校验）
│   ├── event_filter.py        # 事件过滤器（去除 chunk 重叠产生的重复事件）
│   ├── entity_disambiguator.py # 实体消歧器
│   ├── timeline_extractor.py  # 时间线抽取器
│   ├── embeddings.py          # 嵌入模型（Contriever/DPR/BGE-M3）
│   ├── graph_store.py         # 知识图谱存储
│   ├── graph_store_cli.py     # 图存储命令行工具
│   ├── retriever_cli.py       # 检索器命令行工具
│   ├── query_parser.py        # 查询解析器
│   ├── query_parser_cli.py    # 查询解析器命令行工具
│   ├── pipeline.py            # 抽取流水线
│   └── retrievers/
│       ├── base.py                    # 检索器基类和数据结构
│       ├── keyword_retriever.py       # 关键词检索器（BM25/TF-IDF）
│       ├── semantic_retriever.py      # 语义检索器（Contriever/DPR/BGE-M3）
│       ├── hybrid_retriever.py        # 混合检索器
│       ├── voting_retriever.py        # 多层投票检索器
│       └── hierarchical_retriever.py  # 三层递进检索器
├── configs/
│   └── timeqa_config.json     # 默认配置文件
├── config_examples/           # 配置示例
│   ├── contriever_bm25_config.json
│   ├── dpr_bm25_config.json
│   └── bge_m3_tfidf_config.json
├── data/timeqa/
│   ├── corpus/                # 语料库 (test/train/validation.json)
│   └── raw/                   # 原始数据
├── test_retrievers.py         # 检索器测试脚本
├── usage_examples.py          # 使用示例
└── pyproject.toml
```

## 流水线阶段

| 阶段 | 名称 | 输入 | 输出 |
|------|------|------|------|
| 1 | chunk | 语料库文档 | 文档分块 |
| 2 | event | 分块 | 时间事件 |
| 3 | event_validate | 事件 | 时间格式校验后的事件 |
| 4 | event_filter | 事件 | 过滤后的事件（去重 + 保留最细粒度） |
| 5 | disambiguate | 过滤后的事件 | 实体聚类 |
| 6 | timeline | 事件+实体 | 时间线 |
| 7 | graph | 全部 | 知识图谱 |

## 命令行使用

### 流水线

```bash
# 单文档测试（验证链路）
python -m timeqa_agent.pipeline --split test --mode single --doc-index 0

# 全量处理 test 集
python -m timeqa_agent.pipeline --split test --mode full

# 处理 train 集
python -m timeqa_agent.pipeline --split train --mode full

# 使用配置文件
python -m timeqa_agent.pipeline --split test --config configs/timeqa_config.json
```

### 指定阶段

```bash
# 从事件抽取阶段开始（使用已有的分块结果）
python -m timeqa_agent.pipeline --split test --start event

# 只执行分块和事件抽取
python -m timeqa_agent.pipeline --split test --start chunk --end event

# 从事件检查阶段开始（使用已有的事件抽取结果）
python -m timeqa_agent.pipeline --split test --start event_validate

# 从事件过滤阶段开始（使用已有的事件检查结果）
python -m timeqa_agent.pipeline --split test --start event_filter

# 只执行时间线抽取
python -m timeqa_agent.pipeline --split test --start timeline --end timeline
```

### 流水线参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| --split | -s | 数据集: test/train/validation | test |
| --mode | -m | 模式: single(单文档)/full(全量) | single |
| --doc-index | -d | 单文档模式的文档索引 | 0 |
| --start | | 起始阶段 | chunk |
| --end | | 结束阶段 | graph |
| --data-dir | | 数据目录 | data/timeqa |
| --config | | 配置文件路径 | None |

### 图存储查询

```bash
# 交互式模式
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json

# 查看统计信息
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json stats

# 列出所有实体
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entities

# 列出所有时间线
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json timelines

# 查询实体详情
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity "John Smith"

# 查询实体参与的事件
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity-events "John Smith"

# 查询实体的时间线
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json entity-timelines "John Smith"

# 查询事件详情
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json event "evt_001"

# 查询时间线详情（含事件列表）
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json timeline "tl_001"

# 模糊搜索实体
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json search "Smith"

# 查询时间范围内的事件
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json time 1990 2000

# JSON 格式输出
python -m timeqa_agent.graph_store_cli -g data/timeqa/graph/test.json --json entity "John"
```

### 图存储交互式命令

进入交互式模式后可用的命令：

| 命令 | 说明 |
|------|------|
| stats | 显示图统计信息 |
| entities | 列出所有实体 |
| timelines | 列出所有时间线 |
| entity \<name\> | 查询实体详情 |
| entity-events \<name\> | 查询实体参与的事件 |
| entity-timelines \<name\> | 查询实体的时间线 |
| event \<id\> | 查询事件详情 |
| timeline \<id\> | 查询时间线详情及其事件 |
| search \<query\> | 模糊搜索实体 |
| time \<start\> \<end\> | 查询时间范围内的事件 |
| help | 显示帮助 |
| quit/exit | 退出 |

### 检索器查询

**模型配置说明**：

检索器命令行工具会自动根据配置文件（`configs/timeqa_config.json`）中的设置选择嵌入模型和检索方法：

- **语义检索模型**：由 `retriever.semantic_model_type` 指定（contriever/dpr/bge-m3）
- **关键词检索算法**：由 `retriever.keyword_algorithm` 指定（bm25/tfidf）
- **模型路径**：由 `retriever.semantic_model_name` 等参数指定
- **设备选择**：由 `retriever.semantic_model_device` 指定（cpu/cuda）

使用自定义配置文件：
```bash
# 使用指定的配置文件运行检索器
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json -c configs/my_config.json

# 使用 Contriever + BM25 配置
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json -c config_examples/contriever_bm25_config.json

# 使用 DPR + BM25 配置
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json -c config_examples/dpr_bm25_config.json
```

**基本用法**：

```bash
# 交互式模式
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json

# 混合检索（关键词 + 语义）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "when did John join the company"

# 关键词检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json keyword "John Smith"

# 语义检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json semantic "career changes"

# 三层递进检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json hierarchical "when did John join"

# 三层递进检索（指定 k1/k2/k3）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json hierarchical "career" --k1 3 --k2 5 --k3 10

# 三层递进检索（含中间层详情）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json hierarchical-details "career history"

# 获取检索结果对应的chunks（事件溯源到原始文本）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json get-chunks "John career" --chunks data/timeqa/chunk/test.json

# 获取指定chunk的前后上下文
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding "doc-00000-chunk-0005" --chunks data/timeqa/chunk/test.json --before 2 --after 2

# 检索事件并获取其chunk的前后上下文
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding-event "John career" --chunks data/timeqa/chunk/test.json --before 1 --after 1

# 只检索事件
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "2020" -t event

# 只检索实体
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "Smith" -t entity

# 只检索时间线
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "career" -t timeline

# 带实体上下文的检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json entity-search "John Smith" "when did he graduate"

# 设置返回数量
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "event" -k 20

# 使用特定融合模式
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "career" --fusion weighted_sum

# JSON 格式输出
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "John" --json

# 禁用语义检索（仅使用关键词）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "John" --no-semantic

# 详细输出
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "John" -v
```

### 事件溯源到Chunk功能

检索器提供了从事件溯源到原始文本chunk的功能，帮助您找到事件的上下文来源。

#### 1. 获取事件对应的chunks

**命令行使用**：

```bash
# 检索事件并获取对应的chunks
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json get-chunks "John career" --chunks data/timeqa/chunk/test.json

# JSON格式输出
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json get-chunks "doc-00000-chunk-0007-event-0018" --chunks data/timeqa/chunk/test.json --json

# 详细输出（显示完整chunk内容）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json get-chunks "John" --chunks data/timeqa/chunk/test.json -v
```

**交互式模式**：

```bash
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json

> chunks-path data/timeqa/chunk/test.json
chunks路径设置为: data/timeqa/chunk/test.json

> get-chunks John career
查询 'John career' 找到 3 个chunk:

--- Chunk 1 ---
  ID: doc-00000-chunk-0001
  文档: John Smith Biography (doc-00000)
  内容: John Smith was born in 1980. He graduated from MIT in 2002...

--- Chunk 2 ---
  ID: doc-00000-chunk-0002
  文档: John Smith Biography (doc-00000)
  内容: After graduation, John joined Google in 2003...
```

#### 2. 获取chunk的前后上下文

**命令行使用**：

```bash
# 获取指定chunk的前后各2个chunk
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding "doc-00000-chunk-0005" --chunks data/timeqa/chunk/test.json --before 2 --after 2

# 检索事件并获取其chunk的前后上下文
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding-event "John career" --chunks data/timeqa/chunk/test.json --before 1 --after 1

# JSON格式输出
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding "doc-00000-chunk-0005" --chunks data/timeqa/chunk/test.json --json

# 详细输出（显示完整内容）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json surrounding-event "John" --chunks data/timeqa/chunk/test.json -v
```

**交互式模式**：

```bash
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json

> chunks-path data/timeqa/chunk/test.json
chunks路径设置为: data/timeqa/chunk/test.json

> before 2
前面chunk数量设置为: 2

> after 2
后面chunk数量设置为: 2

> surrounding doc-00000-chunk-0005
Chunk 'doc-00000-chunk-0005' 的前后上下文:
文档: doc-00000, 当前索引: 5
共 5 个chunks

前面 2 个chunks:
  [doc-00000-chunk-0003]
  In 1998, John started his undergraduate studies at MIT...

  [doc-00000-chunk-0004]
  During his time at MIT, he focused on computer science...

当前chunk:
  [doc-00000-chunk-0005]
  After graduation in 2002, John joined Google as a software engineer...

后面 2 个chunks:
  [doc-00000-chunk-0006]
  At Google, he worked on search infrastructure...

  [doc-00000-chunk-0007]
  In 2010, John was promoted to senior engineer...

> surrounding-event John career
查询 'John career' 找到 2 个事件的上下文:

============================================================
事件 1: John joined Google as a software engineer
事件ID: evt-00042
Chunk索引: 5 (共 5 个chunks)
============================================================

前面 2 个chunks:
  - In 1998, John started his undergraduate studies...
  - During his time at MIT, he focused on...

当前chunk (包含此事件):
  ★ After graduation in 2002, John joined Google as a software engineer...

后面 2 个chunks:
  - At Google, he worked on search infrastructure...
  - In 2010, John was promoted to senior engineer...
```

**功能说明**：

1. **get-chunks**: 检索事件并获取对应的chunks
   - 自动去重：多个事件可能来自同一个chunk，默认会自动去重
   - 支持多种存储格式：JSON文件（列表或字典格式）、内存字典

2. **surrounding**: 获取指定chunk的前后上下文
   - 按文档顺序获取前后chunk
   - 自动处理边界情况（文档开头/结尾）
   - 如果中间chunk缺失会停止查找

3. **surrounding-event**: 检索事件并获取其chunk的前后上下文
   - 结合事件检索和上下文获取
   - 显示事件所在chunk及其前后文
   - 便于理解事件的完整背景

**支持的参数**：

- `--before N`: 获取前面N个chunk（默认1）
- `--after N`: 获取后面N个chunk（默认1）
- `--chunks`: chunks数据文件路径
- `-v/--verbose`: 显示完整chunk内容
- `--json`: JSON格式输出

**使用场景**：

- **验证事件抽取的准确性**：查看原始文本上下文
- **理解事件的完整背景**：获取周围的相关信息
- **时间线分析**：查看事件前后发生了什么
- **调试和分析**：追溯事件的来源文档和位置
- **上下文理解**：获取chunk的前后文以理解完整语境



### 检索器命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| --graph | -g | 图文件路径（必需） | - |
| --config | -c | 配置文件路径 | None |
| --type | -t | 目标类型: all/entity/event/timeline | all |
| --top-k | -k | 返回结果数量 | 10 |
| --fusion | | 融合模式: rrf/weighted_sum/max_score/interleave | rrf |
| --k1 | | 三层递进检索：第一层实体数量 | 配置值 |
| --k2 | | 三层递进检索：第三层时间线数量 | 配置值 |
| --k3 | | 三层递进检索：第三层事件数量 | 配置值 |
| --chunks | | chunks数据文件路径（用于get-chunks/surrounding命令） | None |
| --before | | 获取前面N个chunk（用于surrounding命令） | 1 |
| --after | | 获取后面N个chunk（用于surrounding命令） | 1 |
| --json | | JSON 格式输出 | false |
| --verbose | -v | 详细输出 | false |
| --no-semantic | | 禁用语义检索 | false |

### 检索器交互式命令

进入交互式模式后可用的命令：

| 命令 | 说明 |
|------|------|
| search \<query\> | 混合检索 |
| keyword \<query\> | 关键词检索 |
| semantic \<query\> | 语义检索 |
| hierarchical \<query\> | 三层递进检索 |
| hierarchical-details \<query\> | 三层递进检索（含中间层信息） |
| entity \<name\> \<query\> | 带实体上下文的检索 |
| get-chunks \<query\> | 检索事件并获取对应chunks（需先设置chunks_path） |
| surrounding \<chunk_id\> | 获取指定chunk的前后上下文 |
| surrounding-event \<query\> | 检索事件并获取chunk前后上下文 |
| chunks-path \<path\> | 设置chunks数据文件路径 |
| before \<n\> | 设置获取前面N个chunk（默认1） |
| after \<n\> | 设置获取后面N个chunk（默认1） |
| type \<entity\|event\|timeline\|all\> | 设置目标类型过滤 |
| topk \<n\> | 设置返回数量 |
| k1 \<n\> | 设置三层递进检索实体数量 |
| k2 \<n\> | 设置三层递进检索时间线数量 |
| k3 \<n\> | 设置三层递进检索事件数量 |
| fusion \<mode\> | 设置融合模式 |
| stats | 显示统计信息 |
| verbose | 切换详细输出 |
| help | 显示帮助 |
| quit/exit | 退出 |

### 查询解析器

查询解析器用于将用户问题分解为主干部分和时间约束部分，并生成针对实体、事件、时间线的检索语句。

```bash
# 交互式模式
python -m timeqa_agent.query_parser_cli

# 解析单个问题（完整流程：解析 + 生成检索语句）
python -m timeqa_agent.query_parser_cli parse "Which team did Attaphol Buspakom play for in 2007?"

# 仅解析问题（不生成检索语句）
python -m timeqa_agent.query_parser_cli parse-only "Where did John work during the Beijing Olympics?"

# 仅生成检索语句（直接输入问题主干）
python -m timeqa_agent.query_parser_cli retrieval "Which team did Attaphol Buspakom play for?"

# JSON 格式输出
python -m timeqa_agent.query_parser_cli parse "When did he graduate?" --json

# 使用指定配置文件
python -m timeqa_agent.query_parser_cli -c configs/timeqa_config.json parse "Who was president in 1990?"
```

### 查询解析器命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| --config | -c | 配置文件路径 | None |
| --json | | JSON 格式输出 | false |
| --verbose | -v | 详细输出 | false |

### 查询解析器输出示例

**文本格式输出：**

```
原始问题: Which team did Attaphol Buspakom play for in 2007?
问题主干: Which team did Attaphol Buspakom play for?

时间约束:
  类型: explicit
  原始表达式: in 2007
  标准化时间: 2007
  描述: The year 2007

检索语句:
  实体查询: Attaphol Buspakom, a Thai professional football player
  时间线查询: Attaphol Buspakom's football career, clubs and teams played for
  事件查询 (5 条):
    1. Attaphol Buspakom played for Buriram United F.C.
    2. Attaphol Buspakom played for Chonburi F.C.
    3. Attaphol Buspakom played for Thailand national football team
    4. Attaphol Buspakom joined a football club
    5. Attaphol Buspakom transferred to a new team
```

**JSON 格式输出：**

```json
{
  "parse_result": {
    "original_question": "Which team did Attaphol Buspakom play for in 2007?",
    "question_stem": "Which team did Attaphol Buspakom play for?",
    "time_constraint": {
      "constraint_type": "explicit",
      "original_expression": "in 2007",
      "normalized_time": "2007",
      "description": "The year 2007"
    },
    "event_type": "interval",
    "answer_type": "entity"
  },
  "retrieval_queries": {
    "entity_query": "Attaphol Buspakom, a Thai professional football player",
    "timeline_query": "Attaphol Buspakom's football career, clubs and teams played for",
    "event_queries": [
      "Attaphol Buspakom played for Buriram United F.C.",
      "Attaphol Buspakom played for Chonburi F.C.",
      "Attaphol Buspakom played for Thailand national football team",
      "Attaphol Buspakom joined a football club",
      "Attaphol Buspakom transferred to a new team"
    ]
  }
}
```

### 时间约束类型

| 类型 | 说明 | 示例 |
|------|------|------|
| explicit | 显式时间约束 | "in 2007", "from 1990 to 2000", "before 1980" |
| implicit | 隐式时间约束 | "during the Beijing Olympics", "when he was president" |
| none | 无时间约束 | "Who is the CEO of Apple?" |

## 中间文件

每个阶段的输出会保存到对应目录：

```
data/timeqa/
├── chunk/
│   ├── test.json           # 全量模式
│   └── test_doc0.json      # 单文档模式
├── event/
├── event_validate/
├── event_filter/
├── disambiguate/
├── timeline/
└── graph/
```

## Python API

```python
from timeqa_agent import (
    DocumentChunker,
    EventExtractor,
    EventValidator,
    EventFilter,
    EntityDisambiguator,
    TimelineExtractor,
    TimelineGraphStore,
    ExtractionPipeline,
    PipelineConfig,
    QueryParser,
)

# 使用流水线
config = PipelineConfig(
    split="test",
    mode="single",
    doc_index=0,
)
pipeline = ExtractionPipeline(config)
results = pipeline.run()

# 或单独使用各组件
chunker = DocumentChunker()
chunks = chunker.chunk_document(content, doc_id, doc_title, source_idx)

extractor = EventExtractor()
events = extractor.extract_from_chunk(chunk)

# 查询图存储
store = TimelineGraphStore()
store.load("data/timeqa/graph/test.json")

# 获取统计信息
stats = store.get_stats()

# 查询实体
entity = store.get_entity("John Smith")
events = store.get_entity_events("John Smith")
timelines = store.get_entity_timelines("John Smith")

# 查询时间线
timeline = store.get_timeline("tl_001")
events = store.get_timeline_events("tl_001")

# 搜索实体
results = store.get_entities_by_name("Smith", fuzzy=True)

# 时间范围查询
events = store.get_events_in_time_range("1990", "2000")

# 使用查询解析器
parser = QueryParser()

# 完整处理流程
output = parser.process("Which team did Attaphol Buspakom play for in 2007?")
print(output.parse_result.original_question)  # "Which team did Attaphol Buspakom play for in 2007?"
print(output.parse_result.question_stem)  # "Which team did Attaphol Buspakom play for?"
print(output.parse_result.time_constraint.constraint_type)  # "explicit"
print(output.retrieval_queries.entity_query)  # "Attaphol Buspakom, a Thai professional football player"
print(output.retrieval_queries.event_queries)  # ["Attaphol Buspakom played for...", ...]

# 分步调用
parse_result = parser.parse_question("Where did John work during the Olympics?")
print(parse_result.original_question)  # "Where did John work during the Olympics?"
print(parse_result.question_stem)  # "Where did John work?"
print(parse_result.time_constraint.constraint_type)  # "implicit"

queries = parser.generate_retrieval_queries("Where did John work?")
print(queries.entity_query)
print(queries.timeline_query)
print(queries.event_queries)

# 从检索结果溯源到原始chunks
from timeqa_agent.retrievers import HybridRetriever

retriever = HybridRetriever(graph_store, embed_fn=embed_fn)
events = retriever.retrieve("John career", target_type="event", top_k=5)

# 获取单个事件对应的chunk信息
chunk_info = retriever.get_chunk_info_by_event(events[0])
print(f"Chunk ID: {chunk_info['chunk_id']}")
print(f"Document: {chunk_info['doc_title']}")

# 获取完整的chunk数据（包括内容）
chunks_file = "data/timeqa/chunk/test.json"
chunk_data = retriever.get_chunk_by_event(events[0], chunks_file)
print(f"Chunk内容: {chunk_data['content']}")

# 批量获取多个事件对应的chunks（自动去重）
all_chunks = retriever.get_chunks_for_events(events, chunks_file, deduplicate=True)
print(f"检索到 {len(events)} 个事件，来自 {len(all_chunks)} 个不同的chunks")

# 获取chunk的前后上下文
chunk_id = "doc-00000-chunk-0005"
context = retriever.get_surrounding_chunks(chunk_id, chunks_file, before=2, after=2)
print(f"前面 {len(context['before'])} 个chunks, 后面 {len(context['after'])} 个chunks")
print(f"当前chunk内容: {context['current']['content'][:100]}")

# 从事件获取chunk及其前后上下文
event = events[0]
context = retriever.get_surrounding_chunks_by_event(event, chunks_file, before=2, after=2)
print(f"事件所在chunk: {context['current']['chunk_id']}")
print(f"前后文共 {context['total_chunks']} 个chunks")
for chunk in context['before']:
    print(f"  前: {chunk['content'][:50]}...")
print(f"  当前: {context['current']['content'][:50]}...")
for chunk in context['after']:
    print(f"  后: {chunk['content'][:50]}...")

# 也可以使用内存中的chunks字典
chunks_dict = {
    "chunk-001": {"chunk_id": "chunk-001", "content": "...", "doc_id": "doc-001"},
    "chunk-002": {"chunk_id": "chunk-002", "content": "...", "doc_id": "doc-001"},
}
chunk_data = retriever.get_chunk_by_event(events[0], chunks_dict)
```

---

## 检索器升级 (2026-02-07)

### 🎯 升级内容

本次升级对检索器系统进行了全面改造，支持更多先进的检索算法和嵌入模型。

### ✨ 新功能

#### 1. **语义检索器升级**

现在支持以下嵌入模型：

| 模型 | 类型 | 推荐场景 | 模型大小 |
|------|------|----------|----------|
| **Contriever** | 无监督密集检索 | 通用检索（推荐） | ~438 MB |
| **Contriever-MSMARCO** | 微调版 Contriever | 高性能检索（最推荐） | ~438 MB |
| **DPR** | 双编码器架构 | 问答系统 | ~876 MB |
| **BGE-M3** | 多语言模型 | 多语言/已有模型 | ~2.3 GB |

#### 2. **关键词检索器升级**

现在支持以下算法：

| 算法 | 特点 | 推荐场景 |
|------|------|----------|
| **BM25** | 概率排序函数 | 通用关键词检索（推荐） |
| **TF-IDF** | 经典算法 | 保持兼容旧版本 |

### 📦 安装新依赖

```bash
# 必需依赖
pip install transformers torch rank-bm25

# 可选依赖（用于 BM25 词干提取和停用词）
pip install nltk
```

### 🚀 快速开始

#### **方式 1：使用配置文件**

```python
from timeqa_agent.config import RetrieverConfig
from timeqa_agent.retrievers import HybridRetriever

# 创建配置（使用 Contriever + BM25）
config = RetrieverConfig(
    semantic_model_type="contriever",
    semantic_model_name="./models/contriever-msmarco",
    keyword_algorithm="bm25",
    fusion_mode="rrf"
)

# 创建检索器（自动加载模型）
retriever = HybridRetriever(graph_store, config)

# 执行检索
results = retriever.retrieve("查询内容", top_k=10)
```

#### **方式 2：手动创建嵌入函数**

```python
from timeqa_agent.embeddings import create_embed_fn
from timeqa_agent.retrievers import SemanticRetriever

# 创建 Contriever 嵌入函数
embed_fn = create_embed_fn(
    model_type="contriever",
    model_name="./models/contriever-msmarco",
    device="cpu"
)

# 创建语义检索器
retriever = SemanticRetriever(graph_store, config, embed_fn=embed_fn)
```

### 📝 配置示例

#### **示例 1：Contriever + BM25（推荐）**

```json
{
  "retriever": {
    "semantic_model_type": "contriever",
    "semantic_model_name": "./models/contriever-msmarco",
    "keyword_algorithm": "bm25",
    "fusion_mode": "rrf"
  }
}
```

#### **示例 2：DPR + BM25（高性能）**

```json
{
  "retriever": {
    "semantic_model_type": "dpr",
    "dpr_ctx_encoder": "./models/dpr/ctx-encoder",
    "dpr_question_encoder": "./models/dpr/question-encoder",
    "keyword_algorithm": "bm25",
    "fusion_mode": "weighted_sum"
  }
}
```

#### **示例 3：BGE-M3 + TF-IDF（兼容旧版）**

```json
{
  "retriever": {
    "semantic_model_type": "bge-m3",
    "semantic_model_name": "./models/bge-m3/bge-m3",
    "keyword_algorithm": "tfidf"
  }
}
```

### 🔧 API 参考

#### **RetrieverConfig 配置项**

```python
@dataclass
class RetrieverConfig:
    # 语义检索配置
    semantic_model_type: str = "contriever"  # "contriever", "dpr", "bge-m3"
    semantic_model_name: str = "facebook/contriever-msmarco"
    semantic_model_device: str = "cpu"       # "cpu", "cuda", "cuda:0"
    contriever_normalize: bool = True
    dpr_ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"
    dpr_question_encoder: str = "facebook/dpr-question_encoder-single-nq-base"
    bge_m3_model_path: str = "./models/bge-m3/bge-m3"

    # 关键词检索配置
    keyword_algorithm: str = "bm25"          # "bm25", "tfidf"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_use_stemming: bool = False
    bm25_remove_stopwords: bool = False

    # 混合检索配置
    fusion_mode: str = "rrf"                 # "rrf", "weighted_sum", "max_score", "interleave"
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    enable_keyword: bool = True
    enable_semantic: bool = True
```

#### **创建嵌入函数**

```python
from timeqa_agent.embeddings import create_embed_fn, create_dpr_embed_fn

# 方式 1：Contriever
embed_fn = create_embed_fn(
    model_type="contriever",
    model_name="./models/contriever-msmarco",
    device="cpu"
)

# 方式 2：DPR（返回两个编码器）
ctx_embed_fn, question_embed_fn = create_dpr_embed_fn(
    ctx_encoder_name="./models/dpr/ctx-encoder",
    question_encoder_name="./models/dpr/question-encoder"
)

# 方式 3：BGE-M3
embed_fn = create_embed_fn(
    model_type="bge-m3",
    model_name="./models/bge-m3/bge-m3"
)
```

### 🧪 测试

运行测试脚本验证安装：

```bash
cd d:\Verause\science\codes\timeqa_agent_copy
python test_retrievers.py
```

测试内容包括：
1. ✅ Contriever 嵌入功能
2. ✅ BM25 关键词检索
3. ✅ 检索器配置

### 🔄 向后兼容

旧版代码无需修改即可运行：

```python
# 旧版用法（仍然支持）
from timeqa_agent.embeddings import create_local_embed_fn

embed_fn = create_local_embed_fn("./models/bge-m3/bge-m3")
retriever = SemanticRetriever(graph_store, config, embed_fn=embed_fn)
```

新版推荐用法：

```python
# 新版用法（推荐）
config = RetrieverConfig(
    semantic_model_type="contriever",
    semantic_model_name="./models/contriever-msmarco"
)
retriever = SemanticRetriever(graph_store, config)  # 自动创建 embed_fn
```

### 📊 性能对比

| 配置 | 检索质量 | 速度 | 内存占用 |
|------|---------|------|---------|
| Contriever + BM25 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~1 GB |
| DPR + BM25 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~2 GB |
| BGE-M3 + TF-IDF | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~3 GB |

### ❓ 常见问题

#### **Q1: 如何下载模型？**

```bash
# 运行下载脚本
python download_contriever.py

# 或手动下载
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')
model.save_pretrained('./models/contriever-msmarco')
tokenizer.save_pretrained('./models/contriever-msmarco')
```

#### **Q2: 如何使用 GPU 加速？**

```python
config = RetrieverConfig(
    semantic_model_device="cuda"  # 或 "cuda:0"
)
```

#### **Q3: BM25 参数如何调优？**

- **k1** (1.2-2.0): 控制词频饱和度，越大词频影响越大
- **b** (0.0-1.0): 控制文档长度归一化，越大长度影响越大

推荐值：`k1=1.5, b=0.75`

#### **Q4: 如何选择融合模式？**

| 融合模式 | 特点 | 推荐场景 |
|---------|------|----------|
| `rrf` | 倒数排名融合 | 通用（推荐） |
| `weighted_sum` | 加权求和 | 需要调整权重 |
| `max_score` | 取最大分数 | 保守策略 |
| `interleave` | 交错合并 | 多样性优先 |

### 🛠️ 故障排除

#### **问题 1: 导入错误**

```
ImportError: No module named 'rank_bm25'
```

**解决方案**：
```bash
pip install rank-bm25
```

#### **问题 2: 模型加载失败**

```
OSError: Model not found
```

**解决方案**：
1. 检查模型路径是否正确
2. 确保已下载模型
3. 使用绝对路径或相对于工作目录的路径

#### **问题 3: GPU 内存不足**

```
RuntimeError: CUDA out of memory
```

**解决方案**：
```python
# 方案 1: 使用 CPU
config.semantic_model_device = "cpu"

# 方案 2: 减小批处理大小
config.embed_batch_size = 16  # 默认 32
```

### 📚 参考文献

- [Contriever Paper](https://arxiv.org/abs/2112.09118)
- [DPR Paper](https://arxiv.org/abs/2004.04906)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [MRAG Framework](https://arxiv.org/abs/2412.15540)

---

## 配置

配置文件位于 `configs/timeqa_config.json`，包含以下模块：

### 分块配置 (chunk)

```json
{
  "chunk": {
    "strategy": "fixed_size",      // 分块策略: fixed_size, sentence（仅支持这两种）
    "chunk_size": 1500,            // 目标分块大小（字符数）。建议 1000-2000
    "chunk_overlap": 100,          // 分块重叠大小。建议 chunk_size 的 5-10%
    "max_sentences": 10,           // 每块最大句子数（sentence 策略专用）
    "min_chunk_size": 500,         // 最小分块大小（fixed_size 和 sentence 策略都适用）
    "max_chunk_size": 2000,        // 最大分块大小
    "preserve_sentences": true     // 是否保持句子完整性（fixed_size 策略专用）
  }
}
```

**策略说明**：
- `fixed_size`：按固定字符数分块，使用 chunk_size、chunk_overlap、preserve_sentences、min_chunk_size
  - 当最后一个分块的大小小于 `min_chunk_size` 时，会自动合并到上一个分块
- `sentence`：按句子边界分块，使用 max_sentences、min_chunk_size、max_chunk_size

**建议**：
- 英文文档：chunk_size 1500-2000
- 中文文档：chunk_size 1000-1500（中文信息密度更高）
- 时间事件密集文档：适当减小 chunk_size，提高抽取精度

### 事件抽取配置 (extractor)

```json
{
  "extractor": {
    "model": "deepseek-v3.1-terminus",  // LLM 模型名称
    "base_url": "http://...",           // API 端点
    "temperature": 0.1,                 // 生成温度。建议 0.1-0.3，低温度保证抽取一致性
    "max_retries": 3,                   // 最大重试次数
    "timeout": 180,                     // 请求超时（秒）。复杂文档建议 180-300
    "batch_size": 1,                    // 文档级批处理大小（batch_size=1顺序处理，>1批处理模式）
    "enable_multi_round": true,         // 是否启用多轮抽取
    "max_rounds": 2,                    // 最大抽取轮数
    "review_temperature": 0.0,          // 审查轮次的温度参数
    "prior_events_context_mode": "none", // 前置事件上下文模式: none, full, sliding_window
    "prior_events_window_size": 3       // 滑动窗口大小（仅 sliding_window 模式有效）
  }
}
```

**批处理配置说明**：

`batch_size` 控制文档级并行处理能力。批处理策略确保同一文档的分块按顺序处理，同时允许不同文档的分块并行处理。

| batch_size | 处理模式 | 说明 |
|------------|----------|------|
| 1 | 顺序模式（默认） | 逐个处理分块，向后兼容 |
| >1 | 批处理模式 | 按文档索引分批，提高吞吐量 |

**批处理原理**：
- 同一批次包含来自**不同文档**的**相同索引**分块
- 同一文档的分块**按顺序**处理，确保前置事件上下文正确
- 支持所有前置事件上下文模式（none/full/sliding_window）

**示例**（batch_size=3）：
```
输入文档分块:
  doc0: [c0, c1, c2]
  doc1: [c0, c1]
  doc2: [c0]

批处理执行:
  批次1: [doc0-c0, doc1-c0, doc2-c0]  # 每个文档的第0个分块
  批次2: [doc0-c1, doc1-c1]           # 每个文档的第1个分块
  批次3: [doc0-c2]                    # 剩余分块
```

**前置事件上下文说明**：

前置事件上下文功能允许在抽取当前分块时，将之前分块中已抽取的事件作为上下文信息传递给 LLM。这有助于解析相对时间表达式（如"8岁时"、"两年后"）。

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `none` | 不使用前置事件上下文（默认） | 分块间时间信息独立 |
| `full` | 全量模式：所有前置分块的事件 | 文档较短或事件不多时 |
| `sliding_window` | 第一个分块 + 当前分块前 N 个分块的事件 | 长文档，控制上下文长度 |

**滑动窗口模式示例**（window_size=3）：
- 分块 0：无前置事件
- 分块 1：分块 0 的事件
- 分块 2：分块 0 + 分块 1 的事件
- 分块 3：分块 0 + 分块 1 + 分块 2 的事件
- 分块 4：分块 0 + 分块 1 + 分块 2 + 分块 3 的事件
- 分块 5：分块 0 + 分块 2 + 分块 3 + 分块 4 的事件（始终保留分块 0）

**建议**：
- temperature 保持低值（0.1-0.2）确保抽取结果稳定
- 长文档增加 timeout 避免超时
- 传记类文档建议使用 `sliding_window` 模式，以便利用出生日期等关键时间锚点
- 默认 `batch_size=1` 确保稳定性；API 限速宽松时可设置为 3-5 提高速度
- 传记类长文档建议保持 `batch_size=1`，因为前置事件上下文很重要

### 事件过滤配置 (event_filter)

```json
{
  "event_filter": {
    "enabled": true                         // 是否启用事件过滤
  }
}
```

**功能说明**：

事件过滤阶段用于去除 chunk 重叠导致的重复事件。由于分块时存在 overlap，同一个事件可能被多个 chunk 分别抽取。过滤器会：

1. **去除完全重复的事件**：event_description 和时间字段完全一致的事件仅保留一个
2. **保留最细时间粒度**：对于描述相同但时间粒度不同的事件（如 "2008" vs "2008-07"），仅保留粒度最细的版本
3. **合并 chunk 引用**：被合并的事件的所有来源 chunk ID 直接写入 `chunk_id` 字段（逗号分隔，如 `"doc-00000-chunk-0000,doc-00000-chunk-0001"`）

**独立使用**：

```bash
# 过滤单文档事件
python -m timeqa_agent.event_filter -i data/timeqa/event/test_doc0.json -o data/timeqa/event_filter/test_doc0.json

# 使用配置文件
python -m timeqa_agent.event_filter -i data/timeqa/event/test.json -o data/timeqa/event_filter/test.json --config configs/timeqa_config.json
```

### 事件检查配置 (event_validator)

```json
{
  "event_validator": {
    "enabled": true,                              // 是否启用事件检查
    "model": "deepseek-chat",                     // LLM 模型名称（用于时间格式纠正）
    "base_url": "https://api.deepseek.com/chat/completions",  // API 端点
    "temperature": 0,                             // 生成温度
    "max_retries": 3,                             // 最大重试次数
    "timeout": 60                                 // 请求超时（秒）
  }
}
```

**功能说明**：

事件检查阶段用于验证并纠正事件的时间格式。时间格式必须符合以下规范：
- `YYYY`（如 "2008"）
- `YYYY-MM`（如 "2008-07"）
- `YYYY-MM-DD`（如 "2008-07-15"）
- `null`（时间无法从上下文确定）

检查流程：
1. **格式检查**：使用正则表达式检查 `time_start` 和 `time_end` 是否符合规范格式
2. **规则修复**：对于不规范的时间字符串，首先尝试使用正则表达式提取合法时间
3. **LLM 纠正**：如果规则修复失败，调用 LLM 根据上下文重新解析和纠正时间

**独立使用**：

```bash
# 检查单文档事件
python -m timeqa_agent.event_validator -i data/timeqa/event/test_doc0.json -o data/timeqa/event_validate/test_doc0.json

# 使用配置文件
python -m timeqa_agent.event_validator -i data/timeqa/event/test.json -o data/timeqa/event_validate/test.json --config configs/timeqa_config.json
```

### 实体消歧配置 (disambiguator)

```json
{
  "disambiguator": {
    "embed_model": "text-embedding-3-small",  // 嵌入模型
    "embed_base_url": "http://...",           // 嵌入 API 端点
    "embed_batch_size": 100,                  // 嵌入批处理大小
    "similarity_threshold": 0.85,             // 相似度阈值。高于此值的实体会被合并
    "canonical_name_weight": 2.0              // 规范名称权重。用于优先选择更完整的名称
  }
}
```

**建议**：
- similarity_threshold 0.85-0.90：较高阈值减少误合并，但可能遗漏同义实体
- similarity_threshold 0.75-0.85：较低阈值增加召回，但可能误合并不同实体
- 人名消歧建议 0.85+，组织名建议 0.80+

### 时间线配置 (timeline)

```json
{
  "timeline": {
    "model": "deepseek-v3.1-terminus",  // LLM 模型
    "base_url": "http://...",           // API 端点
    "temperature": 0.1,                 // 生成温度
    "max_retries": 3,                   // 最大重试次数
    "timeout": 180,                     // 请求超时（秒）

    // 迭代式抽取配置
    "enable_iterative": false,          // 是否启用迭代抽取
    "iterative_batch_size": 20,         // 每批事件数量
    "include_timeline_context": true,   // 是否在提示词中包含已有时间线
    "max_context_timelines": 50,        // 最多包含多少条时间线在上下文中
    "sort_events_by_time": true         // 分批前是否按时间排序
  }
}
```

**迭代式抽取功能说明**：

- **功能介绍**：迭代式抽取将实体的事件分批处理，而不是一次性处理所有事件。第一批正常进行时间线聚类，后续批次将已识别的时间线作为上下文，帮助 LLM 更好地将新事件分配到合适的时间线中。

- **适用场景**：
  - 实体拥有大量事件（如知名人物、大型组织）
  - 希望改善时间线聚类的连贯性和准确性
  - 需要控制单次 LLM 调用的上下文大小

- **参数说明**：
  - `enable_iterative`：是否启用迭代抽取（默认 false，保持向后兼容）
  - `iterative_batch_size`：每批处理的事件数量（默认 20）
    - 过小：API 调用次数增加，成本上升
    - 过大：迭代效果减弱，接近单次抽取
  - `include_timeline_context`：是否在提示词中包含已有时间线信息（默认 true）
  - `max_context_timelines`：上下文中最多包含多少条时间线（默认 50，防止提示词过长）
  - `sort_events_by_time`：分批前是否按时间排序事件（默认 true，保持时间连贯性）

- **使用示例**：
  ```bash
  # 启用迭代抽取，每批 15 个事件
  # 修改 configs/timeqa_config.json 中的配置：
  # "enable_iterative": true,
  # "iterative_batch_size": 15

  python -m timeqa_agent.pipeline --split test --start timeline --end timeline
  ```

- **性能影响**：
  - API 调用次数：从 1 次变为 `ceil(事件数 / batch_size)` 次
  - 总处理时间：增加，但单次调用更快（上下文更小）
  - 成本：与 API 调用次数成正比


### 图存储配置 (graph_store)

```json
{
  "graph_store": {
    "store_original_sentence": true,   // 存储原始句子。开启便于溯源，但增加存储
    "store_chunk_metadata": true,      // 存储分块元数据。开启便于调试
    "store_entity_aliases": true       // 存储实体别名。开启支持别名查询
  }
}
```

**建议**：
- 生产环境可关闭 store_chunk_metadata 减少存储
- store_original_sentence 建议保持开启，便于验证抽取结果

### 查询解析器配置 (query_parser)

```json
{
  "query_parser": {
    "enabled": true,              // 是否启用查询解析器
    "model": "deepseek-chat",     // LLM 模型名称
    "base_url": "http://...",     // API 端点
    "temperature": 0,             // 生成温度。默认为 0，保证输出稳定一致
    "max_retries": 3,             // 最大重试次数
    "timeout": 180                // 请求超时（秒）
  }
}
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | true | 是否启用查询解析器。禁用时返回原始问题作为主干 |
| temperature | float | 0 | 生成温度。建议保持为 0 确保输出稳定 |

**功能说明**：
查询解析器将用户问题分解为两个部分：
1. **问题主干**：去除时间约束后的核心问题
2. **时间约束**：显式（如 "in 2007"）或隐式（如 "during the Beijing Olympics"）

然后基于问题主干生成三层检索语句：
- **实体查询**：标准化名称 + 简短描述
- **时间线查询**：时间线名称 + 描述 + 相关实体
- **事件查询**：将问题转为多个陈述句（基于常识推断可能的答案）

### 检索器配置 (retriever)

```json
{
  "retriever": {
    // === 通用参数 ===
    "top_k": 10,                    // 返回结果数量
    "score_threshold": 0.0,         // 分数阈值，低于此分数的结果被过滤
    "include_metadata": true,       // 是否返回元数据
    "fuzzy_match": true,            // 是否模糊匹配（关键词检索）
    "case_sensitive": false,        // 是否大小写敏感

    // === 语义检索配置 ===
    "semantic_model_type": "contriever",  // "contriever", "dpr", "bge-m3"
    "semantic_model_name": "./models/contriever-msmarco",
    "semantic_model_device": "cpu",       // "cpu", "cuda", "cuda:0"
    "contriever_normalize": true,
    "dpr_ctx_encoder": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr_question_encoder": "facebook/dpr-question_encoder-single-nq-base",
    "bge_m3_model_path": "./models/bge-m3/bge-m3",
    "embedding_dim": 768,           // 嵌入维度，需与嵌入模型匹配
    "embed_batch_size": 32,         // 嵌入批处理大小
    "similarity_threshold": 0.5,    // 语义相似度阈值

    // === 关键词检索配置 ===
    "keyword_algorithm": "bm25",          // "bm25", "tfidf"
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "bm25_use_stemming": false,
    "bm25_remove_stopwords": false,
    "min_keyword_length": 2,        // 最小关键词长度

    // === 向量索引参数 ===
    "vector_index_type": "flat",    // 索引类型，仅 flat 生效
    "vector_metric": "cosine",      // 距离度量: cosine, l2, ip

    // === 混合检索参数 ===
    "keyword_weight": 0.3,          // 关键词检索权重
    "semantic_weight": 0.7,         // 语义检索权重
    "fusion_mode": "rrf",           // 融合模式: rrf, weighted_sum, max_score, interleave
    "rrf_k": 60.0,                  // RRF 参数 k
    "enable_keyword": true,         // 启用关键词检索
    "enable_semantic": true         // 启用语义检索
  }
}
```

**配置说明**：

完整配置示例：
- `config_examples/contriever_bm25_config.json` - Contriever + BM25（推荐）
- `config_examples/dpr_bm25_config.json` - DPR + BM25（高性能）
- `config_examples/bge_m3_tfidf_config.json` - BGE-M3 + TF-IDF（兼容）

**实现状态**：
| 参数 | 状态 | 说明 |
|------|------|------|
| vector_index_type | ⚠️ 部分实现 | 仅 `flat` 生效，
| 其他参数 | ✅ 已实现 | 正常工作 |

**检索器建议**：

| 场景 | 推荐配置 |
|------|----------|
| 精确实体查询 | keyword_weight=0.7, semantic_weight=0.3 |
| 语义相似查询 | keyword_weight=0.3, semantic_weight=0.7 |
| 纯关键词 | enable_semantic=false |
| 纯语义 | enable_keyword=false |

**融合模式说明**：
- `rrf`：Reciprocal Rank Fusion，推荐用于混合检索，对排名位置敏感
- `weighted_sum`：加权求和，简单直接
- `max_score`：取最大分数，适合高精度场景
- `interleave`：交替合并，保证多样性

### 路径配置

```json
{
  "data_dir": "data/timeqa",           // 数据根目录
  "corpus_dir": "data/timeqa/corpus",  // 语料库目录
  "output_dir": "data/timeqa/processed" // 输出目录
}
```

### 三层递进检索配置 (hierarchical)

```json
{
  "hierarchical": {
    "enabled": false,                     // 是否启用三层递进检索
    "k1_entities": 5,                     // 第一层：检索实体数量
    "k2_timelines": 10,                   // 第三层：筛选时间线数量
    "k3_events": 20,                      // 第三层：筛选事件数量
    "entity_score_threshold": 0.0,        // 第一层实体分数阈值
    "timeline_score_threshold": 0.0,      // 第三层时间线分数阈值
    "event_score_threshold": 0.0,         // 第三层事件分数阈值
    "include_intermediate_results": false  // 是否返回中间层结果（调试用）
  }
}
```

**检索流程**：

```
查询 Query
  │
  ├── 第一层：混合检索实体 → Top-K1 实体
  │
  ├── 第二层：通过图存储收集 K1 个实体的所有时间线和事件
  │
  └── 第三层：在候选时间线和事件中混合检索 → Top-K2 时间线 + Top-K3 事件
```

与投票检索器（VotingRetriever）的区别：
- **投票检索器**：三层并行检索 → 投票聚合 → 最终排名
- **三层递进检索器**：逐层递进过滤 → 逐步缩小检索范围 → 最终结果

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | false | 是否启用三层递进检索 |
| k1_entities | int | 5 | 第一层检索的实体数量 |
| k2_timelines | int | 10 | 第三层筛选的时间线数量 |
| k3_events | int | 20 | 第三层筛选的事件数量 |
| entity_score_threshold | float | 0.0 | 第一层实体分数阈值，低于此分数的实体被过滤 |
| timeline_score_threshold | float | 0.0 | 第三层时间线分数阈值 |
| event_score_threshold | float | 0.0 | 第三层事件分数阈值 |
| include_intermediate_results | bool | false | 是否返回中间层结果（第一层实体、第二层全量时间线/事件） |

**Python API 使用示例**：

```python
from timeqa_agent.retrievers import HierarchicalRetriever
from timeqa_agent.config import RetrieverConfig, HierarchicalConfig

# 配置
retriever_config = RetrieverConfig(top_k=10, use_tfidf=True)
hierarchical_config = HierarchicalConfig(
    enabled=True,
    k1_entities=5,
    k2_timelines=10,
    k3_events=20,
)

# 初始化
retriever = HierarchicalRetriever(
    graph_store=graph_store,
    embed_fn=embed_fn,
    retriever_config=retriever_config,
    hierarchical_config=hierarchical_config,
    index_dir="data/timeqa/indexes",
)

# 检索（使用配置中的 k1/k2/k3）
results = retriever.retrieve(query="人工智能的发展历程")
print(f"事件: {len(results.events)}, 时间线: {len(results.timelines)}")

# 检索时动态覆盖 k1/k2/k3
results = retriever.retrieve(query="深度学习", k1=3, k2=5, k3=10)

# 调试：查看中间层结果
results = retriever.retrieve_with_details(query="神经网络")
print(f"第一层实体: {[e.canonical_name for e in results.layer1_entities]}")
print(f"第二层收集: {len(results.layer2_all_events)} 事件, {len(results.layer2_all_timelines)} 时间线")
print(f"第三层筛选: {len(results.events)} 事件, {len(results.timelines)} 时间线")

# 查看事件溯源
for event in results.events[:3]:
    print(f"事件: {event.event_description}")
    print(f"  来源实体: {event.source_entity_names}")
    print(f"  分数: {event.hierarchical_score:.4f}")
```

---

## 更新日志

### 2026-02-07 - 检索器系统升级

#### 🎯 升级目标

将 TimeQA Agent 的检索器系统升级为支持多种先进算法：
- **语义检索**: Contriever（默认）、DPR、BGE-M3
- **关键词检索**: BM25（默认）、TF-IDF

#### 📝 修改文件清单

**已修改的文件**：

| 文件路径 | 修改内容 | 状态 |
|---------|---------|------|
| `timeqa_agent/config.py` | 扩展 RetrieverConfig，添加语义模型和关键词算法配置 | ✅ 完成 |
| `timeqa_agent/embeddings.py` | 新增 Contriever 和 DPR 嵌入函数支持 | ✅ 完成 |
| `timeqa_agent/retrievers/keyword_retriever.py` | 添加 BM25Index 类，支持 BM25 算法 | ✅ 完成 |
| `timeqa_agent/retrievers/semantic_retriever.py` | 支持配置驱动的模型自动加载 | ✅ 完成 |
| `timeqa_agent/retrievers/hybrid_retriever.py` | 适配新的检索器接口 | ✅ 完成 |

**新增的文件**：

| 文件路径 | 说明 | 状态 |
|---------|------|------|
| `download_contriever.py` | Contriever 模型下载脚本 | ✅ 完成 |
| `test_retrievers.py` | 检索器功能测试脚本 | ✅ 完成 |
| `usage_examples.py` | 使用示例脚本 | ✅ 完成 |
| `config_examples/contriever_bm25_config.json` | Contriever + BM25 配置示例 | ✅ 完成 |
| `config_examples/dpr_bm25_config.json` | DPR + BM25 配置示例 | ✅ 完成 |
| `config_examples/bge_m3_tfidf_config.json` | BGE-M3 + TF-IDF 配置示例 | ✅ 完成 |

#### 🔧 详细修改说明

**1. config.py**

新增配置项：
```python
# 语义检索配置
semantic_model_type: str = "contriever"  # "contriever", "dpr", "bge-m3"
semantic_model_name: str = "facebook/contriever-msmarco"
semantic_model_device: str = "cpu"
contriever_normalize: bool = True
dpr_ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"
dpr_question_encoder: str = "facebook/dpr-question_encoder-single-nq-base"
bge_m3_model_path: Optional[str] = "./models/bge-m3/bge-m3"

# 关键词检索配置
keyword_algorithm: str = "bm25"  # "bm25", "tfidf"
bm25_k1: float = 1.5
bm25_b: float = 0.75
bm25_use_stemming: bool = False
bm25_remove_stopwords: bool = False
```

**2. embeddings.py**

新增函数：
- `create_contriever_embed_fn()`: 创建 Contriever 嵌入函数
- `create_dpr_embed_fn()`: 创建 DPR 嵌入函数（双编码器）
- `create_embed_fn()`: 工厂函数，根据配置自动创建嵌入函数

特性：
- 支持 GPU/CPU 设备选择
- 支持批处理
- 支持向量归一化
- 自动均值池化（Contriever）

**3. keyword_retriever.py**

新增类：
- `BM25Index`: 基于 rank-bm25 的 BM25 索引实现

新增功能：
- 支持词干提取（可选）
- 支持停用词移除（可选）
- BM25 参数可配置（k1, b）

修改内容：
- `KeywordRetriever.__init__()`: 根据配置选择算法
- `_create_index()`: 工厂方法，创建 BM25 或 TF-IDF 索引
- 统一索引接口，兼容旧代码

**4. semantic_retriever.py**

修改内容：
- `SemanticRetriever.__init__()`: 支持配置驱动的模型加载
- 如果不提供 `embed_fn`，自动根据 `config` 创建
- 保持向后兼容（仍可手动传入 `embed_fn`）

**5. hybrid_retriever.py**

修改内容：
- `HybridRetriever.__init__()`: 接收 `config` 参数，自动创建检索器
- `_init_retrievers()`: 传递 `embed_fn` 给 `SemanticRetriever`
- `set_embed_fn()`: 更新为使用新的 API

#### ✅ 向后兼容性

所有修改**完全向后兼容**，旧代码无需修改即可运行。

#### 🎯 推荐配置

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| **通用检索** | Contriever + BM25 | 平衡性能和质量 |
| **高精度检索** | DPR + BM25 | 最佳检索质量 |
| **保持兼容** | BGE-M3 + TF-IDF | 使用已有模型 |

---

## 环境变量

需要设置 API Token：

```bash
export VENUS_API_TOKEN=your_token
# 或
export OPENAI_API_KEY=your_key
```
