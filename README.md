# TimeQA Agent

时间事件抽取与时间线分析系统。从文本中抽取时间事件，进行实体消歧，构建时间线，并存储为知识图谱。

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

本项目需要嵌入模型进行语义检索。请下载并安装以下模型之一：

- **BGE-M3**（推荐）：`BAAI/bge-m3`
- **BGE-Large-ZH**：`BAAI/bge-large-zh-v1.5`
- 或其他兼容的嵌入模型

下载模型后，将模型文件放置于 `models/` 目录下，并在配置文件中指定模型路径。

## 项目结构

```
timeqa_agent/
├── timeqa_agent/
│   ├── config.py              # 配置模块
│   ├── chunker.py             # 文档分块器
│   ├── event_extractor.py     # 时间事件抽取器
│   ├── entity_disambiguator.py # 实体消歧器
│   ├── timeline_extractor.py  # 时间线抽取器
│   ├── graph_store.py         # 知识图谱存储
│   ├── graph_store_cli.py     # 图存储命令行工具
│   ├── retriever_cli.py       # 检索器命令行工具
│   ├── query_parser.py        # 查询解析器
│   ├── query_parser_cli.py    # 查询解析器命令行工具
│   └── pipeline.py            # 抽取流水线
├── configs/
│   └── timeqa_config.json     # 默认配置文件
├── data/timeqa/
│   ├── corpus/                # 语料库 (test/train/validation.json)
│   └── raw/                   # 原始数据
└── pyproject.toml
```

## 流水线阶段

| 阶段 | 名称 | 输入 | 输出 |
|------|------|------|------|
| 1 | chunk | 语料库文档 | 文档分块 |
| 2 | event | 分块 | 时间事件 |
| 3 | disambiguate | 事件 | 实体聚类 |
| 4 | timeline | 事件+实体 | 时间线 |
| 5 | graph | 全部 | 知识图谱 |

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

```bash
# 交互式模式
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json

# 混合检索（关键词 + 语义）
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json search "when did John join the company"

# 关键词检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json keyword "John Smith"

# 语义检索
python -m timeqa_agent.retriever_cli -g data/timeqa/graph/test.json semantic "career changes"

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

### 检索器命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| --graph | -g | 图文件路径（必需） | - |
| --config | -c | 配置文件路径 | None |
| --type | -t | 目标类型: all/entity/event/timeline | all |
| --top-k | -k | 返回结果数量 | 10 |
| --fusion | | 融合模式: rrf/weighted_sum/max_score/interleave | rrf |
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
| entity \<name\> \<query\> | 带实体上下文的检索 |
| type \<entity\|event\|timeline\|all\> | 设置目标类型过滤 |
| topk \<n\> | 设置返回数量 |
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
├── disambiguate/
├── timeline/
└── graph/
```

## Python API

```python
from timeqa_agent import (
    DocumentChunker,
    EventExtractor,
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
print(output.parse_result.question_stem)  # "Which team did Attaphol Buspakom play for?"
print(output.parse_result.time_constraint.constraint_type)  # "explicit"
print(output.retrieval_queries.entity_query)  # "Attaphol Buspakom, a Thai professional football player"
print(output.retrieval_queries.event_queries)  # ["Attaphol Buspakom played for...", ...]

# 分步调用
parse_result = parser.parse_question("Where did John work during the Olympics?")
print(parse_result.question_stem)  # "Where did John work?"
print(parse_result.time_constraint.constraint_type)  # "implicit"

queries = parser.generate_retrieval_queries("Where did John work?")
print(queries.entity_query)
print(queries.timeline_query)
print(queries.event_queries)
```

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
    "min_chunk_size": 500,         // 最小分块大小（sentence 策略专用）
    "max_chunk_size": 2000,        // 最大分块大小
    "preserve_sentences": true     // 是否保持句子完整性（fixed_size 策略专用）
  }
}
```

**策略说明**：
- `fixed_size`：按固定字符数分块，使用 chunk_size、chunk_overlap、preserve_sentences
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
    "batch_size": 1,                    // 批处理大小。并发抽取数量
    "include_implicit_time": true,      // 是否抽取隐式时间（如"去年"、"上个月"）
    "enable_multi_round": true,         // 是否启用多轮抽取
    "max_rounds": 2,                    // 最大抽取轮数
    "review_temperature": 0.0,          // 审查轮次的温度参数
    "prior_events_context_mode": "none", // 前置事件上下文模式: none, full, sliding_window
    "prior_events_window_size": 3       // 滑动窗口大小（仅 sliding_window 模式有效）
  }
}
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
- include_implicit_time 开启可捕获更多时间信息，但可能增加噪音
- 传记类文档建议使用 `sliding_window` 模式，以便利用出生日期等关键时间锚点

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
    "timeout": 180                      // 请求超时（秒）
  }
}
```

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

    // === 关键词检索参数 ===
    "use_tfidf": true,              // 使用 TF-IDF 排序
    "min_keyword_length": 2,        // 最小关键词长度

    // === 语义检索参数 ===
    "embedding_dim": 768,           // 嵌入维度，需与嵌入模型匹配
    "embed_batch_size": 32,         // 嵌入批处理大小
    "similarity_threshold": 0.5,    // 语义相似度阈值
    "cache_embeddings": true,       // [未实现] 缓存嵌入向量

    // === 向量索引参数 ===
    "vector_index_type": "flat",    // 索引类型，仅 flat 生效，hnsw 未实现
    "vector_metric": "cosine",      // 距离度量: cosine, l2, ip
    "hnsw_m": 16,                   // [未实现] HNSW M 参数
    "hnsw_ef_construction": 200,    // [未实现] HNSW 构建参数
    "hnsw_ef_search": 50,           // [未实现] HNSW 搜索参数

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

**实现状态**：
| 参数 | 状态 | 说明 |
|------|------|------|
| vector_index_type | ⚠️ 部分实现 | 仅 `flat` 生效，`hnsw` 索引未实现 |
| hnsw_* | ❌ 未实现 | HNSW 相关参数暂未使用 |
| cache_embeddings | ❌ 未实现 | 嵌入缓存功能暂未使用 |
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

## 环境变量

需要设置 API Token：

```bash
export VENUS_API_TOKEN=your_token
# 或
export OPENAI_API_KEY=your_key
```
