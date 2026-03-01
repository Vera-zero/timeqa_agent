"""
Document Chunking Module

Supports two chunking strategies:
1. Fixed Size Chunking: Split by token count with overlap (using tiktoken)
2. Sentence Chunking: Split by sentence boundaries for semantic coherence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, falling back to character-based chunking")

from .config import ChunkStrategy, ChunkConfig


@dataclass
class Chunk:
    """文档分块"""
    chunk_id: str                  # 分块唯一标识
    content: str                   # 分块内容

    # 文档元信息
    doc_id: str                    # 所属文档ID
    doc_title: str                 # 文档标题
    source_idx: str                # 原始索引

    # 分块元信息
    chunk_index: int               # 分块在文档中的索引
    start_char: int                # 起始字符位置
    end_char: int                  # 结束字符位置
    strategy: ChunkStrategy        # 使用的分块策略
    token_count: int = 0           # 分块的 token 数量

    # 可选元信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # strategy 可能是 ChunkStrategy 枚举或字符串
        strategy_value = self.strategy.value if hasattr(self.strategy, 'value') else self.strategy
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_idx": self.source_idx,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "strategy": strategy_value,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建"""
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            source_idx=data["source_idx"],
            chunk_index=data["chunk_index"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            strategy=ChunkStrategy(data["strategy"]),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )


class DocumentChunker:
    """文档分块器"""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # 初始化 tiktoken encoder (用于 token 数分块)
        self._encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
                print("Using tiktoken for token-based chunking")
            except Exception as e:
                print(f"Failed to initialize tiktoken encoder: {e}")
                self._encoder = None

        # 句子分割正则表达式
        # 匹配句号、问号、感叹号（包括中英文）
        self._sentence_pattern = re.compile(
            r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[A-Z])'
        )

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception as e:
                print(f"Error encoding text: {e}, falling back to character count")
                return len(text)
        # 如果 tiktoken 不可用，回退到字符数（近似）
        return len(text)

    def _encode_text(self, text: str) -> List[int]:
        """将文本编码为 token 列表"""
        if self._encoder is not None:
            try:
                return self._encoder.encode(text)
            except Exception as e:
                print(f"Error encoding text: {e}")
                return []
        return []

    def _decode_tokens(self, tokens: List[int]) -> str:
        """将 token 列表解码为文本"""
        if self._encoder is not None:
            try:
                return self._encoder.decode(tokens)
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                return ""
        return ""
    
    def chunk_document(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        source_idx: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        对文档进行分块
        
        Args:
            content: 文档内容
            doc_id: 文档ID
            doc_title: 文档标题
            source_idx: 原始索引
            metadata: 额外元信息
            
        Returns:
            分块列表
        """
        if self.config.strategy == ChunkStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(
                content, doc_id, doc_title, source_idx, metadata
            )
        else:
            return self._chunk_by_sentence(
                content, doc_id, doc_title, source_idx, metadata
            )
    
    def _chunk_fixed_size(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        source_idx: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """固定大小分块（基于 token 数）"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # 如果 tiktoken 可用，使用基于 token 的分块
        if self._encoder is not None:
            return self._chunk_fixed_size_by_tokens(
                content, doc_id, doc_title, source_idx, metadata
            )
        else:
            # 回退到基于字符的分块
            return self._chunk_fixed_size_by_chars(
                content, doc_id, doc_title, source_idx, metadata
            )

    def _chunk_fixed_size_by_tokens(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        source_idx: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """基于 token 数的固定大小分块"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # 将整个文本编码为 tokens
        tokens = self._encode_text(content)
        if not tokens:
            # 如果编码失败，回退到字符分块
            return self._chunk_fixed_size_by_chars(
                content, doc_id, doc_title, source_idx, metadata
            )

        total_tokens = len(tokens)

        # 如果 token 数小于等于分块大小，直接返回整个文档
        if total_tokens <= chunk_size:
            chunk_id = f"{doc_id}-chunk-0000"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=doc_id,
                doc_title=doc_title,
                source_idx=source_idx,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                strategy=ChunkStrategy.FIXED_SIZE,
                token_count=total_tokens,
                metadata=metadata or {},
            ))
            return chunks

        # 按 token 数分块
        chunk_index = 0
        start_token_idx = 0

        # 用于追踪字符位置（用于 start_char 和 end_char）
        char_positions = []
        current_pos = 0
        for i in range(len(tokens) + 1):
            # 解码前 i 个 tokens 来获取字符位置
            decoded = self._decode_tokens(tokens[:i])
            char_positions.append(len(decoded))

        while start_token_idx < total_tokens:
            end_token_idx = min(start_token_idx + chunk_size, total_tokens)

            # 提取这个分块的 tokens
            chunk_tokens = tokens[start_token_idx:end_token_idx]

            # 解码为文本
            chunk_content = self._decode_tokens(chunk_tokens).strip()

            if chunk_content:
                # 计算字符位置
                start_char = char_positions[start_token_idx]
                end_char = char_positions[end_token_idx]

                chunk_id = f"{doc_id}-chunk-{chunk_index:04d}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_idx=source_idx,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    strategy=ChunkStrategy.FIXED_SIZE,
                    token_count=len(chunk_tokens),
                    metadata=metadata or {},
                ))
                chunk_index += 1
                print(f"Created chunk {chunk_id} with {len(chunk_tokens)} tokens (start: {start_token_idx}, end: {end_token_idx})")

            # 下一个分块的起始位置（考虑重叠）
            start_token_idx = end_token_idx - overlap
            if start_token_idx <= chunks[-1].start_char if chunks else 0:
                start_token_idx = end_token_idx  # 避免无限循环

        # 处理最后一个分块：如果太小，合并到上一个分块
        if len(chunks) > 1:
            last_chunk_tokens = self._count_tokens(chunks[-1].content)
            min_chunk_tokens = self.config.min_chunk_size  # 假设 min_chunk_size 也是 token 数
            if last_chunk_tokens < min_chunk_tokens:
                print(f"最后一个分块大小 ({last_chunk_tokens} tokens) 小于最小限制 ({min_chunk_tokens} tokens)，进行合并")
                last_chunk = chunks[-1]
                prev_chunk = chunks[-2]

                # 合并内容
                merged_content = prev_chunk.content + ' ' + last_chunk.content

                # 更新倒数第二个分块为合并后的分块
                chunks[-2] = Chunk(
                    chunk_id=prev_chunk.chunk_id,
                    content=merged_content.strip(),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_idx=source_idx,
                    chunk_index=prev_chunk.chunk_index,
                    start_char=prev_chunk.start_char,
                    end_char=last_chunk.end_char,
                    strategy=ChunkStrategy.FIXED_SIZE,
                    token_count=self._count_tokens(merged_content.strip()),
                    metadata=metadata or {},
                )
                # 删除最后一个分块
                chunks.pop()

        return chunks

    def _chunk_fixed_size_by_chars(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        source_idx: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """基于字符数的固定大小分块（回退方案）"""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(content) <= chunk_size:
            # 内容小于分块大小，直接返回整个文档作为一个分块
            chunk_id = f"{doc_id}-chunk-0000"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=doc_id,
                doc_title=doc_title,
                source_idx=source_idx,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                strategy=ChunkStrategy.FIXED_SIZE,
                token_count=self._count_tokens(content),
                metadata=metadata or {},
            ))
            return chunks
        
        start = 0
        end = 0
        chunk_index = 0
        
        while end < len(content):
            end = start + chunk_size
            
            if end >= len(content):
                end = len(content)
            elif self.config.preserve_sentences:
                # 尝试在句子边界处分割
                # 在 end 位置前后寻找最近的句子结束符
                search_start = max(start + chunk_size // 2, start)
                search_text = content[search_start:min(end + 100, len(content))]
                
                # 寻找句子结束符
                for sep in ['. ', '? ', '! ', '。', '？', '！']:
                    pos = search_text.find(sep)
                    if pos != -1:
                        end = search_start + pos + len(sep)
                        break
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk_id = f"{doc_id}-chunk-{chunk_index:04d}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_idx=source_idx,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    strategy=ChunkStrategy.FIXED_SIZE,
                    token_count=self._count_tokens(chunk_content),
                    metadata=metadata or {},
                ))
                chunk_index += 1
            print (f"Created chunk {chunk_id} with size {len(chunk_content)} chars (start: {start}, end: {end})")
            # 下一个分块的起始位置（考虑重叠）
            start = end - overlap
            if start <= chunks[-1].start_char if chunks else 0:
                start = end  # 避免无限循环


        # 处理最后一个分块：如果太小，合并到上一个分块
        if len(chunks) > 1 and len(chunks[-1].content) < self.config.min_chunk_size:
            print(f"最后一个分块大小小于最小限制，进行合并")
            last_chunk = chunks[-1]
            prev_chunk = chunks[-2]
            start = prev_chunk.start_char
            end = last_chunk.end_char
            chunk_content = content[start:end].strip()
            # 合并内容
            merged_content = prev_chunk.content + ' ' + last_chunk.content

            # 更新倒数第二个分块为合并后的分块
            chunks[-2] = Chunk(
                chunk_id=prev_chunk.chunk_id,
                content=chunk_content,
                doc_id=doc_id,
                doc_title=doc_title,
                source_idx=source_idx,
                chunk_index=prev_chunk.chunk_index,
                start_char=prev_chunk.start_char,
                end_char=last_chunk.end_char,
                strategy=ChunkStrategy.FIXED_SIZE,
                token_count=self._count_tokens(chunk_content),
                metadata=metadata or {},
            )
            # 删除最后一个分块
            chunks.pop()

        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 先用正则分割
        sentences = self._sentence_pattern.split(text)
        
        # 清理空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 如果分割效果不好，使用简单的分割符
        if len(sentences) <= 1 and len(text) > 100:
            # 尝试用句号分割
            parts = re.split(r'(?<=[.!?。！？])\s*', text)
            sentences = [p.strip() for p in parts if p.strip()]
        
        return sentences

    def _merge_short_sentences(self, sentences: List[str]) -> List[str]:
        """合并长度小于等于3的短句子到前一个句子"""
        if not sentences:
            return sentences
        
        merged = []
        for sentence in sentences:
            if len(sentence) <= 3 and merged:
                merged[-1] = merged[-1] + ' ' + sentence
            else:
                merged.append(sentence)
        
        return merged


    def _chunk_by_sentence(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        source_idx: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """按句子分块"""
        chunks = []
        sentences = self._split_sentences(content)
        
        if not sentences:
            return chunks
        
        # 合并短句子
        sentences = self._merge_short_sentences(sentences)

        current_chunk_sentences = []
        current_chunk_start = 0
        current_pos = 0
        chunk_index = 0
        
        for idx,sentence in enumerate(sentences):
            # 找到句子在原文中的位置
            sent_start = content.find(sentence, current_pos)
            if sent_start == -1:
                sent_start = current_pos
            sent_end = sent_start + len(sentence)
            
            # 检查是否需要创建新分块
            current_text = ' '.join(current_chunk_sentences + [sentence])
            
            should_split = (
                len(current_chunk_sentences) >= self.config.max_sentences or
                len(current_text) > self.config.max_chunk_size
            )
            
            if should_split and current_chunk_sentences:
                # 保存当前分块
                chunk_content = ' '.join(current_chunk_sentences)
                chunk_id = f"{doc_id}-chunk-{chunk_index:04d}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_idx=source_idx,
                    chunk_index=chunk_index,
                    start_char=current_chunk_start,
                    end_char=sent_start,
                    strategy=ChunkStrategy.SENTENCE,
                    token_count=self._count_tokens(chunk_content),
                    metadata=metadata or {},
                ))
                chunk_index += 1
                
                # 开始新分块 - 添加句子重叠逻辑
                overlap_sentences = []
                if self.config.sentence_overlap > 0 and len(current_chunk_sentences) > 0:
                    # 从当前分块的末尾取重叠句子（直接使用索引）
                    overlap_sentences = sentences[max(0, idx  - self.config.sentence_overlap):idx]
                
                # 新分块包含重叠句子和当前句子
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_start = sent_start - sum(len(s) + 1 for s in overlap_sentences)

            else:
                current_chunk_sentences.append(sentence)
            
            current_pos = sent_end
        
        # 处理最后一个分块
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            
            # 检查是否太小，需要合并到上一个分块
            if chunks and len(chunk_content) < self.config.min_chunk_size:
                # 合并到上一个分块
                last_chunk = chunks[-1]
                merged_content = last_chunk.content + ' ' + chunk_content
                
                # 只有合并后不超过最大限制才合并
                if len(merged_content) <= self.config.max_chunk_size * 1.5:
                    chunks[-1] = Chunk(
                        chunk_id=last_chunk.chunk_id,
                        content=merged_content,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_idx=source_idx,
                        chunk_index=last_chunk.chunk_index,
                        start_char=last_chunk.start_char,
                        end_char=len(content),
                        strategy=ChunkStrategy.SENTENCE,
                        token_count=self._count_tokens(merged_content),
                        metadata=metadata or {},
                    )
                else:
                    # 创建新分块
                    chunk_id = f"{doc_id}-chunk-{chunk_index:04d}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_idx=source_idx,
                        chunk_index=chunk_index,
                        start_char=current_chunk_start,
                        end_char=len(content),
                        strategy=ChunkStrategy.SENTENCE,
                        token_count=self._count_tokens(chunk_content),
                        metadata=metadata or {},
                    ))
            else:
                chunk_id = f"{doc_id}-chunk-{chunk_index:04d}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_idx=source_idx,
                    chunk_index=chunk_index,
                    start_char=current_chunk_start,
                    end_char=len(content),
                    strategy=ChunkStrategy.SENTENCE,
                    token_count=self._count_tokens(chunk_content),
                    metadata=metadata or {},
                ))
        
        return chunks
    
    def chunk_corpus(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        对整个文档库进行分块
        
        Args:
            documents: 文档列表，每个文档包含 doc_id, title, content, source_idx
            
        Returns:
            所有分块的列表
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                doc_title=doc["title"],
                source_idx=doc["source_idx"],
            )
            all_chunks.extend(chunks)
        
        return all_chunks
