"""
对向量检索后的数据进行重新排序
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.setting import QaConfig
from querying.query_pormpts import RERANK_PROMPT


@dataclass
class RetrievedEvidence:
    """检索到的证据片段(经过重排序后输出的结构信息)"""
    content: str
    source: str
    page: Optional[int]
    score: float
    vector_score: float
    llm_score: Optional[float] = None


class RerankPipeline:
    """Pipeline：召回 → 预过滤 → LLM 批量精排"""

    def __init__(self, config: QaConfig, llm: ChatTongyi) -> None:
        self._config = config
        self._llm = llm

        # 批量打分提示词
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", RERANK_PROMPT),
            ("human",
             "问题:{question}\n\n{chunks}\n\n请为以上每个候选片段打分（0-10分），按以下格式输出：\n候选片段 1: X.X\n候选片段 2: X.X\n...\n分数:")
        ])
        self._batch_chain = self._prompt | self._llm | StrOutputParser()

    def rerank(self,
               query: str,
               recalled_docs: list[Document],
               vec_scores: list[float],
               final_top_k: int = 3
               ) -> Tuple[list[RetrievedEvidence], str]:
        """
        重排序流程入口
        :param query: 检索查询
        :param recalled_docs:已召回的文档列表
        :param vec_scores: 对应的向量分数列表
        :param final_top_k: 最终输出的证据数量
        :return: (evidence_list, formatted_context)
        """

        # 预过滤
        pre_docs = self._filter(recalled_docs)
        # 对齐向量分数
        score_map = {doc.page_content: s for doc, s in zip(recalled_docs, vec_scores)}

        # LLM批量精排
        final_docs, llm_scores = self._llm_rerank_batch(query, pre_docs, top_k=final_top_k)

        # 组装数据
        evidence_list = []
        for i, doc in enumerate(final_docs):
            content = doc.page_content
            meta = doc.metadata or {}
            ev = RetrievedEvidence(
                content=content,
                source=meta.get("source", "Unknown"),
                page=meta.get("page"),
                score=llm_scores[i],
                vector_score=score_map.get(content, 0.0),
                llm_score=llm_scores[i]
            )
            evidence_list.append(ev)
        # 格式化上下文
        context_lines = []
        for idx, ev in enumerate(evidence_list, 1):
            page_str = f"第{ev.page + 1}页" if ev.page is not None else "未知页"
            context_lines.append(f"[证据{idx} | 来源：{ev.source} | {page_str}]\n{ev.content}")
        context = "\n\n".join(context_lines)

        return evidence_list, context


    def _filter(self, recalled_docs: list[Document], min_len: int = 40, max_len: int = 2000) -> list[Document]:
        """过滤处理已召回的文档列表"""
        filtered = []
        # 过滤过长和过短的文档内容
        for doc in recalled_docs:
            text = doc.page_content
            if min_len <= len(text) <= max_len:
                filtered.append(doc)

        # 去重
        seen = set()
        kept = []
        for doc in filtered:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                kept.append(doc)
        return kept

    def _llm_rerank_batch(self, question: str, documents: list[Document], top_k: int = 3) -> Tuple[
        list[Document], list[float]]:
        """
        批量打分：现对所有召回文档进行打分
        :param question: 用户问题
        :param documents: 待精排的文档列表
        :param top_k: 最终保留的文档数量
        :return: (排序后的文档列表,LLM分数列表)
        """

        if not documents:
            return [], []

        # 构建批量打分提示词
        chunks_text = []
        for idx, doc in enumerate(documents, 1):
            # 截断过长的文本，避免超出上下文
            content = doc.page_content[:1000] if len(doc.page_content) > 1000 else doc.page_content
            chunks_text.append(f"候选片段{idx}:\n{content}")

        chunks_text = "\n\n".join(chunks_text)

        try:
            # 批量请求
            result_text = self._batch_chain.invoke({
                "question": question,
                "chunks": chunks_text
            })
            """
             '候选片段 1: 3.0  \n候选片段 2: 5.0  \n候选片段 3: 0.0  
            """
            # 解析模型结果
            # 1.解析分数
            scores = self._parse_batch_scores(result_text, len(documents))

            # 限制分数范围
            scores = [max(0.0, min(10, s)) for s in scores]

            # 按分数降序排行
            scored = list(zip(scores, documents))
            scored.sort(key=lambda x: x[0], reverse=True)

            docs = [d for _, d in scored[:top_k]]
            final_scores = [s for s, _ in scored[:top_k]]

            return docs, final_scores

        except Exception as e:
            print(f"打分失败:{e}")
            return [], []

    def _parse_batch_scores(self, text: str, expected_count: int) -> list[float]:
        """
        解析批量打分结果
        :param text: llm返回文本
        :param expected_count: 期望分数的数量
        :return: 分数列表
        """
        scores = []

        # 尝试标准格式：候选片段 1: 8.5 或 候选片段1:8.5
        pattern = r'候选片段\s*(\d+)[:：]\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)

        if matches:
            # 如果成功匹配，按索引组织
            score_dict = {int(idx): float(score) for idx, score in matches}
            """
            {1: 3.0, 2: 5.0, 3: 0.0, 4: 3.0, 5: 10.0, 6: 3.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0}
            """
            scores = [score_dict.get(i + 1, 0.0) for i in range(expected_count)]
        else:
            # 完全解析失败，返回默认分数
            scores = [0.0] * expected_count

        return scores
