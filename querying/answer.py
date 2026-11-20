from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from querying.query_pormpts import READING_HELPER_PROMPT


class AnswerGenerator:
    """答案生成器:根据问题、对话历史，解锁上下文，生成答案"""

    def __init__(self,llm:ChatTongyi)->None:
        """
        初始化生成器，预编译 prompt 链
        :param llm: 已配置好的通义千问 LLM 实例
        """
        self._llm=llm

        self._prompt=ChatPromptTemplate.from_messages([
            ("system",READING_HELPER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human","{user_message}")
        ])

        self._chain=self._prompt|self._llm|StrOutputParser()


    def generate(self,question:str,chat_history:list[BaseMessage],context:str)->str:
        """
        生成答案
        :param question:用户原始问题
        :param chat_history: 上下文历史记录
        :param context: 格式化后的上下文字符串
        :return: 答案
        """

        return self._chain.invoke({
            "user_message":question,
            "chat_history":chat_history,
            "context":context
        })