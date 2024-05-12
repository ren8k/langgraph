from langchain.globals import set_debug

set_debug(False)  # debug時はTrue

from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ユーザー入力
user_input = "Amazonの生成AIの分野での注力範囲は？"

# Retrieve用のプロンプトの定義
prompt_pre = PromptTemplate.from_template(
    """
    あなたはquestionから、検索ツールへの入力となる検索キーワードを考えます。
    questionに後続処理への指示（例：「説明して」「要約して」）が含まれる場合は取り除きます。
    検索キーワードは文章では無く簡潔な単語で指定します。
    検索キーワードは複数の単語を受け付ける事が出来ます。
    検索キーワードは日本語が標準ですが、ユーザー問い合わせに含まれている英単語はそのまま使用してください。
    回答形式は文字列です。
    <question>{question}</question>
    """
)

# 回答生成用のプロンプトの定義
prompt_main = PromptTemplate.from_template(
    """
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
    """
)

# LLMの定義
LLM = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.0},
)

# Retriever(Knowledge Base)の定義
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="APNZCYJTKD",  # ナレッジベースIDを指定
    region_name="us-east-1",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
            # "overrideSearchType": "HYBRID",
        }
    },
)

# chainの定義
chain = (
    {
        "context": prompt_pre | LLM | StrOutputParser() | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt_main
    | LLM
    | StrOutputParser()
)

# chainの実行
answer = chain.invoke({"question": user_input})

# chainの実行
print(answer)
