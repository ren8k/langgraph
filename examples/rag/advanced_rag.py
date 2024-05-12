from langchain.globals import set_debug

set_debug(False)  # debug時はTrue

import utils
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# ユーザー入力
user_input = "Amazonの生成AIの分野での注力範囲は？"

# クエリ拡張用のプロンプト
prompt_pre = PromptTemplate.from_template(
    """
    検索エンジンに入力するクエリを最適化し、様々な角度から検索を行うことで、より適切で幅広い検索結果が得られるようにします。
    具体的には、類義語や日本語と英語の表記揺れを考慮し、多角的な視点からクエリを生成します。

    以下の<question>タグ内にはユーザーの入力した質問文が入ります。
    この質問文に基づいて、{n_queries}個の検索用クエリを生成してください。
    各クエリは30トークン以内とし、日本語と英語を適切に混ぜて使用することで、広範囲の文書が取得できるようにしてください。

    生成されたクエリは、<format>タグ内のフォーマットに従って出力してください。

    <example>
    question: Knowledge Bases for Amazon Bedrock ではどのベクトルデータベースを使えますか？
    query_1: Knowledge Bases for Amazon Bedrock vector databases engine DB
    query_2: Amazon Bedrock ナレッジベース ベクトルエンジン vector databases DB
    query_3: Amazon Bedrock RAG 検索拡張生成 埋め込みベクトル データベース エンジン
    </example>

    <format>
    {output_format}
    </format>

    <question>
    {question}
    </question>
    """
)

# 関連度評価用のプロンプト
prompt_post = PromptTemplate.from_template(
    """
    あなたは、ユーザーからの質問と検索で得られたドキュメントの関連度を評価する専門家です。
    <excerpt>タグ内は、検索により取得したドキュメントの抜粋です。

    <excerpt>{context}</excerpt>

    <question>タグ内は、ユーザーからの質問です。

    <question>{question}</question>

    このドキュメントの抜粋は、ユーザーの質問に回答するための正確な情報を含んでいるかを慎重に判断してください。
    正確な情報を含んでいる場合は 'yes'、含んでいない場合は 'no' のバイナリスコアを返してください。

    {format_instructions}
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


chain = prompt_pre | LLM | StrOutputParser()

# chainの実行
answer = chain.invoke(
    {
        "n_queries": 3,
        "output_format": "JSON形式で、各キーには単一のクエリを格納する。",
        "question": user_input,
    }
)

# chainの実行
print(answer)


def main():
    region = "us-east-1"
    conf_llm_path = "./config/config_llm.yaml"
    prompt_template_path = "./config/prompt_template.yaml"
    conf_llm = utils.load_yaml(conf_llm_path)
    prompt_template = utils.load_yaml(prompt_template_path)

    prompt_pre = PromptTemplate.from_template(
        prompt_template["query_expansion"]["template"]
    )

    LLM = ChatBedrock(
        model_id=conf_llm["model_id"],
        region_name=region,
        model_kwargs=conf_llm["query_expansion"]["args"],
    )

    chain = prompt_pre | LLM | StrOutputParser()

    answer = chain.invoke(
        {
            "n_queries": prompt_template["query_expansion"]["n_queries"],
            "output_format": prompt_template["query_expansion"]["output_format"],
            "question": user_input,
        }
    )

    # chainの実行
    print(answer)


if __name__ == "__main__":
    main()
