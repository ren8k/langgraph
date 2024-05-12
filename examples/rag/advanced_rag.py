import json
from typing import Any

from langchain.globals import set_debug

set_debug(False)  # debug時はTrue

import utils
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# ユーザー入力
question = "Amazonの生成AIの分野での注力範囲は？"


def create_prompt(template: str, kwargs: dict, question: str):
    template = PromptTemplate.from_template(template)
    return template.format(**kwargs, question=question)


def create_chat_template(include_ai_message=False):
    messages = [
        SystemMessage(content="{system_message}"),
        HumanMessagePromptTemplate.from_template("{prompt}"),
    ]
    if include_ai_message:
        messages.append(AIMessagePromptTemplate.from_template("{assistant_message}"))
    return ChatPromptTemplate.from_messages(messages)


def main():
    region = "us-east-1"
    conf_llm_path = "./config/config_llm.yaml"
    prompt_template_path = "./config/prompt_template.yaml"
    question_path = "./config/question.yaml"
    conf_llm = utils.load_yaml(conf_llm_path)
    conf_prompt_template = utils.load_yaml(prompt_template_path)
    question = utils.load_yaml(question_path)

    # step1. Expand query
    prompt = create_prompt(
        conf_prompt_template["query_expansion"]["template"],
        conf_prompt_template["query_expansion"]["args"],
        question,
    )
    chat_template = create_chat_template(include_ai_message=True)

    LLM = ChatBedrock(
        model_id=conf_llm["query_expansion"]["model_id"],
        region_name=region,
        model_kwargs=conf_llm["query_expansion"]["args"],
    )

    chain = chat_template | LLM | StrOutputParser()

    answer = chain.invoke(
        {
            "system_message": conf_llm["query_expansion"]["system_message"],
            "prompt": prompt,
            "assistant_message": conf_llm["query_expansion"]["assistant_message"],
        }
    )
    queries_expanded: str = "{" + answer
    queries_expanded_dict = json.loads(queries_expanded)
    queries_expanded_dict["query_0"] = question
    print(queries_expanded_dict)


if __name__ == "__main__":
    main()
