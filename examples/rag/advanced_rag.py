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


def main():
    region = "us-east-1"
    conf_llm_path = "./config/config_llm.yaml"
    prompt_template_path = "./config/prompt_template.yaml"
    conf_llm = utils.load_yaml(conf_llm_path)
    conf_prompt_template = utils.load_yaml(prompt_template_path)

    prompt_template = PromptTemplate.from_template(
        conf_prompt_template["query_expansion"]["template"]
    )

    kwargs = {
        "n_queries": conf_prompt_template["query_expansion"]["n_queries"],
        "output_format": conf_prompt_template["query_expansion"]["output_format"],
        "question": question,
    }
    prompt = prompt_template.format(**kwargs)

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="{system_message}"),
            HumanMessagePromptTemplate.from_template("{prompt}"),
            AIMessagePromptTemplate.from_template("{assistant_message}"),
        ]
    )

    LLM = ChatBedrock(
        model_id=conf_llm["model_id"],
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

    # chainの実行
    print("{" + answer)


if __name__ == "__main__":
    main()
