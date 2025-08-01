"""
Minimalist LangChain implementation for OpenAI with log probabilities.
"""

import yaml
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def load_config() -> Dict[str, Any]:
    """Load configuration from keys.yaml"""
    with open("2_train_eval\\keys.yaml") as f:
        return yaml.safe_load(f)


def get_model(api_key: str, model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """
    Initialize ChatOpenAI (logprobs bound at call-time, per official example).

    Args:
        api_key: OpenAI API key
        model_name: Model identifier

    Returns:
        ChatOpenAI instance (without logprobs; bind them on the runnable)
    """
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.7)


def main():
    """Main execution flow"""
    # Load configuration
    config = load_config()

    # Initialize model and bind logprobs like the official example
    llm = get_model(config["OPENAI_API_KEY"]).bind(logprobs=True, top_logprobs=5)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Respond briefly, only the bare minimum"),
            ("human", "{input}"),
        ]
    )

    # Create chain (do NOT parse to string so we keep response_metadata)
    chain = prompt | llm

    # Run inference
    msg = chain.invoke({"input": "What is 1+1?"})

    # Access log probabilities exactly like the official example
    content_logprobs = msg.response_metadata["logprobs"]["content"]
    print("\nFirst 5 token logprobs:")
    for entry in content_logprobs[:5]:
        token = entry.get("token")
        logprob = entry.get("logprob")
        top = entry.get("top_logprobs", [])
        print({"token": token, "logprob": logprob, "top_logprobs": top})

    print("\nResponse:", msg.content)


if __name__ == "__main__":
    main()
