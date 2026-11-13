from groq import Groq
from production_rag.settings import load_llm_config



def generate_text(
    client: Groq,
    messages: list[dict[str, str]],
    model_name: str,
) -> str:
    if not model_name.strip():
        raise ValueError("Model name cannot be empty")

    config = load_llm_config()
    max_output_tokens = config["limits"]["output_tokens"]

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0.0,
        max_tokens=max_output_tokens,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no answer choices")

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer")

    return answer.strip()
