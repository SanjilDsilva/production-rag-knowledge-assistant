from groq import Groq


def generate_text(
    client: Groq,
    messages: list[dict[str, str]],
    model_name: str,
) -> str:
    if not model_name.strip():
        raise ValueError("Model name cannot be empty")

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0.0,
        max_tokens=512,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no answer choices")

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer")

    return answer.strip()
