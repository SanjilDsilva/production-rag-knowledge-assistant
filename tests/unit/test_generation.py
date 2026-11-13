from types import SimpleNamespace

from production_rag.generation import generate_text


class FakeGroqClient:
    def __init__(self):
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" Grounded answer [chunk-1] ")
                )
            ]
        )


def test_generate_text_sends_messages_and_returns_answer():
    client = FakeGroqClient()
    messages = [{"role": "user", "content": "Question"}]

    answer = generate_text(client, messages, model_name="test_model")

    assert answer == "Grounded answer [chunk-1]"
    assert client.request["model"] == "test_model"
    assert client.request["messages"] == messages
    assert client.request["max_tokens"] == 512
