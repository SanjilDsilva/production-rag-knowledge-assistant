from production_rag.settings import load_llm_config


def test_load_llm_config_reads_models_and_limits():
    config = load_llm_config()

    assert config["models"]["primary"] == "gemini-3.6-flash"
    assert config["models"]["backup"] == "openai/gpt-oss-20b"
    assert config["limits"]["context_tokens"] == 3000
    assert config["limits"]["output_tokens"] == 512
