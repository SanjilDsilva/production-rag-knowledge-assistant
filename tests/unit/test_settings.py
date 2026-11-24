from production_rag.settings import load_llm_config


def test_load_llm_config_reads_models_and_limits():
    config = load_llm_config()

    assert config["models"]["primary"] == "gemini-3.1-flash-lite"
    assert config["models"]["backup"] == "openai/gpt-oss-20b"
    assert config["limits"]["context_tokens"] == 3000
    assert config["limits"]["retrieval_top_k"] == 10
    assert config["limits"]["citation_labels_follow_prompt_order"] is True
    assert config["limits"]["max_retry_wait_seconds"] == 10
    assert config["limits"]["output_tokens"] == 1000
    assert config["limits"]["user_questions_per_minute"] == 5
    assert config["limits"]["global_questions_per_minute"] == 10
    assert config["limits"]["global_questions_per_day"] == 100
