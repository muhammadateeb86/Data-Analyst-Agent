from src.app.streamlit_app import create_agent, load_project_env
from src.agent import DataAnalystAgent


def test_streamlit_factory_builds_real_agent(monkeypatch):
    class FakeProvider:
        def __call__(self, prompt):
            return "{}"

    monkeypatch.setattr("src.app.streamlit_app.ChatCompletionProvider.from_env", lambda: FakeProvider())
    assert isinstance(create_agent(), DataAnalystAgent)


def test_load_project_env_does_not_override_existing_values(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=file-key\nLLM_PROVIDER=groq\n")
    monkeypatch.setenv("GROQ_API_KEY", "shell-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    load_project_env(env_file)
    assert __import__("os").environ["GROQ_API_KEY"] == "shell-key"
    assert __import__("os").environ["LLM_PROVIDER"] == "groq"
