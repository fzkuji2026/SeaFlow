import os

from glee_agent.keys import load_all


def test_load_all_uses_selected_env_without_overwriting_explicit_values(
    monkeypatch, tmp_path
):
    env_file = tmp_path / ".env.local"
    env_file.write_text("GLEE_API_KEY=test-key\nGLEE_CONCURRENCY=6\n")
    monkeypatch.setenv("GLEE_ENV_FILE", str(env_file))
    monkeypatch.setenv("GLEE_CONCURRENCY", "16")
    monkeypatch.delenv("GLEE_API_KEY", raising=False)

    load_all()

    assert os.environ["GLEE_API_KEY"] == "test-key"
    assert os.environ["GLEE_CONCURRENCY"] == "16"
