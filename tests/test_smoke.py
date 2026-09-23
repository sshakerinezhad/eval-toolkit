import llm
import smoke


def _fake_smoke(results):
    async def f(specs, timeout):
        return [results[s] for s in specs]
    return f


def test_smoke_cli_with_models_returns_0_when_all_ok(monkeypatch, capsys):
    monkeypatch.setattr(smoke.llm, "smoke", _fake_smoke({
        "openai:a": llm.SmokeResult("openai:a", True, 0.5, None, "OK"),
    }))
    assert smoke.main(["--model", "openai:a"]) == 0
    assert "OK" in capsys.readouterr().out


def test_smoke_cli_returns_1_and_prints_error_on_failure(monkeypatch, capsys):
    err = llm.Attempt(1, 401, "AuthenticationError", "invalid_api_key", "bad key", False)
    monkeypatch.setattr(smoke.llm, "smoke", _fake_smoke({
        "openai:a": llm.SmokeResult("openai:a", True, 0.5, None, "OK"),
        "anthropic:b": llm.SmokeResult("anthropic:b", False, 0.1, err, ""),
    }))
    assert smoke.main(["--model", "openai:a", "--model", "anthropic:b"]) == 1
    out = capsys.readouterr().out
    assert "401" in out and "bad key" in out and "1/2" in out


def test_smoke_cli_reads_models_from_config(tmp_path, monkeypatch):
    (tmp_path / "t.jsonl").write_text('{"id":1,"prompt":"p"}\n')
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"tasks: {tmp_path / 't.jsonl'}\nmodels: [openai:a, openrouter:x/y]\nsystem_prompt: s\nmax_tokens: 5\n")
    seen = {}

    async def f(specs, timeout):
        seen["specs"] = specs
        return [llm.SmokeResult(s, True, 0.1, None, "OK") for s in specs]

    monkeypatch.setattr(smoke.llm, "smoke", f)
    assert smoke.main([str(cfg)]) == 0
    assert seen["specs"] == ["openai:a", "openrouter:x/y"]
