"""Tests for evals/marathi-llm-eval: language check, answer parsing, romanization, statistics, the two
provider clients (request building / response parsing, no network) and the full pipeline with mock
models on small fixture data."""

import json
import types

import pytest
from conftest import load_module

PROJECT = "evals/marathi-llm-eval"
langid = load_module(PROJECT, "langid_mr.py")
data = load_module(PROJECT, "marathi_data.py")
st = load_module(PROJECT, "eval_stats.py")
providers = load_module(PROJECT, "providers.py")


# ---------------------------------------------------------------- language check
@pytest.mark.parametrize(
    "text,label",
    [
        ("मी शाळेत जातो आणि माझे मित्र तिथे आहेत.", "marathi"),
        ("महाराष्ट्राची राजधानी मुंबई आहे. ते खूप मोठे शहर आहे.", "marathi"),
        ("मैं स्कूल जाता हूँ और मेरे दोस्त वहाँ हैं।", "hindi"),
        ("भारत की राजधानी नई दिल्ली है।", "hindi"),
        ("The capital of Maharashtra is Mumbai.", "english"),
        ("", "empty"),
    ],
)
def test_language_check(text, label):
    assert langid.classify(text)[0] == label


# ---------------------------------------------------------------- data helpers
@pytest.mark.parametrize(
    "reply,index",
    [
        ("B", 1),
        ("**C**", 2),
        ("(D) because", 3),
        ("The answer is A.", 0),
        ("उत्तर: C", 2),
        ("I'd pick option b", 1),
        ("no idea", None),
    ],
)
def test_parse_letter(reply, index):
    assert data.parse_letter(reply) == index


def test_reversed_options_map_back_to_original_indices():
    item = {"passage": "p", "question": "q", "options": ["w", "x", "y", "z"], "answer": 0}
    _, user, order = data.mcq_prompt(item, reverse=True)
    assert "A. z" in user and "D. w" in user and order[data.parse_letter("D")] == 0


def test_informal_romanization():
    pytest.importorskip("indic_transliteration")
    assert data.informal_romanize("माझे घर पुण्यात आहे.") == "majhe ghar punyat ahe."
    assert data.informal_romanize("मित्र") == "mitra"  # schwa kept after a consonant cluster


# ---------------------------------------------------------------- statistics
def test_kappa_and_spearman():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert st.quadratic_weighted_kappa(a, a) == pytest.approx(1.0)
    assert st.spearman(a, a) == pytest.approx(1.0)
    b = [2, 2, 3, 5, 4, 3, 1]
    sk = pytest.importorskip("sklearn.metrics")
    assert st.quadratic_weighted_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b, weights="quadratic"))
    scipy_stats = pytest.importorskip("scipy.stats")
    assert st.spearman(a, b) == pytest.approx(scipy_stats.spearmanr(a, b).statistic)


def test_paired_bootstrap_and_ci():
    import numpy as np

    rng = np.random.default_rng(0)
    b = (rng.random(900) < 0.6).astype(int)
    a = b.copy()
    a[np.where(b == 0)[0][:45]] = 1  # +5 points
    res = st.paired_bootstrap(a, b)
    assert res["diff"] == pytest.approx(0.05) and res["ci_low"] > 0 and res["p"] < 0.01
    mean, lo, hi = st.bootstrap_ci(b)
    assert lo < mean < hi


# ---------------------------------------------------------------- provider clients (no network)
def test_anthropic_request_and_response(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = load_module(PROJECT, "anthropic_client.py").AnthropicClient("claude-opus-5-5", effort="low")
    req = client.build_request("sys", "प्रश्न", 1000, json_schema={"type": "object"})
    assert req["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert req["system"] == "sys" and "temperature" not in req and "fallbacks" not in req
    usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)
    blocks = [types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text="B")]
    r = client.parse_response(types.SimpleNamespace(content=blocks, usage=usage, stop_reason="end_turn"), 0.1)
    assert r.text == "B" and not r.refused and r.output_tokens == 5
    refusal = client.parse_response(types.SimpleNamespace(content=[], usage=usage, stop_reason="refusal"), 0.1)
    assert refusal.refused and refusal.text == ""


def test_openai_compatible_requests(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod = load_module(PROJECT, "openai_compat_client.py")
    oa = mod.OpenAICompatClient("openai", "some-model").build_request(None, "hi", 100)
    assert oa["max_completion_tokens"] == 100 and oa["messages"] == [{"role": "user", "content": "hi"}]
    gem = mod.OpenAICompatClient("gemini", "g", temperature=0.0)
    req = gem.build_request("s", "hi", 50, json_schema={"type": "object"})
    assert req["max_tokens"] == 50 and req["temperature"] == 0.0 and req["response_format"]["type"] == "json_schema"
    assert "generativelanguage" in str(gem.client.base_url)
    msg = types.SimpleNamespace(content="A", refusal=None)
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
        usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )
    assert gem.parse_response(resp, 0.2).text == "A"


def test_cache_avoids_repeat_calls_and_skips_errors(tmp_path):
    calls = []

    class Fake:
        def complete(self, system, user, max_tokens=None, json_schema=None):
            calls.append(user)
            return providers.LLMResponse(text="ok", error="boom" if user == "bad" else None)

    c = providers.CachedClient("mock:fake", Fake(), str(tmp_path / "c.sqlite"))
    assert not c.complete("s", "u").cached and c.complete("s", "u").cached
    c.complete("s", "bad")
    c.complete("s", "bad")
    assert calls == ["u", "bad", "bad"]


def test_cost():
    assert providers.cost_usd("anthropic:claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(4.0 + 2.0)
    assert providers.cost_usd("xai:unknown", 10, 10) is None


# ---------------------------------------------------------------- end to end with mock models
def write_belebele_fixture(path, n=40):
    """Parallel fixture items: the correct option is the one saying 'correct' / 'बरोबर'."""
    path.mkdir()
    import random

    words = {
        "eng_Latn": "river valley mountain city market school train garden temple festival harvest monsoon".split(),
        "mar_Deva": "नदी दरी डोंगर शहर बाजार शाळा गाडी बाग मंदिर सण पीक पाऊस".split(),
    }
    for lang, good, bad in [("eng_Latn", "correct", "wrong"), ("mar_Deva", "बरोबर", "चूक")]:
        rng = random.Random(0)
        with open(path / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                passage = " ".join(rng.choice(words[lang]) for _ in range(30))  # distinct passages per item
                ans = i % 4
                opts = {f"mc_answer{j + 1}": (good if j == ans else f"{bad} {j}") for j in range(4)}
                f.write(
                    json.dumps(
                        {
                            "link": "x",
                            "question_number": i,
                            "flores_passage": passage,
                            "question": f"Q{i}?",
                            "correct_answer_num": str(ans + 1),
                            **opts,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def oracle_english(system, user):
    """Finds the option containing 'correct' (so it is perfect in English and guesses 'A' in Marathi)."""
    for line in user.splitlines():
        if line[:2] in ("A.", "B.", "C.", "D.") and "correct" in line:
            return line[0]
    return "A"


def test_full_pipeline_with_mock_models(tmp_path):
    pytest.importorskip("indic_transliteration")
    providers_mod = load_module(PROJECT, "providers.py")
    run_eval = load_module(PROJECT, "run_eval.py")
    report = load_module(PROJECT, "report.py")
    label = load_module(PROJECT, "label.py")
    # run_eval/report import "providers" by plain name; register mocks on that module too
    import providers as plain_providers

    for mod in (providers_mod, plain_providers):
        mod.MOCKS["oracle-en"] = oracle_english
        mod.MOCKS["marathi-writer"] = lambda s, u: "महाराष्ट्राची राजधानी मुंबई आहे."
        mod.MOCKS["hindi-writer"] = lambda s, u: "भारत की राजधानी नई दिल्ली है।"
    bel = tmp_path / "belebele"
    write_belebele_fixture(bel)
    known = [
        json.loads(line)["flores_passage"] for lang in ("eng_Latn", "mar_Deva") for line in open(bel / f"{lang}.jsonl")
    ]

    def memorizer(system, user):
        """Returns the true continuation of any fixture passage (a perfectly contaminated model)."""
        return next((p[len(user) :].strip() for p in known if p.startswith(user)), "")

    for mod in (providers_mod, plain_providers):
        mod.MOCKS["memorizer"] = memorizer
    aya = tmp_path / "aya.jsonl"
    aya.write_text(
        "\n".join(
            json.dumps({"inputs": f"प्रश्न {i}", "targets": "उत्तर", "language": "Marathi"}, ensure_ascii=False)
            for i in range(12)
        ),
        encoding="utf-8",
    )
    common = ["--out_dir", str(tmp_path / "results"), "--run", "t", "--workers", "2"]
    run_eval.main(["mcq", "--models", "mock:oracle-en", "mock:always-a", "--data_dir", str(bel), "--reverse", *common])
    run_eval.main(["generate", "--models", "mock:marathi-writer", "mock:hindi-writer", "--aya_file", str(aya), *common])
    run_eval.main(["judge", "--judges", "mock:judge-3", *common])
    run_eval.main(["contamination", "--models", "mock:memorizer", "mock:always-a", "--data_dir", str(bel), *common])
    run_dir = tmp_path / "results" / "t"
    answers = iter(["4", "3", "m"] * 50)
    assert label.label(str(run_dir), 6, read=lambda prompt: next(answers), show=lambda *a: None) == 6

    summary = report.report(str(run_dir))
    oracle = summary["mcq"]["mock:oracle-en"]
    assert oracle["conditions"]["en"]["accuracy"] == 1.0
    assert oracle["conditions"]["mr"]["accuracy"] == pytest.approx(0.25)  # always 'A' -> 1 in 4 correct
    assert oracle["gap_en_minus_mr"]["diff"] == pytest.approx(0.75) and oracle["gap_en_minus_mr"]["ci_low"] > 0.5
    assert (
        summary["mcq"]["mock:always-a"]["conditions"]["mr"]["order_consistency"] == 0.0
    )  # 'A' flips meaning when reversed
    assert summary["generation"]["language"]["mock:marathi-writer"]["marathi"] == 1.0
    assert summary["generation"]["language"]["mock:hindi-writer"]["hindi"] == 1.0
    assert "mock:judge-3 vs human" in summary["agreement"]
    assert summary["agreement"]["language check vs human"]["n"] == 6
    contam = summary["contamination"]
    assert contam["mock:memorizer"]["mar_Deva"]["overlap"] == pytest.approx(1.0)
    assert contam["mock:memorizer"]["eng_Latn"]["excess"]["ci_low"] > 0.1  # well above the other-passage baseline
    assert contam["mock:always-a"]["eng_Latn"]["verbatim_share"] == 0.0
    for f in ("summary.md", "mcq_accuracy.png", "language_fidelity.png"):
        assert (run_dir / f).exists()


def _mock_transport(sdk, payload, captured):
    """A transport for the SDK's own HTTP library (httpx for older SDKs, httpx2 for anthropic>=1) that records
    the request and returns payload."""
    httpx = pytest.importorskip(type(sdk.DefaultHttpxClient()).__mro__[1].__module__.split(".")[0])

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_anthropic_sdk_sends_the_intended_request():
    """Through the real SDK (mock transport): body as built, thinking blocks skipped when parsing."""
    anthropic = pytest.importorskip("anthropic")
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "C"}],
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 34},
    }
    captured = {}
    transport = _mock_transport(anthropic, payload, captured)
    mod = load_module(PROJECT, "anthropic_client.py")
    client = mod.AnthropicClient("claude-opus-5-5", effort="low")
    client.client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(transport=transport))
    r = client.complete("sys", "Q?", max_tokens=2000)
    assert captured["body"]["output_config"] == {"effort": "low"} and "temperature" not in captured["body"]
    assert captured["url"].endswith("/v1/messages") and r.text == "C" and r.output_tokens == 34


def test_openai_sdk_sends_gemini_request_to_compat_endpoint(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    payload = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "g",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "मुंबई"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    captured = {}
    transport = _mock_transport(openai, payload, captured)
    mod = load_module(PROJECT, "openai_compat_client.py")
    client = mod.OpenAICompatClient("gemini", "g")
    client.client = openai.OpenAI(
        api_key="k", base_url=mod.ENDPOINTS["gemini"][0], http_client=openai.DefaultHttpxClient(transport=transport)
    )
    r = client.complete(None, "राजधानी?", max_tokens=64)
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert captured["body"]["max_tokens"] == 64 and r.text == "मुंबई" and r.input_tokens == 5
