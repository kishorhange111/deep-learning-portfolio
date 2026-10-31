# How much do frontier LLMs lose in Marathi?

Marathi has over 80 million native speakers, but evaluations rarely ask how well frontier models serve them. This eval measures it from several angles, across **Claude, OpenAI, Gemini, Grok and open-weight models** behind one interface:
- whether models understand Marathi as well as English;
- whether they reply in Marathi at all;
- whether that holds when Marathi is typed in Latin script;
- whether the LLM judges used to score them can be trusted.

**Status:** the whole pipeline is built and tested: 24 tests, mock models on fixture data, and request/response round trips through the real Anthropic and OpenAI SDKs over a mock transport. **No API runs have been made yet**, because they need your API keys (rough costs below). There are no model results in this README until `report.py` produces them.

## Tracks
| Track | Data | What it measures |
|---|---|---|
| **1. Cross-lingual reading comprehension** | [Belebele](https://huggingface.co/datasets/facebook/belebele): 900 multiple-choice questions, professionally translated, so *identical* in English and Marathi | accuracy in **English vs. Marathi vs. Marathi in Latin script** on the same items; the paired English−Marathi gap with a 95 % CI; whether the model gives the same answer in both languages; and whether it gives the same answer when options are reversed (position bias) |
| **2. Open-ended answers** | 200 human-written Marathi prompts from the [Aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset), sent as-is with no system prompt | **language fidelity**: does the model answer in Marathi, or drift into Hindi or English? Measured by a transparent automatic check ([`langid_mr.py`](langid_mr.py)): script shares plus Marathi-vs-Hindi function words and ळ |
| **3. LLM-as-judge, validated** | the answers from track 2 | helpfulness and Marathi fluency (1–5) from **two judges from different providers**, blind to which model wrote the answer, with structured JSON output. Then **agreement with human labels** ([`label.py`](label.py)) via Spearman and quadratic-weighted κ, so you know whether to trust the judge before trusting its scores |
| **4. Memorisation probe** | the Belebele / FLORES passages themselves | given the first half of a passage, can the model continue it verbatim? Measured as character 5-gram overlap vs. a mismatched-passage baseline. Public benchmarks may be in training data, and this checks for it. |

## Design decisions
- **Parallel items isolate the language effect.** Question difficulty is identical across conditions, so the English−Marathi difference is a paired comparison over the same 900 questions: a paired bootstrap, not two independent accuracies.
- **Romanized Marathi** is how many people actually type. It is produced by a rule-based informal romanizer (ITRANS, lower-cased, word-final schwa deletion: शाळेत → *shalet*, मित्र → *mitra*). It approximates real typing, which varies more (z/j, w/v, English loanwords).
- **Reversed options** are a cheap robustness check. A model that understands the question picks the same answer whatever position it's in.
- **The token tax:** the report shows how many times more input tokens the same question costs in Marathi than in English, measured from each API's usage. Tokenizers are often much less efficient on Devanagari, which makes Marathi slower and more expensive to serve.
- **No refusal fallbacks and no silent retries on other models.** Claude's server-side fallback would answer a refused request with a different model, which would corrupt the measurement. Refusals are recorded as refusals.
- **Claude settings are explicit:** `output_config.effort` (default `medium`) is passed and recorded, and sampling parameters are not sent, since current models reject them.
- **Everything is cached** in SQLite, keyed by model, settings and prompt. Reruns are free, interrupted runs resume, and every number can be traced to the exact response behind it.

## Providers
```
anthropic:claude-opus-5-5       ANTHROPIC_API_KEY   (Anthropic SDK)
openai:<model>                  OPENAI_API_KEY      (OpenAI SDK)
gemini:<model>                  GEMINI_API_KEY      (Gemini's OpenAI-compatible endpoint)
xai:<model>                     XAI_API_KEY         (Grok's OpenAI-compatible endpoint)
local:<hf-model>[@url]          any vLLM / SGLang / Ollama server, e.g. Indic models such as Sarvam-M
```
Use each provider's current model IDs from its docs. Claude prices are built in; give other providers' prices with `report.py --prices prices.json` to get cost columns.

## Run it
```bash
pip install anthropic openai datasets indic-transliteration numpy matplotlib
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... XAI_API_KEY=...
M="anthropic:claude-opus-5-5 openai:<model> gemini:<model> xai:<model>"
python run_eval.py mcq --models $M --reverse --n 300     # first pass on 300 questions; drop --n for all 900
python run_eval.py generate --models $M                   # 200 Aya prompts
python run_eval.py judge --judges anthropic:claude-opus-5-5 openai:<model>
python run_eval.py contamination --models $M
python label.py --n 100                                   # ~1 hour of a fluent reader's time
python report.py                                          # results/main/summary.md + figures
```
**Rough cost per frontier model** (an estimate, not measured):
- full MCQ track with reversal: 5,400 requests, about $20–50 at medium effort;
- generation and judging: a few dollars.

Start with `--n 300`.

## Outputs (`results/<run>/`)
- `summary.md` / `summary.json`, containing:
  - accuracy per condition with CIs, the English−Marathi gap, cross-lingual and order consistency, the token ratio and cost;
  - the language-fidelity table and judge scores;
  - judge–judge, judge–human and automatic-check–human agreement;
  - the memorisation probe.
- Figures: `mcq_accuracy.png` and `language_fidelity.png`.
- `cache.sqlite` plus `mcq_*.jsonl`, `gen_*.jsonl`, `judge_*.jsonl` and `contam_*.jsonl`: every response behind every number.

## Limitations
- **Belebele is reading comprehension of translated text.** It does not cover cultural knowledge or naturally written Marathi; track 2 partly addresses that.
- **The language check is a heuristic.** Its agreement with human labels is reported, so check that number first.
- **LLM judges tend to favour answers from their own model family** (Panickssery et al., 2024). That's why there are two judges from different providers, plus human labels.
- **Low continuation overlap does not prove a model never saw the data.** High overlap is evidence that it did.

## Tests (`pytest tests/test_marathi_eval.py`, 24 tests, ~7 s)
- **Language check:** works on Marathi, Hindi, English and empty text.
- **Parsing and prompts:** answer-letter parsing, the reversed-option mapping and romanization rules.
- **Statistics:** kappa and Spearman match scikit-learn and SciPy.
- **Provider clients:** request building per provider. The **real SDKs send the intended body to the right endpoint** (mock HTTP transport).
- **Cache and cost:** the cache never stores errors, and cost accounting is correct.
- **Full pipeline:** mock models covering all four tracks plus human labels and the report.
  - A mock that only understands English shows a +75-point gap with a tight CI.
  - "Always answer A" scores 0 % order consistency.
  - A memorising mock is flagged by the probe.

## References
Bandarkar et al., *The Belebele Benchmark* (2024) · Singh et al., *Aya Dataset* (2024) · NLLB Team, *FLORES-200* (2022) · Zheng et al., *Judging LLM-as-a-Judge* (2023) · Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations* (2024).
