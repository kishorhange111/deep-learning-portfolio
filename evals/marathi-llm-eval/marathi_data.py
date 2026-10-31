"""
Eval data: Belebele reading comprehension in parallel English / Marathi, and human-written Marathi
prompts from the Aya dataset.

Belebele (Bandarkar et al., 2024; facebook/belebele, CC BY-SA 4.0) has 900 multiple-choice
questions on FLORES-200 passages, professionally translated so the *same* question exists in 122
languages. Because the items are parallel, the English–Marathi accuracy difference measures what a
model loses in Marathi, with question difficulty held fixed.

Conditions built from each item:
  en              English passage, question and options
  mr              Marathi (Devanagari)
  mr_roman        Marathi in Latin script, informal style (rule-based approximation, see below)
Each condition can also be asked with the options in reversed order, which tests position bias:
a model that understood the question picks the same answer either way.
"""

import json
import random
import re

LETTERS = "ABCD"

INSTRUCTION = {
    "en": "Read the passage and answer the question. Reply with only the letter (A, B, C or D) of the correct option.",
}


def load_belebele(lang, data_dir=None):
    """Belebele items for one FLORES language code (eng_Latn, mar_Deva) as plain dicts.

    data_dir: optional folder with <lang>.jsonl files in Belebele's schema (offline use, tests)."""
    if data_dir:
        with open(f"{data_dir}/{lang}.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        rows = load_dataset("facebook/belebele", lang, split="test")
    return [
        {
            "id": f"{r['link']}#{r['question_number']}",
            "passage": r["flores_passage"],
            "question": r["question"],
            "options": [r[f"mc_answer{i}"] for i in range(1, 5)],
            "answer": int(r["correct_answer_num"]) - 1,
        }
        for r in rows
    ]


def informal_romanize(text):
    """Devanagari Marathi -> Latin script roughly as people type it in chats.

    ITRANS transliteration, lower-cased, with the inherent vowel dropped at the end of words
    ("शाळेत" -> ITRANS "shALeta" -> "shalet"). Real typing varies far more (z/j, w/v,
    English loanwords), so this tests the script, not the full mess of user-typed text."""
    from indic_transliteration import sanscript

    itrans = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    itrans = itrans.replace("~N", "n").replace("~n", "n").replace(".N", "n").replace(".n", "n").replace(".h", "")
    lower = itrans.lower().replace("aa", "a").replace("ii", "i").replace("uu", "u")
    # word-final schwa deletion after a single consonant (शाळेत -> shalet, घर -> ghar), but not after a
    # cluster, where Marathi keeps it (मित्र -> mitra)
    c = "[bcdfgjklmnpqrstvwxyz]"
    return re.sub(rf"(?:(?<=[aeiou]{c})|(?<=[aeiou](?:sh|ch|th|kh|gh|jh|dh|ph|bh)))a\b", "", lower)


def build_conditions(en_items, mr_items, romanize=True):
    """Pair English and Marathi items by id and return {condition: [items]}."""
    by_id = {it["id"]: it for it in mr_items}
    pairs = [(e, by_id[e["id"]]) for e in en_items if e["id"] in by_id]
    conds = {"en": [e for e, _ in pairs], "mr": [m for _, m in pairs]}
    if romanize:
        conds["mr_roman"] = [
            {
                **m,
                "passage": informal_romanize(m["passage"]),
                "question": informal_romanize(m["question"]),
                "options": [informal_romanize(o) for o in m["options"]],
            }
            for _, m in pairs
        ]
    return conds


def mcq_prompt(item, reverse=False):
    """(system, user, mapping) where mapping[letter_index] = original option index."""
    order = list(range(4))[::-1] if reverse else list(range(4))
    options = "\n".join(f"{LETTERS[i]}. {item['options'][j]}" for i, j in enumerate(order))
    user = f"Passage:\n{item['passage']}\n\nQuestion: {item['question']}\n\n{options}\n\nAnswer:"
    return INSTRUCTION["en"], user, order


def parse_letter(text):
    """First standalone A-D in the reply (after stripping markdown), else None."""
    if not text:
        return None
    cleaned = re.sub(r"[*_`#]", "", text).strip()
    m = re.match(r"^\(?([ABCD])[\).:\s]|^\(?([ABCD])\)?$", cleaned)
    if m:
        return LETTERS.index(m.group(1) or m.group(2))
    m = re.search(r"\b(?:answer|option|उत्तर)\s*(?:is|:)?\s*\(?([ABCD])\b", cleaned, re.IGNORECASE)
    if m:
        return LETTERS.index(m.group(1).upper())
    m = re.search(r"\b([ABCD])\b", cleaned)
    return LETTERS.index(m.group(1)) if m else None


def load_aya_marathi(n, seed=0, data_file=None):
    """n human-written Marathi prompt/response pairs from CohereForAI/aya_dataset (Apache-2.0)."""
    if data_file:
        with open(data_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        rows = [r for r in load_dataset("CohereForAI/aya_dataset", split="train") if r["language"] == "Marathi"]
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    return [{"id": f"aya-{i}", "prompt": r["inputs"], "reference": r["targets"]} for i, r in enumerate(rows)]
