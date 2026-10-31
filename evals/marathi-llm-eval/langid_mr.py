"""
Did the model answer in Marathi? A transparent, rule-based language check.

Marathi and Hindi share the Devanagari script, so counting script characters is not enough. On top of
script shares, this counts frequent function words that differ between the two languages, plus
the letter ळ (common in Marathi, absent from standard Hindi). It is deliberately simple and
inspectable. Its accuracy should be checked against the human labels collected with label.py
(report.py prints the agreement) before its numbers are trusted.

Labels: "marathi", "hindi", "english", "mixed", "empty".
"""

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[ऀ-ॿ]+")

# Frequent words that are distinctive for each language (verb "to be", negation, conjunctions,
# postpositions, pronouns). Words shared by both languages are left out.
MARATHI_WORDS = {
    "आहे",
    "आहेत",
    "आहोत",
    "नाही",
    "नाहीत",
    "आणि",
    "म्हणजे",
    "मध्ये",
    "होते",
    "होता",
    "होती",
    "असे",
    "असते",
    "करणे",
    "त्यामुळे",
    "पण",
    "किंवा",
    "तुम्ही",
    "आम्ही",
    "मी",
    "माझे",
    "माझा",
    "माझी",
    "त्याचे",
    "त्याची",
    "त्यांच्या",
    "हे",
    "ही",
    "हा",
    "तो",
    "ती",
    "ते",
    "काय",
    "कसे",
    "केले",
    "करा",
    "आपण",
    "आपल्या",
    "येथे",
    "तसेच",
}
HINDI_WORDS = {
    "है",
    "हैं",
    "हूँ",
    "हूं",
    "नहीं",
    "और",
    "में",
    "के",
    "की",
    "का",
    "था",
    "थी",
    "थे",
    "यह",
    "वह",
    "क्योंकि",
    "लेकिन",
    "आप",
    "हम",
    "मैं",
    "मेरा",
    "मेरी",
    "उसका",
    "उसकी",
    "उनके",
    "क्या",
    "कैसे",
    "किया",
    "करें",
    "जाता",
    "जाती",
    "होता",
    "होती",
    "भी",
    "तो",
    "से",
    "पर",
    "को",
    "अपने",
    "यहाँ",
    "तथा",
}
# Words that occur in both languages ("तो": Marathi pronoun / Hindi particle; "ही": Marathi pronoun /
# Hindi emphatic particle; forms of "होणे/होना") would cancel out or mislead, so they are excluded.
for _shared in ("तो", "ही", "होते", "होता", "होती"):
    MARATHI_WORDS.discard(_shared)
    HINDI_WORDS.discard(_shared)


def script_shares(text):
    """Fractions of letters that are Devanagari and Latin."""
    dev, lat = len(_DEVANAGARI.findall(text)), len(_LATIN.findall(text))
    total = dev + lat
    return (dev / total, lat / total) if total else (0.0, 0.0)


def marathi_score(text):
    """In [0, 1]: share of Marathi evidence among Marathi + Hindi evidence (None if there is none)."""
    words = _WORD.findall(text)
    mr = sum(w in MARATHI_WORDS for w in words) + text.count("ळ")
    hi = sum(w in HINDI_WORDS for w in words)
    return mr / (mr + hi) if mr + hi else None


def classify(text, script_threshold=0.6, lang_threshold=0.6):
    """Return (label, details)."""
    dev, lat = script_shares(text)
    score = marathi_score(text)
    details = {"devanagari_share": dev, "latin_share": lat, "marathi_score": score}
    if dev == 0 and lat == 0:
        return "empty", details
    if lat >= script_threshold:
        return "english", details
    if dev >= script_threshold:
        if score is None:
            return "mixed", details
        if score >= lang_threshold:
            return "marathi", details
        if score <= 1 - lang_threshold:
            return "hindi", details
    return "mixed", details
