"""Versioned input-only query expansion, frozen to the diagnostic condition.

This is a spelling glossary, not a C++ parser. Tokens in comments/literals and
ambiguous spellings can be included; no semantic or overload claims are made.
"""
import re

TRANSFORM_VERSION = "operator_gloss_v1"
GLOSSES = {"++": "increment", "--": "decrement", "&&": "logical AND", "||": "logical OR",
           "<<": "left shift spelling", ">>": "right shift spelling", "=": "assignment spelling",
           "+=": "addition assignment", "-=": "subtraction assignment", "?": "conditional question-mark token",
           "[": "square bracket token", "->": "member access arrow", "+": "plus token"}
CODE_TOKENS = re.compile(r"<<=|>>=|\+\+|--|&&|\|\||<<|>>|\+=|-=|==|!=|<=|>=|->|[=?\[+]")


def operator_gloss(code: str) -> str:
    tokens = dict.fromkeys(CODE_TOKENS.findall(code))
    return "; ".join(f"{token}: {GLOSSES[token]}" for token in tokens if token in GLOSSES)


def transform_query(text: str, code: str, transform: str = "none") -> str:
    if transform == "none":
        return text
    if transform != "operator_gloss":
        raise ValueError(f"Unknown query transform: {transform}")
    gloss = operator_gloss(code)
    return text + ("\nOperator spellings in the code: " + gloss if gloss else "")
