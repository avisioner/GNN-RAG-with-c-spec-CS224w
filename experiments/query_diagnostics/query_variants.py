"""Fixed input-only query transformations; no labels or corpus access."""
import re

VARIANTS = ("question_only", "question_code", "code_only", "labeled_code", "code_first",
            "explicit_one_sentence", "verbose_explicit", "operator_gloss", "question_domain")
CONTROLS = {"question_code": "question_only", "code_only": "question_code",
            "labeled_code": "question_code", "code_first": "labeled_code",
            "explicit_one_sentence": "question_code", "verbose_explicit": "explicit_one_sentence",
            "operator_gloss": "question_code", "question_domain": "question_only"}
GLOSSES = {"++": "increment", "--": "decrement", "&&": "logical AND", "||": "logical OR",
           "<<": "left shift spelling", ">>": "right shift spelling", "=": "assignment spelling",
           "+=": "addition assignment", "-=": "subtraction assignment", "?": "conditional question-mark token",
           "[": "square bracket token", "->": "member access arrow", "+": "plus token"}
CODE_TOKENS = re.compile(r"<<=|>>=|\+\+|--|&&|\|\||<<|>>|\+=|-=|==|!=|<=|>=|->|[=?\[+]")


def operator_gloss(code: str) -> str:
    # Token spellings only: no inference about overload resolution or behavior.
    tokens = dict.fromkeys(CODE_TOKENS.findall(code))
    return "; ".join(f"{t}: {GLOSSES[t]}" for t in tokens if t in GLOSSES)


def build_variants(question: str, code: str, explicit_question: str) -> dict[str, str]:
    if not question.strip() or not code.strip():
        raise ValueError("Paired diagnostics require both original question and code")
    if "\n" in explicit_question or not explicit_question.endswith("?"):
        raise ValueError("Explicit rewrite must be a single question")
    raw = question + "\n" + code
    explicit = explicit_question + "\n" + code
    gloss = operator_gloss(code)
    return {
        "question_only": question,
        "question_code": raw,
        "code_only": code,
        "labeled_code": f"Question: {question}\nC++ code:\n```cpp\n{code}\n```",
        "code_first": f"C++ code:\n```cpp\n{code}\n```\nQuestion: {question}",
        "explicit_one_sentence": explicit,
        "verbose_explicit": ("Please assess the following C++23 example using the language specification.\n"
                             + explicit_question + "\nFocus on the supplied program and the applicable language rules.\n" + code),
        "operator_gloss": raw + ("\nOperator spellings in the code: " + gloss if gloss else ""),
        "question_domain": "C++23 specification: " + question,
    }
