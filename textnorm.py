"""Нормализация текста и формул для поиска.

Вопросы пишут формулы в LaTeX ($\\partial f/\\partial\\bar z$), корпус — в Unicode
(\\frac{∂f}{∂\\bar{z}}). Приводим оба к общему виду, чтобы совпадали токены.
"""
import re

import snowballstemmer

_stem = snowballstemmer.stemmer("russian").stemWord

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "varepsilon": "ε", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "tau": "τ", "varphi": "φ", "phi": "φ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "∆", "Omega": "Ω",
    "partial": "∂", "infty": "∞", "int": "∫", "oint": "∮", "sum": "∑", "prod": "∏",
    "le": "⩽", "leq": "⩽", "ge": "⩾", "geq": "⩾", "ne": "≠", "neq": "≠", "to": "→",
    "in": "∈", "subset": "⊂", "setminus": "\\", "equiv": "≡", "cdot": "·", "triangle": "△",
}
SYMBOLS = "∂∫∮∑∏∞πζγφαβδεξηλσωψ∆△≡"


def normalize(text: str) -> str:
    """Единый вид формул: \\bar z -> zbar, \\partial -> ∂, убрать $, \\frac{a}{b} -> a / b."""
    t = text.replace("$", " ")
    t = re.sub(r"\\(?:bar|overline)\s*\{?\s*([A-Za-zα-ω])\s*\}?", r" \1bar ", t)
    t = re.sub(r"\\(?:mathbb|mathrm|operatorname)\s*\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\(?:tfrac|dfrac|frac)", " frac ", t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: f" {GREEK.get(m.group(1), m.group(1))} ", t)
    t = t.replace("ё", "е").replace("Ё", "Е")
    return t


TOKEN_RE = re.compile(
    rf"[a-zα-ω]bar|[а-яе]+|[a-z]+|\d+(?:\.\d+)*|[{SYMBOLS}]", re.IGNORECASE
)


def tokenize(text: str) -> list[str]:
    toks = []
    for tok in TOKEN_RE.findall(normalize(text).lower()):
        if re.match(r"[а-я]", tok):
            if len(tok) > 2:
                toks.append(_stem(tok))
        else:
            toks.append(tok)
    return toks
