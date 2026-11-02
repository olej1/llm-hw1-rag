"""PDF учебника -> corpus.jsonl (block_id, page, section, text).

Проблемы PDF и как они решаются:
  * кириллица в шрифтах LaTeX (T2A) читается как Latin-1 -> перекодируем в CP1251
    (плюс особые коды T2A: § и тире);
  * дробные черты и черты сопряжения нарисованы линиями, а не символами ->
    восстанавливаем по геометрии: \\frac{num}{den} и \\bar{z};
  * степени/индексы видны только по размеру и сдвигу шрифта -> ^{...}, _{...};
  * большие операторы и скобки (шрифт CMEX) -> ∫ ∑ ∮ ( ) | с пределами _{..}^{..}.
LLM при сборке корпуса не используется: всё восстанавливается детерминированно.
"""
import argparse
import json
import re

import pymupdf

T2A_SPECIAL = {0x16: "—", 0x15: "–", 0x9F: "§"}
CMEX_MAP = {
    "\x00": "(", "\x01": ")", "\x10": "(", "\x11": ")", "\x12": "(", "\x13": ")",
    "\x20": "(", "\x21": ")", "\x30": "(", "\x31": ")", "\x40": "(", "\x41": ")",
    "\x02": "[", "\x03": "]", "\x14": "[", "\x15": "]", "\x22": "[", "\x23": "]",
    "\x08": "{", "\x09": "}", "\x1a": "{", "\x1b": "}", "\x28": "{", "\x29": "}",
    "\x0c": "|", "\x0d": "‖", "\x0e": "/", "\x0f": "\\",
    "P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫", "H": "∮", "I": "∮",
}
BIG_OPS = {"∑", "∏", "∫", "∮"}
DISPLAY_OPS = set("XYZI")  # display-варианты операторов в CMEX: пределы снизу/сверху

HEADER_RE = re.compile(r"^(\d{1,3}\s*)?(ЛЕКЦИИ ПО ТФКП|В\. ?В\. ГОРЯЙНОВ, Е\. ?С\. ПОЛОВИНКИН)?(\s*\d{1,3})?$")
BLOCK_START_RE = re.compile(
    r"^(Теорема|Определение|Лемма|Следствие|Замечание|Пример|Доказательство|Утверждение|§\s*\d|\d+\.\d+\.\s+[А-Я])"
)
SECTION_RE = re.compile(r"^§\s*(\d+)\.\s*(.+)")
MAX_BLOCK_CHARS = 1000


def fix_char(c: str) -> str:
    """Символы 0x80–0xFF — это байты CP1251, ошибочно прочитанные как Latin-1."""
    o = ord(c)
    if o in T2A_SPECIAL:
        return T2A_SPECIAL[o]
    if 0x80 <= o <= 0xFF:
        try:
            return bytes([o]).decode("cp1251")
        except UnicodeDecodeError:
            return c
    return c


class Item:
    """Кусок текста с геометрией: один символ или собранная формула."""

    def __init__(self, text, bbox, size, base, cmex=False, raw=""):
        self.text = text
        self.raw = raw
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.size = size
        self.base = base  # y базовой линии
        self.cmex = cmex

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def space(self):
        return self.text == " "


def page_items(page):
    items = []
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                cmex = s["font"].startswith("CMEX")
                for ch in s["chars"]:
                    c = ch["c"]
                    if cmex:
                        t = CMEX_MAP.get(c, "")
                    elif c.isspace():
                        t = " "
                    else:
                        t = fix_char(c)
                    if not t or 0xE000 <= ord(t[0]) <= 0xF8FF or (ord(t[0]) < 32):
                        continue
                    items.append(Item(t, ch["bbox"], s["size"], ch["origin"][1], cmex, c))
    return merge_stacked_cmex(items)


def merge_stacked_cmex(items):
    """Большие скобки/черты собраны из кусков друг над другом -> один символ."""
    rest, cm = [], []
    for it in items:
        (cm if it.cmex and it.text not in BIG_OPS else rest).append(it)
    cm.sort(key=lambda it: (round(it.x0), it.y0))
    merged = []
    for it in cm:
        last = merged[-1] if merged else None
        if last and last.text == it.text and abs(last.x0 - it.x0) < 1 and it.y0 - last.y1 < 2:
            last.y1 = max(last.y1, it.y1)
            last.base = last.cy + 2.5
        else:
            it.base = it.cy + 2.5
            merged.append(it)
    for it in rest:
        if it.cmex:  # большой оператор: базовая линия по оси формулы
            # display-глиф ~22pt уходит вниз от bbox: ось ≈ y0 + 11
            it.base = it.y0 + 13.5 if it.raw in DISPLAY_OPS else it.cy + 2.5
    return rest + merged


def horizontal_lines(page):
    res = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.height < 1.2 and 2 < r.width < 400:
            res.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    return sorted(res, key=lambda t: t[1] - t[0])  # короткие (вложенные) первыми


def line_text(items):
    """Склеивает элементы одной строки слева направо, размечая степени/индексы."""
    items = sorted(items, key=lambda it: it.x0)
    solid = [it for it in items if not it.space]
    if not solid:
        return ""
    main = max(it.size for it in solid)
    big = [it for it in solid if it.size >= main * 0.85 and not it.cmex]
    bases = sorted(it.base for it in (big or solid))
    base = bases[len(bases) // 2]
    out, mode, prev = [], None, None
    for it in items:
        if it.space:
            if out and out[-1] != " ":
                out.append(" ")
            continue
        m = None
        if it.size < main * 0.85 and not it.text.startswith("\\frac") and not it.cmex:
            if it.base < base - 1:
                m = "^"
            elif it.base > base + 0.5:
                m = "_"
        if m != mode:
            if mode:
                out.append("}")
            if m:
                if out and out[-1] == " ":
                    out.pop()
                out.append(m + "{")
            mode = m
        elif prev is not None and it.x0 - prev.x1 > 0.6 * main and out[-1] != " ":
            out.append(" ")
        out.append(it.text)
        prev = it
    if mode:
        out.append("}")
    return "".join(out).strip()


def rebuild_math(items, hlines):
    """\\frac по дробной черте, \\bar по короткой черте над символом, пределы у ∫∑."""
    for x0, x1, ly in hlines:
        inside = [it for it in items if x0 - 1 <= it.cx <= x1 + 1 and not it.space]
        # замеры по PDF: числитель на ~4.2pt выше черты, знаменатель на ~9.3pt ниже,
        # буква под чертой сопряжения на ~5.7pt ниже, строка текста выше — от 6.5pt.
        core = [it for it in inside if ly - 5.5 < it.base < ly - 0.8]
        if any("А" <= it.text[0] <= "я" for it in core):  # кусок слова над чертой — не числитель
            core = []
        above = core + [it for it in inside if it.size < 8.5 and ly - 11 < it.base <= ly - 5.5] if core else []
        below = [it for it in inside if ly + 3 < it.base < ly + 12]
        if above and below:
            num, den = line_text(above), line_text(below)
            used = set(map(id, above + below))
            items = [it for it in items if id(it) not in used]
            both = above + below
            bbox = (min(it.x0 for it in both), min(it.y0 for it in above),
                    max(it.x1 for it in both), max(it.y1 for it in below))
            size = max(it.size for it in both)
            items.append(Item(f"\\frac{{{num}}}{{{den}}}", bbox, size, ly + 2.5))
        elif below and not above and x1 - x0 < 30:
            for it in below:
                if it.base - ly < 10 and not it.text.startswith("\\"):
                    it.text = f"\\bar{{{it.text}}}"
    # пределы больших операторов (display-стиль: снизу и сверху)
    for op in [it for it in items if it.cmex and it.text in BIG_OPS]:
        if op.raw not in DISPLAY_OPS:  # текстовый стиль: пределы справа, их разметит line_text
            continue
        # у ∫ пределы смещены вправо (наклон знака), у ∑ — по центру
        right = 12 if op.text in ("∫", "∮") else 5
        near = [it for it in items if not it.space and it is not op and it.size < op.size * 0.85
                and op.x0 - 5 <= it.cx <= op.x1 + right]
        # bbox display-оператора покрывает только верх глифа (~10pt из ~22pt):
        # нижний предел на ~17pt ниже bbox, верхний — у верхнего края
        lo = [it for it in near if op.y1 + 4 < it.cy < op.y1 + 22]
        hi = [it for it in near if op.y0 - 7 < it.cy < op.y0 + 3]
        txt = op.text
        if lo:
            txt += "_{" + line_text(lo) + "}"
        if hi:
            txt += "^{" + line_text(hi) + "}"
        used = set(map(id, lo + hi))
        items = [it for it in items if id(it) not in used]
        op.text = txt
    return items


def group_lines(items):
    """Строки по базовой линии (у разных шрифтов bbox разные, а baseline общий)."""
    def is_small(it):
        return it.size < 8.5 and not it.text.startswith("\\frac")

    lines = []
    for it in sorted([it for it in items if not is_small(it)], key=lambda it: it.base):
        if lines and abs(lines[-1]["base"] - it.base) < 4.0:
            lines[-1]["items"].append(it)
        else:
            lines.append({"base": it.base, "items": [it]})
    # индексы/степени (мелкий шрифт) приклеиваем к ближайшей строке по вертикали
    for it in [it for it in items if is_small(it)]:
        best = min(lines, key=lambda ln: abs(ln["base"] - it.base), default=None)
        if best is not None and abs(best["base"] - it.base) < 8:
            best["items"].append(it)
        else:
            lines.append({"base": it.base, "items": [it]})
    lines.sort(key=lambda ln: ln["base"])
    out = []
    for ln in lines:
        t = line_text(ln["items"])
        if t:
            out.append((ln["base"], t))
    return out


def clean(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # переносы слов
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+}", "}", text)
    return text.strip()


def extract(pdf_path):
    doc = pymupdf.open(pdf_path)
    blocks, section, started = [], "", False
    cur = {"lines": [], "page": None}

    def flush():
        txt = clean("\n".join(cur["lines"]))
        if started and len(txt) > 20:
            blocks.append({"page": cur["page"], "section": section, "text": txt})
        cur["lines"] = []

    for pno in range(doc.page_count):
        page = doc[pno]
        items = rebuild_math(page_items(page), horizontal_lines(page))
        for _, text in group_lines(items):
            if HEADER_RE.match(text.strip()):
                continue
            m = SECTION_RE.match(text)
            starts = BLOCK_START_RE.match(text) is not None
            size = sum(len(l) for l in cur["lines"])
            if cur["lines"] and (starts or cur["page"] != pno + 1 or size > MAX_BLOCK_CHARS):
                flush()
            if m:
                started = True  # всё до «§ 1.» — оглавление
                section = f"§ {m.group(1)}. {m.group(2)}"
            cur["page"] = pno + 1
            cur["lines"].append(text)
        flush()
    for i, b in enumerate(blocks):
        b["block_id"] = f"b_{i:04d}"
    return [{k: b[k] for k in ("block_id", "page", "section", "text")} for b in blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    blocks = extract(args.pdf)
    with open(args.output, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"{len(blocks)} blocks -> {args.output}")


if __name__ == "__main__":
    main()
