"""tfkp_15_qa.md (+ авторские вопросы) -> *_questions.jsonl и *_gold.jsonl.

Допустимые наборы источников задаются якорями (страница PDF, начало текста блока),
а не номерами block_id: так эталоны не ломаются при пересборке корпуса.
"""
import argparse
import json
import re

# Якоря эталонных блоков для 15 открытых вопросов (по страницам из tfkp_15_qa.md).
TH21 = (11, "Теорема 2.1")
DEF22 = (11, "Определение 2.2")
DZBAR = (12, "df =")
CR_INTRO = (12, "Соотношения (2.3)")
LAPLACE = (13, "Система уравнений Коши")
DEF23 = (13, "Определение 2.3")
TH41 = (26, "Теорема 4.1")
EX_2PI = (28, "случае таких простых областей")
REM41 = (31, "Замечание 4.1")
TH44 = (30, "Теорема 4.4")
TH43 = (30, "Теорема 4.3")
TH51 = (31, "Теорема 5.1")
COR51 = (33, "Следствие 5.1")
EXP = (20, "Повторяя этот процесс")
TH53 = (34, "Теорема 5.3")
MORERA = (35, "Теорема 5.4")
TH61 = (36, "Теорема 6.1")
REM61 = (37, "Замечание 6.1")
TH62 = (37, "Теорема 6.2")
LIOUV = (37, "Следствие 6.1")
ENTIRE = (37, "Определение 6.1")
TRIG = (21, "Тригонометрические функции")
TH63 = (38, "Теорема 6.3")

GOLD_ANCHORS = {
    1: [[TH21], [TH21, DEF22]],
    2: [[DZBAR], [CR_INTRO, DZBAR]],
    3: [[TH21, DEF23], [LAPLACE, DEF23], [TH21, LAPLACE, DEF23]],
    4: [[EX_2PI, REM41], [TH41, EX_2PI, REM41]],
    5: [[TH44], [TH43, TH44]],
    6: [[TH51, COR51]],
    7: [[TH53], [EXP, TH53]],
    8: [[MORERA]],
    9: [[LAPLACE, TH53], [TH21, LAPLACE, TH53], [DEF23, TH53], [TH21, DEF23, TH53]],
    10: [[TH61], [TH61, REM61]],
    11: [[TH62], [TH62, LIOUV]],
    12: [[TH63], [TRIG, TH63], [ENTIRE, TH63], [TRIG, ENTIRE, TH63]],
    13: [[LIOUV]],
    14: [[EX_2PI]],
    15: [[TH61]],
}
CATEGORY = {"условия": "conditions", "сопоставление": "comparison",
            "применение": "computation", "недостаточность данных": "insufficient"}


def resolve(corpus, anchor):
    page, prefix = anchor
    for b in corpus:
        if b["page"] == page and b["text"].startswith(prefix):
            return b["block_id"]
    for b in corpus:  # запасной вариант: заголовок не в начале блока
        if b["page"] == page and prefix in b["text"]:
            return b["block_id"]
    raise KeyError(f"anchor not found: {anchor}")


def parse_qa(md_path):
    text = open(md_path, encoding="utf-8").read()
    parts = re.split(r"\n## (\d+)\. ", text)[1:]
    items = []
    for num, body in zip(parts[::2], parts[1::2]):
        num = int(num)
        qtype = re.search(r"\*\*Тип:\*\*\s*([^.]+)\.", body).group(1).strip()
        question = re.search(r"\*\*Вопрос\.\*\*\s*(.+?)\n\n", body, re.S).group(1).strip()
        options = [{"id": m.group(1), "text": m.group(2).strip()}
                   for m in re.finditer(r"^- \*\*([A-F])\.\*\*\s*(.+)$", body, re.M)]
        ans = re.search(r"\*\*Ответ:\*\*\s*(.+?)\.?\n", body).group(1).strip()
        abstain = ans.startswith("Отказ")
        answer_ids = [] if abstain else [a.strip() for a in ans.split(",")]
        after = body.split("**Ответ:**", 1)[1].split("\n\n", 2)
        rationale = after[1].strip() if len(after) > 1 else ""
        items.append(dict(num=num, qtype=qtype, question=question, options=options,
                          answer_ids=answer_ids, abstain=abstain, rationale=rationale))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-prefix", default="dev")
    args = ap.parse_args()
    corpus = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    items = parse_qa(args.qa)
    with open(f"{args.out_prefix}_questions.jsonl", "w", encoding="utf-8") as fq, \
         open(f"{args.out_prefix}_gold.jsonl", "w", encoding="utf-8") as fg:
        for it in items:
            qid = f"q_{it['num']:03d}"
            fq.write(json.dumps({"id": qid, "question": it["question"], "options": it["options"]},
                                ensure_ascii=False) + "\n")
            sets = [[resolve(corpus, a) for a in s] for s in GOLD_ANCHORS[it["num"]]]
            fg.write(json.dumps({"id": qid, "category": CATEGORY[it["qtype"]],
                                 "answer_ids": it["answer_ids"], "abstain": it["abstain"],
                                 "citation_sets": [] if it["abstain"] else sets,
                                 "evidence": sets[0], "rationale": it["rationale"]},
                                ensure_ascii=False) + "\n")
    print(f"{len(items)} questions -> {args.out_prefix}_questions.jsonl / _gold.jsonl")


if __name__ == "__main__":
    main()
