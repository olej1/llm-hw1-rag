"""Проверка формата и оценка по правилам задания.

python evaluate.py --questions q.jsonl --gold gold.jsonl --answers a.jsonl --trace t.jsonl \
    [--name final --csv experiments.csv --cref 4000 --no-q]
"""
import argparse
import csv
import json
import os
import statistics

LIMITS = {"llm_calls": 4, "input_tokens": 16384, "output_tokens": 4096}
COLUMNS = ["name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens",
           "mean_llm_calls", "mean_cost", "median_latency_ms"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def format_errors(a, q):
    errs = []
    opt = {o["id"] for o in q["options"]}
    if set(a) - {"id", "answer_ids", "abstain", "rationale", "citations"}:
        errs.append("extra fields")
    if not isinstance(a.get("abstain"), bool):
        errs.append("abstain not bool")
    ans, cit = a.get("answer_ids", []), a.get("citations", [])
    if len(set(ans)) != len(ans) or not set(ans) <= opt:
        errs.append("bad answer_ids")
    if a.get("abstain") and (ans or cit):
        errs.append("abstain with answers/citations")
    if not a.get("abstain") and (not ans or not cit):
        errs.append("empty answers/citations")
    if len(cit) > 4 or len(set(cit)) != len(cit):
        errs.append("bad citations")
    r = a.get("rationale")
    if not isinstance(r, str) or not r.strip() or len(r) > 1200:
        errs.append("bad rationale")
    return errs


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def evaluate(questions, gold, answers, traces, check_format=True):
    qmap = {q["id"]: q for q in questions}
    amap = {a["id"]: a for a in answers}
    tmap = {t["id"]: t for t in traces}
    assert len(amap) == len(answers), "duplicate ids in answers"
    rows, by_cat, correct, over_limit = [], {}, 0, 0
    for g in gold:
        qid = g["id"]
        a, t = amap.get(qid), tmap.get(qid, {})
        bad = a is None or (check_format and format_errors(a, qmap[qid]))
        exceeded = any(t.get(k, 0) > v for k, v in LIMITS.items())
        over_limit += exceeded
        if g["abstain"]:
            ok = bool(a) and a.get("abstain") is True
            s = 1.0 if ok and not bad and not exceeded else 0.0
        else:
            ok = bool(a) and not a.get("abstain") and set(a.get("answer_ids", [])) == set(g["answer_ids"])
            fbest = max((f1(a.get("citations", []), cs) for cs in g["citation_sets"]), default=0.0) if a else 0.0
            s = (0.8 + 0.2 * fbest) if ok and not bad and not exceeded else 0.0
        correct += ok
        by_cat.setdefault(g["category"], []).append(s)
        rows.append({"id": qid, "category": g["category"], "ok": ok, "score": round(s, 3),
                     "format": format_errors(a, qmap[qid]) if a else ["missing"]})
    cat_means = {c: statistics.mean(v) for c, v in by_cat.items()}
    Q = statistics.mean(cat_means.values())
    n = len(gold)
    cost = [t.get("input_tokens", 0) + 4 * t.get("output_tokens", 0) + 256 * t.get("llm_calls", 0)
            for t in (tmap.get(g["id"], {}) for g in gold)]
    summary = {
        "answer_accuracy": correct / n, "Q": Q, "categories": cat_means,
        "mean_input_tokens": statistics.mean(tmap.get(g["id"], {}).get("input_tokens", 0) for g in gold),
        "mean_output_tokens": statistics.mean(tmap.get(g["id"], {}).get("output_tokens", 0) for g in gold),
        "mean_llm_calls": statistics.mean(tmap.get(g["id"], {}).get("llm_calls", 0) for g in gold),
        "mean_cost": statistics.mean(cost),
        "median_latency_ms": statistics.median(tmap.get(g["id"], {}).get("elapsed_ms", 0) for g in gold),
        "limit_violations": over_limit,
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--name")
    ap.add_argument("--csv")
    ap.add_argument("--cref", type=float, help="средняя стоимость baseline для множителя E")
    ap.add_argument("--no-q", action="store_true", help="режим без контекста: Q не считается")
    args = ap.parse_args()
    summary, rows = evaluate(load(args.questions), load(args.gold), load(args.answers), load(args.trace),
                             check_format=not args.no_q)
    for r in rows:
        print(r)
    if args.cref:
        E = min(1.0, args.cref / max(summary["mean_cost"], 1))
        if summary["limit_violations"]:
            E = 0.0
        summary["E"] = E
        summary["final_score"] = 75 * summary["Q"] + 25 * summary["Q"] * E
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.csv and args.name:
        new = not os.path.exists(args.csv)
        rows_csv = [] if new else [r for r in csv.DictReader(open(args.csv, encoding="utf-8")) if r["name"] != args.name]
        row = {"name": args.name, **{k: round(summary[k], 4) for k in COLUMNS[1:]}}
        if args.no_q:
            row["Q"] = ""
        rows_csv.append(row)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows_csv)


if __name__ == "__main__":
    main()
