"""Ответы на вопросы:
python predict.py --index ./index --questions q.jsonl --output answers.jsonl --trace trace.jsonl --config config.json

Режимы (config.pipeline.mode):
  no_context — модель без контекста (только для сравнения точности вариантов);
  single     — один поиск и один вызов LLM;
  final      — single + адаптивный повторный поиск, если модель сообщила, чего не хватает.
"""
import argparse
import json
import os
import re
import time

from openai import OpenAI

from retrieval import Retriever

SYSTEM = """Ты отвечаешь на тестовые вопросы по теории функций комплексного переменного строго по фрагментам учебника Горяйнова и Половинкина.
Правила:
1. Проверь каждый вариант отдельно. Правильных вариантов может быть несколько; выбери все верные.
2. Опирайся только на приведённые фрагменты; допустим короткий вывод или вычисление из них. Знания «из памяти» без опоры на фрагменты не используй.
3. abstain=true (отказ) — только если сам вопрос недоопределён: по условию нельзя однозначно выбрать ответ (не задана константа, ориентация, известна лишь оценка и т. п.). При отказе answer_ids и citations пустые, в rationale — каких данных не хватает.
4. Если для проверки нужен факт (теорема, формула), которого нет во фрагментах, НЕ отказывайся, а напиши в поле missing короткий поисковый запрос (например «теорема Морера треугольник»). Иначе missing — пустая строка.
5. citations — 1–4 id фрагментов, которые вместе подтверждают ответ; не добавляй лишних.
6. rationale — 1–3 коротких предложения: на какое утверждение книги опирается ответ и почему отвергнуты остальные варианты.
Ответь только JSON без Markdown:
{"answer_ids": ["A"], "abstain": false, "rationale": "...", "citations": ["b_0000"], "missing": ""}"""

SYSTEM_REWRITE = """Ты помогаешь искать по учебнику ТФКП. По вопросу с вариантами назови 2–3 поисковых запроса на русском: названия теорем, формул и понятий учебника, которые нужны для проверки вариантов (например «интегральная формула Коши для производных», «теорема единственности предельная точка нулей»). Не отвечай на вопрос.
Ответь только JSON: {"queries": ["...", "..."]}"""

SYSTEM_NO_CONTEXT ="""Ты отвечаешь на тестовые вопросы по теории функций комплексного переменного. Правильных вариантов может быть несколько; выбери все верные. Если по условию ответ нельзя определить однозначно, верни abstain=true и пустой answer_ids.
Ответь только JSON без Markdown:
{"answer_ids": ["A"], "abstain": false, "rationale": "..."}"""


def approx_tokens(text):
    # консервативная оценка для бюджета (рус. текст + формулы ≈ 2.6 символа на токен у Qwen)
    return int(len(text) / 2.6) + 1


def load_cfg(path):
    cfg = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(cfg["models"]["cache_dir"]):
        cfg["models"]["cache_dir"] = os.path.join(base, cfg["models"]["cache_dir"])
    cfg["runtime_cfg"] = json.load(open(os.path.join(base, cfg["runtime"]), encoding="utf-8"))
    return cfg


class LLM:
    def __init__(self, runtime):
        self.rt = runtime
        self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                             timeout=runtime["timeouts"]["request_s"], max_retries=0)
        self.model = os.environ.get("LLM_MODEL", runtime["model"]["served_model_name"])
        self.submission = os.environ.get("SUBMISSION_ID", "local")

    def chat(self, qid, system, user, usage, max_tokens=None):
        lim = self.rt["limits_per_question"]
        max_out = min(max_tokens or self.rt["max_output_tokens_per_call"],
                      lim["output_tokens_total"] - usage["output_tokens"])
        est_in = approx_tokens(system + user) + 20
        if (usage["llm_calls"] >= lim["llm_calls"] or max_out < min(200, max_tokens or 200)
                or usage["input_tokens"] + est_in > lim["input_tokens_total"]
                or est_in + max_out > lim["input_plus_max_output_per_request"]):
            return None
        s = self.rt["sampling"]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_out, temperature=s["temperature"], top_p=s["top_p"], seed=s["seed"],
            extra_headers={"X-Question-ID": qid, "X-Submission-ID": self.submission},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.rt["thinking"]["enabled"]}},
        )
        usage["llm_calls"] += 1
        usage["input_tokens"] += resp.usage.prompt_tokens
        usage["output_tokens"] += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return {}


def format_question(q):
    return q["question"] + "\n" + "\n".join(f"{o['id']}. {o['text']}" for o in q["options"])


def format_context(retriever, ids, budget):
    parts, used = [], 0
    for bid in ids:
        b = retriever.get(bid)
        chunk = f"[{bid}] (с. {b['page']}; {b['section']})\n{b['text']}"
        t = approx_tokens(chunk)
        if used + t > budget and parts:
            continue
        parts.append(chunk)
        used += t
    return "\n\n".join(parts)


def finalize(q, data, shown_ids, fallback_cites):
    """Приводит ответ модели к строгому формату задания."""
    opt_ids = [o["id"] for o in q["options"]]
    ans = [a for a in dict.fromkeys(str(x).strip() for x in data.get("answer_ids") or []) if a in opt_ids]
    abstain = bool(data.get("abstain")) or not ans
    rationale = str(data.get("rationale") or "").strip()
    if abstain:
        return {"id": q["id"], "answer_ids": [], "abstain": True,
                "rationale": (rationale or "Недостаточно оснований для однозначного ответа.")[:1200],
                "citations": []}
    cites = [c for c in dict.fromkeys(str(x).strip() for x in data.get("citations") or []) if c in shown_ids][:4]
    if not cites:
        cites = fallback_cites[:1]
    return {"id": q["id"], "answer_ids": ans, "abstain": False,
            "rationale": (rationale or "Ответ следует из процитированных фрагментов.")[:1200], "citations": cites}


def answer(q, cfg, llm, retriever):
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    mode = cfg["pipeline"]["mode"]
    trace = {"decisions": []}
    if mode == "no_context":
        raw = llm.chat(q["id"], SYSTEM_NO_CONTEXT, format_question(q), usage)
        d = parse_json(raw)
        opt = [o["id"] for o in q["options"]]
        ans = [a for a in d.get("answer_ids") or [] if a in opt]
        out = {"id": q["id"], "answer_ids": ans, "abstain": bool(d.get("abstain")) or not ans,
               "rationale": str(d.get("rationale") or "без контекста")[:1200], "citations": []}
        if out["abstain"]:
            out["answer_ids"] = []
        return out, usage, [], trace

    budget = cfg["pipeline"]["context_token_budget"]
    rewrites = []
    if cfg["pipeline"].get("rewrite"):
        # rewriting: модель называет теоремы/понятия учебника -> дополнительные поисковые запросы
        d = parse_json(llm.chat(q["id"], SYSTEM_REWRITE, format_question(q), usage, max_tokens=120))
        rewrites = [str(x) for x in d.get("queries") or [] if str(x).strip()][:3]
        trace["rewrites"] = rewrites
    ranked = retriever.search(q["question"], q["options"], extra_queries=rewrites)
    shown = retriever.expand(ranked)
    retrieved = list(shown)
    ctx = format_context(retriever, shown, budget)
    user = f"Фрагменты учебника:\n\n{ctx}\n\nВопрос:\n{format_question(q)}"
    data = parse_json(llm.chat(q["id"], SYSTEM, user, usage))

    missing = str(data.get("missing") or "").strip()
    if cfg["pipeline"].get("adaptive_retry") and missing:
        # адаптивный шаг: модель указала, какого факта не хватает -> повторный поиск
        extra = retriever.search(q["question"], q["options"], extra_queries=[missing] + rewrites)
        cited = [c for c in data.get("citations") or [] if c in shown]
        new_ids = list(dict.fromkeys(cited + [b for b in retriever.expand(extra) if b not in cited]))
        retrieved += [b for b in new_ids if b not in retrieved]
        ctx2 = format_context(retriever, new_ids, budget)
        user2 = (f"Фрагменты учебника:\n\n{ctx2}\n\nВопрос:\n{format_question(q)}\n\n"
                 "Это повторная попытка с дополнительным поиском: missing оставь пустым.")
        raw2 = llm.chat(q["id"], SYSTEM, user2, usage)
        trace["decisions"].append({"action": "retry_search", "reason": "missing", "query": missing,
                                   "executed": raw2 is not None})
        if raw2 is not None:
            d2 = parse_json(raw2)
            if d2:
                data, shown = d2, new_ids
    else:
        trace["decisions"].append({"action": "accept", "reason": "no missing" if not missing else "retry disabled"})
    return finalize(q, data, set(shown), ranked), usage, retrieved, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    llm = LLM(cfg["runtime_cfg"])
    retriever = None if cfg["pipeline"]["mode"] == "no_context" else Retriever(args.index, cfg)
    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    with open(args.output, "w", encoding="utf-8") as fo, open(args.trace, "w", encoding="utf-8") as ft:
        for q in questions:
            t0 = time.time()
            try:
                out, usage, retrieved, extra = answer(q, cfg, llm, retriever)
            except Exception as e:  # ответ всё равно должен быть валидным
                out = {"id": q["id"], "answer_ids": [], "abstain": True,
                       "rationale": f"Ошибка обработки: {type(e).__name__}", "citations": []}
                usage, retrieved, extra = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}, [], {"error": repr(e)}
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": q["id"], **usage, "elapsed_ms": int((time.time() - t0) * 1000),
                                 "retrieved_block_ids": retrieved, **extra}, ensure_ascii=False) + "\n")
            fo.flush(); ft.flush()
            print(q["id"], out["answer_ids"] or "ABSTAIN", usage, flush=True)


if __name__ == "__main__":
    main()
