"""Индекс и поиск: BM25 (свой, numpy) + dense (bge-m3) + RRF + reranker + соседи."""
import json
import math
import os
from collections import Counter

import numpy as np

from textnorm import normalize, tokenize


def _st_model(name, revision, cache_dir, kind):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if kind == "embed":
        return SentenceTransformer(name, revision=revision, cache_folder=cache_dir, device="cpu")
    return CrossEncoder(name, revision=revision, cache_folder=cache_dir, device="cpu", max_length=512)


def doc_text(block):
    return f"{block['section']}. {block['text']}" if block.get("section") else block["text"]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = np.array([len(t) for t in docs_tokens], dtype=float)
        self.avg = float(self.len.mean())
        df = Counter(w for t in docs_tokens for w in set(t))
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q_tokens):
        s = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for w in set(q_tokens):
            idf = self.idf.get(w)
            if idf is None:
                continue
            f = np.array([tf.get(w, 0) for tf in self.tf], dtype=float)
            s += idf * f * (self.k1 + 1) / (f + norm)
        return s


def build(corpus_path, out_dir, cfg):
    blocks = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blocks.jsonl"), "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    tokens = [tokenize(doc_text(b)) for b in blocks]
    with open(os.path.join(out_dir, "bm25_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False)
    r = cfg["retrieval"]
    if r.get("use_dense", True):
        m = cfg["models"]["embedding"]
        model = _st_model(m["name"], m["revision"], cfg["models"]["cache_dir"], "embed")
        emb = model.encode([normalize(doc_text(b)) for b in blocks], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=True)
        np.save(os.path.join(out_dir, "dense.npy"), emb.astype(np.float32))
    with open(os.path.join(out_dir, "index_meta.json"), "w") as f:
        json.dump({"n_blocks": len(blocks), "embedding": cfg["models"]["embedding"]}, f, indent=2)
    return len(blocks)


class Retriever:
    def __init__(self, index_dir, cfg):
        self.cfg = cfg
        self.r = cfg["retrieval"]
        self.blocks = [json.loads(l) for l in open(os.path.join(index_dir, "blocks.jsonl"), encoding="utf-8")]
        self.pos = {b["block_id"]: i for i, b in enumerate(self.blocks)}
        self.bm25 = BM25(json.load(open(os.path.join(index_dir, "bm25_tokens.json"), encoding="utf-8")))
        self.dense = self.embedder = self.reranker = None
        dpath = os.path.join(index_dir, "dense.npy")
        if self.r.get("use_dense", True) and os.path.exists(dpath):
            self.dense = np.load(dpath)
            m = cfg["models"]["embedding"]
            self.embedder = _st_model(m["name"], m["revision"], cfg["models"]["cache_dir"], "embed")
        if self.r.get("use_reranker", True):
            m = cfg["models"]["reranker"]
            self.reranker = _st_model(m["name"], m["revision"], cfg["models"]["cache_dir"], "rerank")

    def _rank_lists(self, query, k):
        lists = [np.argsort(-self.bm25.scores(tokenize(query)))[:k]]
        if self.dense is not None:
            q = self.embedder.encode([normalize(query)], normalize_embeddings=True)[0]
            lists.append(np.argsort(-(self.dense @ q))[:k])
        return lists

    def search(self, question, options=(), extra_queries=()):
        """Мульти-запрос: вопрос, вопрос+вариант для каждого варианта, доп. запросы -> RRF."""
        k, c = self.r["candidates_per_query"], self.r["rrf_k"]
        queries = [question] + list(extra_queries)
        if self.r.get("per_option_queries", True):
            queries += [f"{question} {o['text']}" for o in options]
        fused = Counter()
        for q in queries:
            for lst in self._rank_lists(q, k):
                for rank, i in enumerate(lst):
                    fused[int(i)] += 1.0 / (c + rank + 1)
        cand = [i for i, _ in fused.most_common(self.r["rerank_pool"])]
        if self.reranker is not None:
            rq = question + " " + " ".join(o["text"] for o in options)
            for q in extra_queries:
                rq += " " + q
            pairs = [(normalize(rq), normalize(doc_text(self.blocks[i]))) for i in cand]
            scores = self.reranker.predict(pairs, batch_size=16)
            cand = [cand[j] for j in np.argsort(-np.asarray(scores))]
        return [self.blocks[i]["block_id"] for i in cand[: self.r["top_k"]]]

    def with_neighbors(self, ids):
        """Добавляет следующий блок той же страницы (теорема -> доказательство/продолжение)."""
        if not self.r.get("neighbors", True):
            return list(ids)
        out = []
        for bid in ids:
            i = self.pos[bid]
            for j in (i, i + 1):
                if j < len(self.blocks):
                    nb = self.blocks[j]["block_id"]
                    if nb not in out and (j == i or self.blocks[j]["page"] - self.blocks[i]["page"] <= 1):
                        out.append(nb)
        return out

    def get(self, bid):
        return self.blocks[self.pos[bid]]
