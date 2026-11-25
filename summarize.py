from collections import Counter, defaultdict

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def textrank(sentences, top_k=2, damping=0.85):
    """Rank sentences with PageRank over a TF-IDF cosine-similarity graph."""
    uniq = list(dict.fromkeys(s for s in sentences if s.strip()))
    if len(uniq) <= top_k:
        return uniq
    vec = TfidfVectorizer(token_pattern=r"[^\s]+").fit_transform(uniq)
    sim = cosine_similarity(vec)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(uniq)))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if sim[i, j] > 0:
                graph.add_edge(i, j, weight=float(sim[i, j]))
    scores = nx.pagerank(graph, alpha=damping, weight="weight")
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [uniq[i] for i in sorted(ranked)]


def opinion_summary(analyses, top_k=2):
    """analyses: list of pipeline outputs (each with an 'aspects' list).
    Returns per-aspect counts, majority polarity and TextRank-picked evidence."""
    by_aspect = defaultdict(lambda: {"counts": Counter(), "clauses": defaultdict(list)})
    for a in analyses:
        for asp in a["aspects"]:
            entry = by_aspect[asp["aspect"]]
            entry["counts"][asp["label"]] += 1
            entry["clauses"][asp["label"]].append(asp["clause"])
    summary = []
    for aspect, e in by_aspect.items():
        total = sum(e["counts"].values())
        ranked = e["counts"].most_common()
        majority, n = ranked[0]
        tie = len(ranked) > 1 and ranked[1][1] == n
        pool = [c for lab, _ in ranked for c in e["clauses"][lab]] if tie else e["clauses"][majority]
        evidence = textrank(pool, top_k=top_k)
        if tie:
            majority = "mixed"
        summary.append({
            "aspect": aspect,
            "mentions": total,
            "positive": e["counts"]["positive"],
            "negative": e["counts"]["negative"],
            "neutral": e["counts"]["neutral"],
            "verdict": majority,
            "share": round(n / total, 2),
            "evidence": evidence,
        })
    summary.sort(key=lambda s: -s["mentions"])
    return summary


def summary_text(summary):
    lines = []
    for s in summary:
        pct = int(round(100 * s["share"]))
        quote = f' e.g. "{s["evidence"][0]}"' if s["evidence"] else ""
        if s["verdict"] == "mixed":
            lines.append(f'{s["aspect"].capitalize()}: mixed ({s["positive"]} positive, {s["negative"]} negative, {s["neutral"]} neutral).{quote}')
        else:
            lines.append(f'{s["aspect"].capitalize()}: mostly {s["verdict"]} ({pct}% of {s["mentions"]} mention{"s" if s["mentions"] != 1 else ""}).{quote}')
    return "\n".join(lines)
