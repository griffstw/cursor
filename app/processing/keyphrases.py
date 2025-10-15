from __future__ import annotations

import re
from collections import Counter
from typing import List

# Lightweight RAKE-like implementation without NLTK downloads
# - Splits text by stopwords and punctuation into candidate phrases
# - Ranks by degree*frequency score similar to RAKE heuristic

_DEFAULT_STOPWORDS = set(
    """
    a about above after again against all am an and any are aren't as at be because been before being below
    between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself
    his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
    not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
    so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this that those through to too under until up very was wasn't we we'd we'll we're we've were
    weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

_PUNCT_SPLIT = re.compile(r"[\s,.;:!?()\[\]{}\-\n\r]+")


def _candidate_phrases(text: str) -> List[str]:
    tokens = [t.lower() for t in _PUNCT_SPLIT.split(text) if t]
    phrases: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok in _DEFAULT_STOPWORDS:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        phrases.append(current)
    return [" ".join(p) for p in phrases if p]


def extract_key_phrases(text: str, top_k: int = 12) -> List[str]:
    cands = _candidate_phrases(text)
    if not cands:
        return []

    word_freq: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()

    for phrase in cands:
        words = phrase.split()
        degree = len(words) - 1
        unique_words = set(words)
        for w in words:
            word_freq[w] += 1
            word_degree[w] += degree

    word_score = {w: (word_degree[w] + word_freq[w]) / (word_freq[w]) for w in word_freq}

    phrase_scores = []
    for phrase in cands:
        score = sum(word_score.get(w, 0.0) for w in phrase.split())
        phrase_scores.append((phrase, score))

    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    out = []
    seen = set()
    for phrase, _ in phrase_scores:
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= top_k:
            break
    return out
