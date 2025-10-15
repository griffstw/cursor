from __future__ import annotations

import os
from typing import Iterable, List, Optional

from openai import OpenAI
import re
from collections import Counter

from app.processing.text import clean_text, split_sentences
from app.processing.keyphrases import extract_key_phrases


class LLMClient:
    """Pluggable LLM client with environment-driven provider selection.

    - Primary: OpenAI via `OPENAI_API_KEY`
    - Fallback: Simple local heuristic summarizer for offline/demo
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client: Optional[OpenAI] = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    def summarize(self, text: str, max_words: int = 120) -> str:
        if not text.strip():
            return ""
        if self._client is None:
            return self._local_summarize(text, max_words)
        prompt = (
            "You are a concise expert summarizer. "
            f"Summarize the following content in up to {max_words} words, "
            "preserving key ideas and terminology.\n\n" + text
        )
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return self._local_summarize(text, max_words)

    def generate_qa(self, text: str, num_pairs: int = 8) -> List[dict]:
        if not text.strip():
            return []
        if self._client is None:
            return self._local_qa(text, num_pairs)
        prompt = (
            "Create concise study Q&A pairs from the content. "
            f"Return exactly {num_pairs} JSON objects with 'question' and 'answer'.\n\n" + text
        )
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message.content or "[]"
            import json

            data = json.loads(content)
            cleaned = []
            for item in data:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    cleaned.append({"question": q, "answer": a})
            return cleaned[:num_pairs]
        except Exception:
            return self._local_qa(text, num_pairs)

    def generate_flashcards(self, text: str, num_cards: int = 12) -> List[dict]:
        if not text.strip():
            return []
        if self._client is None:
            return self._local_flashcards(text, num_cards)
        prompt = (
            "Create concise flashcards capturing key facts/definitions. "
            f"Return exactly {num_cards} JSON objects with 'front' and 'back'.\n\n" + text
        )
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message.content or "[]"
            import json

            data = json.loads(content)
            cleaned = []
            for item in data:
                f = str(item.get("front", "")).strip()
                b = str(item.get("back", "")).strip()
                if f and b:
                    cleaned.append({"front": f, "back": b})
            return cleaned[:num_cards]
        except Exception:
            return self._local_flashcards(text, num_cards)

    # -------------------- Local fallbacks --------------------
    def _local_summarize(self, text: str, max_words: int) -> str:
        # Lightweight frequency-based extractive summarizer (no NLTK required)
        sentences = split_sentences(clean_text(text))
        if not sentences:
            return ""
        tokens_re = re.compile(r"[A-Za-z0-9']+")
        stopwords = {
            "a","about","above","after","again","against","all","am","an","and","any","are","as","at","be","because","been","before","being","below","between","both","but","by","can","could","did","do","does","doing","down","during","each","few","for","from","further","had","has","have","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","it","its","itself","just","let","me","more","most","my","myself","no","nor","not","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","same","she","should","so","some","such","than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","very","was","we","were","what","when","where","which","while","who","whom","why","with","would","you","your","yours","yourself","yourselves"
        }
        def sentence_tokens(s: str) -> list[str]:
            return [t.lower() for t in tokens_re.findall(s)]

        word_freq: Counter[str] = Counter()
        for s in sentences:
            for t in sentence_tokens(s):
                if t in stopwords or len(t) <= 2:
                    continue
                word_freq[t] += 1
        if not word_freq:
            return " ".join(sentences[: max(1, min(3, len(sentences)))])

        scores: list[tuple[int, float]] = []
        for idx, s in enumerate(sentences):
            toks = [t for t in sentence_tokens(s) if t not in stopwords and len(t) > 2]
            if not toks:
                scores.append((idx, 0.0))
                continue
            score = sum(word_freq[t] for t in toks) / (len(toks) ** 0.8)
            scores.append((idx, score))

        sentences_target = max(3, min(10, max_words // 25))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:sentences_target]
        top_indices = sorted(i for i, _ in top)
        selected = [sentences[i] for i in top_indices]
        result = " ".join(selected)
        # Soft trim if too long
        words = result.split()
        if len(words) > max_words * 2:
            result = " ".join(words[: max_words * 2])
        return result

    def _local_qa(self, text: str, num_pairs: int) -> List[dict]:
        key_sents = self._top_sentences(text, k=max(6, num_pairs * 2))
        pairs = []
        for idx, sent in enumerate(key_sents[:num_pairs]):
            question = f"What is the main idea of: '{sent[:100]}'?"
            answer = sent
            pairs.append({"question": question, "answer": answer})
        return pairs

    def _local_flashcards(self, text: str, num_cards: int) -> List[dict]:
        key_phrases = extract_key_phrases(text, top_k=num_cards)
        cards = []
        for phrase in key_phrases:
            cards.append({"front": phrase, "back": f"Definition or explanation of '{phrase}'."})
        return cards

    def _rake_phrases(self, text: str, top_k: int = 15) -> List[str]:
        # Backward compatibility: delegate to local extractor without NLTK
        return extract_key_phrases(text, top_k=top_k)

    def _top_sentences(self, text: str, k: int = 10) -> List[str]:
        # Use same frequency-based scoring to pick top sentences
        sentences = split_sentences(clean_text(text))
        if not sentences:
            return []
        tokens_re = re.compile(r"[A-Za-z0-9']+")
        stopwords = {
            "a","about","above","after","again","against","all","am","an","and","any","are","as","at","be","because","been","before","being","below","between","both","but","by","can","could","did","do","does","doing","down","during","each","few","for","from","further","had","has","have","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","it","its","itself","just","let","me","more","most","my","myself","no","nor","not","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","same","she","should","so","some","such","than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","very","was","we","were","what","when","where","which","while","who","whom","why","with","would","you","your","yours","yourself","yourselves"
        }
        def sentence_tokens(s: str) -> list[str]:
            return [t.lower() for t in tokens_re.findall(s)]

        word_freq: Counter[str] = Counter()
        for s in sentences:
            for t in sentence_tokens(s):
                if t in stopwords or len(t) <= 2:
                    continue
                word_freq[t] += 1
        scores = []
        for idx, s in enumerate(sentences):
            toks = [t for t in sentence_tokens(s) if t not in stopwords and len(t) > 2]
            if not toks:
                scores.append((idx, 0.0))
                continue
            score = sum(word_freq[t] for t in toks) / (len(toks) ** 0.8)
            scores.append((idx, score))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        top_indices = sorted(i for i, _ in top)
        return [sentences[i] for i in top_indices]
