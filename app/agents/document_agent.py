from __future__ import annotations

from typing import Dict, List, Tuple

from app.llm.client import LLMClient
from app.processing.text import clean_text, split_sentences, chunk_sentences
from app.processing.keyphrases import extract_key_phrases


class DocumentAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def extract_key_ideas(self, text: str, top_k: int = 12) -> List[str]:
        # RAKE-style extraction without external downloads
        return extract_key_phrases(text, top_k=top_k)

    def summarize(self, text: str, max_words: int = 150) -> str:
        text = clean_text(text)
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, max_tokens=450)
        partials: List[str] = []
        for ch in chunks:
            partials.append(self.llm.summarize(ch, max_words=120))
        merged = "\n".join(partials)
        final = self.llm.summarize(merged, max_words=max_words)
        return final

    def study_guide(self, text: str, num_qa: int = 8, num_cards: int = 12) -> Dict[str, List[dict]]:
        text = clean_text(text)
        summary = self.summarize(text, max_words=160)
        qa = self.llm.generate_qa(summary, num_pairs=num_qa)
        cards = self.llm.generate_flashcards(summary, num_cards=num_cards)
        return {"summary": summary, "qa": qa, "flashcards": cards}
