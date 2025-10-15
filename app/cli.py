from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from app.agents.document_agent import DocumentAgent
from app.ingestion.loader import sniff_and_load
from app.processing.keyphrases import extract_key_phrases

app = typer.Typer(help="Document summarization and study guide agent CLI")
console = Console()


@app.command()
def summarize(path: Path, max_words: int = 160) -> None:
    agent = DocumentAgent()
    if path.is_dir():
        texts = []
        for p in path.rglob("*"):
            if p.is_file():
                texts.append(sniff_and_load(p.name, p.read_bytes()))
        text = "\n\n".join(texts)
    else:
        text = sniff_and_load(path.name, path.read_bytes())
    summary = agent.summarize(text, max_words=max_words)
    key_ideas = extract_key_phrases(text)
    console.rule("Summary")
    console.print(summary)
    console.rule("Key Ideas")
    for i, idea in enumerate(key_ideas, 1):
        console.print(f"{i}. {idea}")


@app.command()
def study(path: Path, num_qa: int = 8, num_cards: int = 12) -> None:
    agent = DocumentAgent()
    if path.is_dir():
        texts = []
        for p in path.rglob("*"):
            if p.is_file():
                texts.append(sniff_and_load(p.name, p.read_bytes()))
        text = "\n\n".join(texts)
    else:
        text = sniff_and_load(path.name, path.read_bytes())
    result = agent.study_guide(text, num_qa=num_qa, num_cards=num_cards)
    console.rule("Summary")
    console.print(result["summary"])
    console.rule("Q&A")
    for i, qa in enumerate(result["qa"], 1):
        console.print(f"Q{i}: {qa['question']}")
        console.print(f"A{i}: {qa['answer']}")
    console.rule("Flashcards")
    for i, fc in enumerate(result["flashcards"], 1):
        console.print(f"{i}. {fc['front']} -> {fc['back']}")


if __name__ == "__main__":
    app()
