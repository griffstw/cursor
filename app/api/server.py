from __future__ import annotations

import io
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.agents.document_agent import DocumentAgent
from app.ingestion.loader import sniff_and_load
from app.processing.keyphrases import extract_key_phrases

app = FastAPI(title="Document Summarization and Study Guide Agent", version="0.1.0")
agent = DocumentAgent()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize")
async def summarize(files: List[UploadFile] = File(...)):
    texts: List[str] = []
    for f in files:
        data = await f.read()
        text = sniff_and_load(f.filename or "file", data)
        if text:
            texts.append(text)
    joined = "\n\n".join(texts)
    summary = agent.summarize(joined)
    key_ideas = extract_key_phrases(joined)
    return JSONResponse({"summary": summary, "key_ideas": key_ideas})


@app.post("/study")
async def study(files: List[UploadFile] = File(...)):
    texts: List[str] = []
    for f in files:
        data = await f.read()
        text = sniff_and_load(f.filename or "file", data)
        if text:
            texts.append(text)
    joined = "\n\n".join(texts)
    result = agent.study_guide(joined)
    return JSONResponse(result)
