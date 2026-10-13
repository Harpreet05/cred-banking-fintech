from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal
from uuid import uuid4

from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from api.structured_logging import write_request_log

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import run_conversation
from rag.chunks import build_chunk_sets
from rag.embeddings import create_embeddings


CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"


class AskRequest(BaseModel):
	query: str = Field(min_length=1, max_length=4000)
	conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class AgentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	intent: Literal["rag", "status"]
	answer: str = Field(min_length=1)
	fallback: bool
	record_id: str | None
	escalation_score: float | None = Field(ge=0, le=1)
	escalate: bool | None


class AskResponse(BaseModel):
	conversation_id: str
	response: AgentResponse


class AddDocumentRequest(BaseModel):
	doc_id: str = Field(pattern=r"^[A-Za-z0-9_-]{3,64}$")
	title: str = Field(min_length=1, max_length=200)
	text: str = Field(min_length=1, max_length=50000)


class AddDocumentResponse(BaseModel):
	doc_id: str
	chunks_indexed: dict[str, int]


app = FastAPI(title="Cred Banking Support Agent")


@app.middleware("http")
async def log_requests(request: Request, call_next):
	trace_id = request.headers.get("X-Trace-Id") or str(uuid4())
	start_time = perf_counter()
	request_text = ""
	status_code = 500
	try:
		request_body = await request.body()
		request_text = request_body.decode("utf-8", errors="replace") if request_body else ""
		if not request_text:
			request_text = request.url.query
		try:
			json_body = json.loads(request_text)
			request_text = json.dumps(json_body, ensure_ascii=False)
		except Exception:
			pass
		request._body = request_body
		response = await call_next(request)
		status_code = response.status_code
		response.headers["X-Trace-Id"] = trace_id
		return response
	finally:
		write_request_log(
			method=request.method,
			route=request.url.path,
			request_text=request_text,
			trace_id=trace_id,
			started_at=start_time,
			finished_at=perf_counter(),
			status_code=status_code,
		)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
	return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
	conversation_id = request.conversation_id or str(uuid4())
	result = run_conversation(request.query, conversation_id)
	return AskResponse(
		conversation_id=conversation_id,
		response=AgentResponse.model_validate(result["response"]),
	)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
	client = PersistentClient(path=str(CHROMA_DIRECTORY))
	collection_names = {collection.name for collection in client.list_collections()}
	for name in ("knowledge_base_fixed", "knowledge_base_sentence"):
		if name in collection_names:
			existing = client.get_collection(name).get(
				where={"doc_id": request.doc_id},
				include=["metadatas"],
			)
			if existing["ids"]:
				raise HTTPException(status_code=409, detail="doc_id is already indexed")

	document = request.model_dump()
	chunk_sets = build_chunk_sets([document])
	try:
		create_embeddings([document], persist_directory=str(CHROMA_DIRECTORY))
	except Exception as error:
		raise HTTPException(status_code=500, detail="Document could not be embedded") from error

	return AddDocumentResponse(
		doc_id=request.doc_id,
		chunks_indexed={strategy: len(chunks) for strategy, chunks in chunk_sets.items()},
	)
