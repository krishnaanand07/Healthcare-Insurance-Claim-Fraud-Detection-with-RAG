import sys
import os
import uuid
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.llm_service import llm_service
from rag.retriever import retrieve_relevant_documents

class ChatService:
    def __init__(self):
        # In-memory session store: investigation_id -> List[Dict[str, str]]
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, investigation_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(investigation_id, [])

    def add_message(self, investigation_id: str, role: str, content: str):
        if investigation_id not in self.sessions:
            self.sessions[investigation_id] = []
        self.sessions[investigation_id].append({"role": role, "content": content})
        # Keep only the last 10 messages to maintain bounded context
        if len(self.sessions[investigation_id]) > 10:
            self.sessions[investigation_id] = self.sessions[investigation_id][-10:]

    def process_chat(
        self,
        query: str,
        claim_data: Optional[Dict[str, Any]] = None,
        investigation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not investigation_id:
            investigation_id = str(uuid.uuid4())

        history = self.get_history(investigation_id)

        # Retrieve relevant RAG chunks based on question and claim context
        search_query = query
        if claim_data:
            diag = claim_data.get("diagnosis_code") or claim_data.get("Diagnosis_Code")
            proc = claim_data.get("procedure_code") or claim_data.get("Procedure_Code")
            if diag or proc:
                search_query += f" Diagnosis {diag} Procedure {proc}"

        rag_results = retrieve_relevant_documents(search_query, top_k=4)

        docs = rag_results.get("documents", [])
        metas = rag_results.get("metadata", [])
        scores = rag_results.get("scores", [])

        retrieved_sources = []
        for i in range(len(docs)):
            m = metas[i] if i < len(metas) else {}
            retrieved_sources.append({
                "document": docs[i][:300],
                "filename": m.get("filename", "knowledge_base.md"),
                "section": m.get("section", "General"),
                "score": scores[i] if i < len(scores) else 0.0
            })

        # Generate LLM reply
        reply = llm_service.generate_chat_reply(
            query=query,
            claim_data=claim_data,
            rag_context=rag_results,
            history=history
        )

        # Record messages in memory
        self.add_message(investigation_id, "user", query)
        self.add_message(investigation_id, "assistant", reply)

        return {
            "reply": reply,
            "retrieved_sources": retrieved_sources,
            "investigation_id": investigation_id
        }

chat_service = ChatService()
