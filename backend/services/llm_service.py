import os
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from schemas.ai_schemas import AIInvestigationAnalysis, SupportingEvidence
from rag.prompts import INVESTIGATION_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT

class NVIDIALLMService:
    def __init__(self):
        self.base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.model_name = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
        self.client = None
        self._init_client()

    def _init_client(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.model_name = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
        if self.api_key:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
                print(f"[NVIDIALLMService] Initialized client with model '{self.model_name}'")
            except Exception as e:
                print(f"[NVIDIALLMService] Error initializing OpenAI client: {e}")
                self.client = None
        else:
            self.client = None

    def generate_investigation_report(
        self,
        claim_data: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        rag_context: Dict[str, Any]
    ) -> AIInvestigationAnalysis:
        """
        Generate structured AI Investigation Analysis using NVIDIA Hosted API and Pydantic validation.
        """
        # Formulate context for prompt
        docs = rag_context.get("documents", [])
        metas = rag_context.get("metadata", [])
        scores = rag_context.get("scores", [])

        formatted_context_items = []
        for i in range(len(docs)):
            src = metas[i].get("filename", "knowledge_base") if i < len(metas) else "knowledge_base"
            sec = metas[i].get("section", "") if i < len(metas) else ""
            scr = scores[i] if i < len(scores) else 0.0
            formatted_context_items.append(f"--- Source: {src} (Section: {sec}, Relevance Score: {scr}) ---\n{docs[i]}")

        retrieved_str = "\n\n".join(formatted_context_items) if formatted_context_items else "No relevant knowledge-base documents retrieved."

        user_message = (
            f"CLAIM DATA:\n{json.dumps(claim_data, indent=2)}\n\n"
            f"MACHINE LEARNING PREDICTION:\n{json.dumps(ml_prediction, indent=2)}\n\n"
            f"RETRIEVED RAG EVIDENCE / KNOWLEDGE BASE CONTEXT:\n{retrieved_str}\n\n"
            "Generate the comprehensive AI Investigation Report in valid JSON format."
        )

        if not self.client:
            self._init_client()

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": INVESTIGATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                )
                content = response.choices[0].message.content
                parsed_dict = self._parse_json_response(content)
                if parsed_dict:
                    # Enforce ML probability if present
                    ml_prob = ml_prediction.get("fraud_probability")
                    if ml_prob is None and "risk_score" in ml_prediction:
                        ml_prob = round(ml_prediction["risk_score"] / 100.0, 2)
                    if ml_prob is not None:
                        parsed_dict["fraud_probability"] = float(ml_prob)

                    return AIInvestigationAnalysis(**parsed_dict)
            except Exception as e:
                print(f"[NVIDIALLMService] API Call failed: {e}. Generating fallback structured report.")

        # Fallback structured report if NVIDIA API is unconfigured or fails
        return self._build_fallback_report(claim_data, ml_prediction, rag_context)

    def generate_chat_reply(
        self,
        query: str,
        claim_data: Optional[Dict[str, Any]],
        rag_context: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> str:
        """
        Generate conversational reply for the RAG Assistant.
        """
        docs = rag_context.get("documents", [])
        metas = rag_context.get("metadata", [])
        
        context_sources = []
        for i, d in enumerate(docs):
            src = metas[i].get("filename", "Doc") if i < len(metas) else "Doc"
            context_sources.append(f"[{src}]: {d}")

        context_str = "\n\n".join(context_sources) if context_sources else "No knowledge base documents retrieved."

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        # Append lightweight history
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        prompt_content = f"Question: {query}\n\n"
        if claim_data:
            prompt_content += f"Claim Details: {json.dumps(claim_data)}\n\n"
        prompt_content += f"Retrieved Knowledge Evidence:\n{context_str}"

        messages.append({"role": "user", "content": prompt_content})

        if not self.client:
            self._init_client()

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"[AI Assistant Notice] Unable to reach NVIDIA LLM API ({e}). Based on retrieved context:\n\n{context_str[:400]}..."

        # Fallback chat response
        return f"Based on the retrieved knowledge base:\n\n{context_str[:500]}"

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        cleaned = text.strip()
        # Remove markdown code block fences if present
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = re.sub(r"^```[a-z]*", "", cleaned)
                cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Fallback regex search for outer JSON object
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _build_fallback_report(
        self,
        claim_data: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        rag_context: Dict[str, Any]
    ) -> AIInvestigationAnalysis:
        prob = ml_prediction.get("fraud_probability")
        if prob is None and "risk_score" in ml_prediction:
            prob = round(ml_prediction["risk_score"] / 100.0, 2)
        prob = prob if prob is not None else 0.50

        risk_level = ml_prediction.get("risk_level", "MEDIUM")
        insights = ml_prediction.get("insights", ["Claim submitted requires verification."])

        docs = rag_context.get("documents", [])
        metas = rag_context.get("metadata", [])
        
        evidence_items = []
        for i in range(min(len(docs), 3)):
            src = metas[i].get("filename", "knowledge_base.md") if i < len(metas) else "knowledge_base.md"
            evidence_items.append(SupportingEvidence(
                source=src,
                relevance="High" if i == 0 else "Medium",
                text=docs[i][:200] + "..."
            ))

        return AIInvestigationAnalysis(
            summary=f"Claim evaluated with {risk_level} risk level ({int(prob*100)}% probability). Key anomalies detected in claim metrics.",
            risk_level=risk_level,
            fraud_probability=prob,
            key_risk_factors=insights,
            claim_analysis=f"The machine learning model flagged this claim with risk score {int(prob*100)}. Diagnostic and procedure combinations require investigator audit.",
            supporting_evidence=evidence_items,
            recommended_investigation_steps=[
                "Verify itemized medical records against provider billing logs",
                "Contact patient to confirm service date and location",
                "Check provider licensing and NPI registration status"
            ],
            limitations=[
                "NVIDIA LLM offline or fallback mode active",
                "Requires direct human verification of physical medical chart"
            ]
        )

llm_service = NVIDIALLMService()
