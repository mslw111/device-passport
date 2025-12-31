# agent_r2v3.py
import random
from llm_client import LLMClient
from tools import evaluate_r2v3_risk

class AgentR2V3:
    def __init__(self): 
        self.name = "R2v3_Wearables"
        self.llm = LLMClient("openai") # or gemini

    def round1(self, dev):
        # Extract watch specific flags
        flags = dev.get("r2v3_flags", {})
        specs = dev.get("specs", {})
        
        # New "Watch-Specific" Prompt
        prompt = (
            f"Review Smartwatch Condition. Model: {dev.get('brand')} {dev.get('model')}. "
            f"Case: {specs.get('case_size_mm')}mm. "
            f"Cosmetics: {flags.get('cosmetic_grade')}. "
            f"Defect: {flags.get('primary_defect')}. "
            "Evaluate strictly based on R2v3 Wearable standards. "
            "Focus on: 1. Bio-hazard (sweat/skin cells), 2. Battery safety, 3. Data destruction on eSIM."
            "Format: NARRATIVE, RISK RATING (Low/Med/High), UNCERTAINTY."
        )
        
        txt = self.llm.complete(prompt)
        return {
            "agent": self.name, 
            "round": 1, 
            "text": txt, 
            "risk_rating": "High" if "Fail" in flags.get("cosmetic_grade") else "Low", 
            "uncertainty": 0.2
        }

    def round2(self, ops):
        txt = self.llm.complete(f"Critique these smartwatch assessments: {ops}")
        return {"agent": self.name, "round": 2, "text": txt}
