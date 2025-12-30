import os
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    def __init__(self, provider="openai"):
        self.provider = provider
        if provider == "gemini":
            self.key = os.getenv("GEMINI_API_KEY")
            from google import genai
            self.client = genai.Client(api_key=self.key) if self.key else None
        elif provider == "openai":
            self.key = os.getenv("OPENAI_API_KEY")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key) if self.key else None
    def complete(self, prompt):
        if not self.client: return "MOCK AGI ANALYSIS: (Keys missing in .env)"
        try:
            if self.provider == "gemini": return self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
            if self.provider == "openai": return self.client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}]).choices[0].message.content
        except Exception as e: return f"Error: {e}"
