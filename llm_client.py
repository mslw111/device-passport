import os
import streamlit as st
from dotenv import load_dotenv

# Try loading local .env (silently fails on cloud, which is fine)
load_dotenv()

class LLMClient:
    def __init__(self, provider="openai"):
        self.provider = provider
        
        # 1. Try Local Environment
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # 2. If missing, Try Streamlit Cloud Secrets
        if not self.gemini_key and "GEMINI_API_KEY" in st.secrets:
            self.gemini_key = st.secrets["GEMINI_API_KEY"]
        
        if not self.openai_key and "OPENAI_API_KEY" in st.secrets:
            self.openai_key = st.secrets["OPENAI_API_KEY"]

        # 3. Initialize Clients
        if provider == "gemini":
            if not self.gemini_key:
                print("❌ GEMINI_API_KEY not found in .env or Secrets.")
                self.client = None
            else:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=self.gemini_key)
                except Exception:
                    # Fallback for older library versions
                    import google.generativeai as genai
                    genai.configure(api_key=self.gemini_key)
                    self.client = genai.GenerativeModel('gemini-pro')

        elif provider == "openai":
            if not self.openai_key:
                print("❌ OPENAI_API_KEY not found in .env or Secrets.")
                self.client = None
            else:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_key)

    def complete(self, prompt):
        if not self.client: 
            return "⚠️ SYSTEM ERROR: API Keys missing. Check Streamlit Secrets."
        
        try:
            if self.provider == "gemini": 
                # Handle both library versions
                if hasattr(self.client, 'models'):
                    return self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
                else:
                    return self.client.generate_content(prompt).text
            
            if self.provider == "openai": 
                return self.client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}]).choices[0].message.content
        except Exception as e: 
            return f"⚠️ API ERROR: {str(e)}"
