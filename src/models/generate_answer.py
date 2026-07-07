# Self-healing Windows DLL fix
try:
    import msvc_runtime
except ImportError:
    pass

import os
import time
from openai import OpenAI
from advanced_retriever import AdvancedResearchRetriever

class ResearchGPTEngine:
    """
    The Master Orchestrator: Connects Phase 2 Hybrid Retrieval with 
    Modern Autoregressive LLM Synthesis.
    """
    def __init__(self, provider="ollama"):
        print(f"Initializing ResearchGPT Engine [Inference Provider: {provider.upper()}]...")
        self.retriever = AdvancedResearchRetriever()
        self.provider = provider.lower()
        
        # Configure the universal LLM Client based on your preferred backend
        if self.provider == "ollama":
            # Points to local Ollama running on Windows (http://localhost:11434)
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3" # Ensure you ran `ollama pull llama3` in PowerShell
        elif self.provider == "groq":
            # Points to Groq's high-speed Llama-3 API
            groq_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            self.model_name = "llama-3.3-70b-versatile"
        elif self.provider == "openai":
            # Points to standard OpenAI
            openai_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
            self.client = OpenAI(api_key=openai_key)
            self.model_name = "gpt-4o-mini"
        else:
            raise ValueError("Invalid provider. Choose from: 'ollama', 'groq', or 'openai'.")

    def format_context_window(self, docs: list) -> str:
        """Assembles the retrieved abstracts into a strict citation-ready block."""
        sections = []
        for idx, doc in enumerate(docs, 1):
            sections.append(
                f"[Paper {idx}] Title: {doc['title']}\n"
                f"URL: {doc['pdf_url']}\n"
                f"Abstract: {doc['text']}\n"
            )
        return "\n---\n".join(sections)

    def ask(self, query: str, top_k: int = 3, temperature: float = 0.1):
        """
        Executes the complete Phase 2 RAG cycle:
        Query -> Hybrid Search -> Cross-Encoder Rerank -> LLM Generation
        """
        start_time = time.time()
        
        # 1. Retrieve & Rerank (High Precision)
        print(f"\n[1/3] Executing Hybrid Search & MS-MARCO Reranking for: '{query}'...")
        top_papers = self.retriever.retrieve_and_rerank(query=query, final_top_k=top_k)
        
        # 2. Assemble Context
        print(f"[2/3] Assembling {len(top_papers)} verified papers into context window...")
        context_block = self.format_context_window(top_papers)
        
        system_prompt = (
            "You are ResearchGPT, an elite academic synthesis AI. "
            "Your job is to answer the user's question strictly using the provided literature abstracts below.\n"
            "RULES:\n"
            "1. Ground every claim directly in the provided text. Do not use training knowledge.\n"
            "2. Always cite sources inline using the format [Paper X] or by explicit paper title.\n"
            "3. If the context does not answer the question, state: 'Based on the indexed literature, I lack sufficient data to answer this.' Do not guess."
        )
        
        user_prompt = (
            f"RESEARCH QUERY:\n{query}\n\n"
            f"RETRIEVED ACADEMIC LITERATURE:\n{context_block}\n\n"
            "Synthesize a comprehensive, professional academic answer citing the papers above."
        )
        
        # 3. Generate Autoregressive Response
        print(f"[3/3] Streaming tokens from {self.model_name} (Temperature: {temperature})...\n")
        print("="*65)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=True # Stream tokens smoothly like ChatGPT!
        )
        
        full_answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_answer += token
                
        duration = round(time.time() - start_time, 2)
        print(f"\n\n" + "="*65)
        print(f"RAG Cycle Completed in {duration}s | Referenced {len(top_papers)} Papers")
        return full_answer

if __name__ == "__main__":
    # Ensure you install the openai SDK: `python -m pip install openai`
    
    # Select your provider here: "ollama" (local), "groq" (fast cloud), or "openai"
    engine = ResearchGPTEngine(provider="ollama") 
    
    engine.ask(query="What is Retrieval-Augmented Generation (RAG) and how does it work?")