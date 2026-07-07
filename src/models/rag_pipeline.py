# Self-healing Windows DLL fix (keeps PyTorch from crashing on startup)
try:
    import msvc_runtime
except ImportError:
    pass

import os
import chromadb
from chromadb.utils import embedding_functions

class ResearchGPTRetriever:
    """
    Phase 1 Baseline Retriever: Connects to your local ChromaDB vector store
    and executes semantic similarity searches over your 1,000 arXiv papers.
    """
    def __init__(self, db_path="data/vector_store", collection_name="arxiv_papers"):
        print(f"Connecting to ChromaDB database at: {db_path}...")
        if not os.path.exists(db_path):
            raise FileNotFoundError("Vector database directory not found. Please run build_embeddings.py first.")
            
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Must match the exact embedding model used during indexing
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.ef
        )
        print(f"Successfully connected! Total indexed papers available: {self.collection.count()}\n")

    def retrieve_papers(self, query: str, top_k: int = 4, similarity_threshold: float = 0.20):
        """
        Embeds the user query on the fly and retrieves top-k semantic matches.
        Converts ChromaDB L2/Cosine distances back into intuitive 0.0 - 1.0 similarity scores.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_docs = []
        # Parse the nested arrays returned by ChromaDB
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text_content = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to Cosine Similarity score (1.0 - distance)
            similarity_score = 1.0 - distance
            
            # Filter out low-relevance noise
            if similarity_score >= similarity_threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "title": metadata.get("title", "Unknown Title"),
                    "published": metadata.get("published", "Unknown Date"),
                    "categories": metadata.get("categories", ""),
                    "pdf_url": metadata.get("pdf_url", ""),
                    "content": text_content,
                    "score": round(similarity_score, 4)
                })
                
        return retrieved_docs


class ResearchGPTGenerator:
    """
    Part 2 Context Engineering: Formats retrieved document payloads into a structured
    context window and applies strict anti-hallucination system prompts.
    """
    @staticmethod
    def build_context_window(retrieved_docs: list) -> str:
        """Compiles candidate dictionaries into a cleanly labeled text block for the LLM."""
        if not retrieved_docs:
            return "NO RELEVANT ACADEMIC PAPERS FOUND IN THE INDEX."
            
        context_sections = []
        for idx, doc in enumerate(retrieved_docs, 1):
            section = (
                f"[Paper {idx}] Title: {doc['title']}\n"
                f"ID: {doc['id']} | Published: {doc['published']} | Categories: {doc['categories']}\n"
                f"URL: {doc['pdf_url']}\n"
                f"Content: {doc['content']}\n"
                f"Relevance Score: {doc['score']}\n"
            )
            context_sections.append(section)
            
        return "\n---\n".join(context_sections)

    @classmethod
    def generate_prompt(cls, user_query: str, retrieved_docs: list) -> dict:
        """
        Constructs the complete payload ready to be sent to an LLM (OpenAI, Groq, or Ollama).
        """
        context_text = cls.build_context_window(retrieved_docs)
        
        system_instruction = (
            "You are ResearchGPT, an elite AI research assistant and academic synthesizer. "
            "Your task is to answer the user's query strictly and exclusively using the provided academic paper abstracts below.\n\n"
            "MANDATORY OPERATIONAL RULES:\n"
            "1. Ground all technical claims directly in the provided context block. Do not use external training knowledge.\n"
            "2. Always cite your sources by referencing the exact Paper Title or [Paper X] designation in your text.\n"
            "3. If the provided context does not contain sufficient information to answer the query, state explicitly: "
            "'Based on the indexed academic corpus, I do not have enough literature to answer this query.' Do not guess.\n"
            "4. Maintain an objective, rigorous, and scholarly tone."
        )
        
        user_message = (
            f"USER RESEARCH QUERY:\n{user_query}\n\n"
            f"RETRIEVED ACADEMIC LITERATURE CONTEXT:\n{context_text}\n\n"
            "Please provide a comprehensive, synthesized answer based strictly on the literature above."
        )
        
        return {
            "system": system_instruction,
            "user": user_message,
            "retrieved_count": len(retrieved_docs)
        }


def run_interactive_test():
    """Terminal loop to test vector retrieval and prompt assembly in real time."""
    retriever = ResearchGPTRetriever()
    generator = ResearchGPTGenerator()
    
    print("="*65)
    print(" 🔎 ResearchGPT: Phase 1 Baseline Retrieval Pipeline Active")
    print(" Type your research question below (or 'exit' to quit)")
    print("="*65)
    
    while True:
        try:
            query = input("\n[Enter Research Query]: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting search loop. Goodbye!")
                break
                
            # 1. Retrieve candidate papers from ChromaDB
            docs = retriever.retrieve_papers(query=query, top_k=4, similarity_threshold=0.15)
            
            # 2. Build the LLM prompt payload
            prompt_payload = generator.generate_prompt(user_query=query, retrieved_docs=docs)
            
            # 3. Print Retrieval Metrics & Results
            print(f"\n--- 📊 RETRIEVAL RESULTS (Found {len(docs)} matching papers) ---")
            if not docs:
                print("No papers crossed the similarity threshold. Try broadening your terms.")
            else:
                for idx, doc in enumerate(docs, 1):
                    print(f"{idx}. [Score: {doc['score']}] {doc['title']}")
                    print(f"   Categories: {doc['categories']} | URL: {doc['pdf_url']}")
                
            # 4. Preview the Context Window
            print("\n--- 📝 ASSEMBLED LLM CONTEXT WINDOW PREVIEW ---")
            preview_text = prompt_payload["user"][:600] + "\n... [Context truncated for terminal preview] ..." if len(prompt_payload["user"]) > 600 else prompt_payload["user"]
            print(preview_text)
            print("="*65)
            
        except KeyboardInterrupt:
            print("\nExiting search loop.")
            break

if __name__ == "__main__":
    run_interactive_test()