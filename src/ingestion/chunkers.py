import msvc_runtime
import nltk
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class WholeDocChunker:
    def chunk(self, text):
        return [text]

class FixedSizeChunker:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def chunk(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

class OverlapChunker:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", max_tokens=512, overlap=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.step = max_tokens - overlap

    def chunk(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        if len(tokens) <= self.max_tokens:
            return [self.tokenizer.decode(tokens)]
            
        # Standard sliding window chunking
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + self.max_tokens >= len(tokens):
                break
            i += self.step
        return chunks

class SemanticChunker:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", max_tokens=512):
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = SentenceTransformer(model_name)
        self.max_tokens = max_tokens
        
        # Matches formats like "1. Introduction", "I. INTRODUCTION", "Introduction"
        self.header_regex = re.compile(r'^\s*(\d+\.?|[IVXLCDM]+\.?)?\s*(Abstract|Introduction|Methodology|Experiments|Results|Conclusion|Discussion|Related Work)\b', re.IGNORECASE)

    def chunk(self, text):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []
            
        embeddings = self.encoder.encode(sentences, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1])
            similarities.append(sim)
            
        if similarities:
            threshold = np.percentile(similarities, 20)
        else:
            threshold = 0
            
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_tokens = len(self.tokenizer.encode(sentences[0], add_special_tokens=False))
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            is_header = bool(self.header_regex.match(sentence))
            sim_drop = similarities[i-1] < threshold if i-1 < len(similarities) else False
            exceeds_cap = (current_chunk_tokens + sentence_tokens) > 500
            
            # Bounded Safety Window
            if exceeds_cap or is_header or (sim_drop and current_chunk_tokens >= 150):
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        # Post-Processing Micro-Chunk Merger
        final_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            tokens = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            
            if tokens < 100:
                if i < len(chunks) - 1:
                    next_tokens = len(self.tokenizer.encode(chunks[i+1], add_special_tokens=False))
                    if tokens + next_tokens <= 512:
                        chunks[i+1] = chunk + " " + chunks[i+1]
                    elif final_chunks and len(self.tokenizer.encode(final_chunks[-1], add_special_tokens=False)) + tokens <= 512:
                        final_chunks[-1] += " " + chunk
                    else:
                        chunks[i+1] = chunk + " " + chunks[i+1]
                else:
                    if final_chunks and len(self.tokenizer.encode(final_chunks[-1], add_special_tokens=False)) + tokens <= 512:
                        final_chunks[-1] += " " + chunk
                    else:
                        final_chunks[-1] += " " + chunk
            else:
                final_chunks.append(chunk)
            i += 1
            
        # Fix any > 512 chunks by splitting them
        enforced = []
        for c in final_chunks:
            toks = self.tokenizer.encode(c, add_special_tokens=False)
            if len(toks) > 512:
                sub_sentences = nltk.sent_tokenize(c)
                cur = []
                cur_len = 0
                for s in sub_sentences:
                    s_len = len(self.tokenizer.encode(s, add_special_tokens=False))
                    if cur_len + s_len > 512 and cur:
                        enforced.append(" ".join(cur))
                        cur = [s]
                        cur_len = s_len
                    else:
                        cur.append(s)
                        cur_len += s_len
                if cur:
                    enforced.append(" ".join(cur))
            else:
                enforced.append(c)
                
        really_final = []
        for c in enforced:
            if len(self.tokenizer.encode(c, add_special_tokens=False)) < 100 and really_final:
                if len(self.tokenizer.encode(really_final[-1], add_special_tokens=False)) + len(self.tokenizer.encode(c, add_special_tokens=False)) <= 512:
                    really_final[-1] += " " + c
                else:
                    really_final.append(c)
            else:
                really_final.append(c)
                
        return really_final
