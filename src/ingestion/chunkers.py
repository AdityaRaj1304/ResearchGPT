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
            chunk_text = self.tokenizer.decode(chunk_tokens)
            while len(self.tokenizer.encode(chunk_text, add_special_tokens=False)) > self.max_tokens:
                chunk_tokens = chunk_tokens[:-1]
                chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
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
            chunk_text = self.tokenizer.decode(chunk_tokens)
            while len(self.tokenizer.encode(chunk_text, add_special_tokens=False)) > self.max_tokens:
                chunk_tokens = chunk_tokens[:-1]
                chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
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
            
        # Fix 2: Pre-process monster sentences
        processed_sentences = []
        for sent in sentences:
            sent_tokens = self.tokenizer.encode(sent, add_special_tokens=False)
            if len(sent_tokens) > 500:
                for i in range(0, len(sent_tokens), 500):
                    slice_ids = sent_tokens[i:i+500]
                    sub_sent = self.tokenizer.decode(slice_ids)
                    processed_sentences.append(sub_sent)
            else:
                processed_sentences.append(sent)
        sentences = processed_sentences
            
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
            
        # Fix 3: Aggressive Micro-Chunk Merging
        final_chunks = chunks.copy()
        
        while True:
            merged_in_this_pass = False
            if len(final_chunks) <= 1:
                break
                
            for i in range(len(final_chunks)):
                toks = self.tokenizer.encode(final_chunks[i], add_special_tokens=False)
                if len(toks) < 100:
                    left_len = float('inf')
                    right_len = float('inf')
                    
                    if i > 0:
                        left_len = len(self.tokenizer.encode(final_chunks[i-1], add_special_tokens=False))
                    if i < len(final_chunks) - 1:
                        right_len = len(self.tokenizer.encode(final_chunks[i+1], add_special_tokens=False))
                        
                    if left_len == float('inf') and right_len == float('inf'):
                        break
                        
                    if left_len <= right_len:
                        if left_len + len(toks) <= 512:
                            final_chunks[i-1] = final_chunks[i-1] + " " + final_chunks[i]
                            final_chunks.pop(i)
                            merged_in_this_pass = True
                            break
                        elif right_len != float('inf') and right_len + len(toks) <= 512:
                            final_chunks[i] = final_chunks[i] + " " + final_chunks[i+1]
                            final_chunks.pop(i+1)
                            merged_in_this_pass = True
                            break
                    else:
                        if right_len + len(toks) <= 512:
                            final_chunks[i] = final_chunks[i] + " " + final_chunks[i+1]
                            final_chunks.pop(i+1)
                            merged_in_this_pass = True
                            break
                        elif left_len != float('inf') and left_len + len(toks) <= 512:
                            final_chunks[i-1] = final_chunks[i-1] + " " + final_chunks[i]
                            final_chunks.pop(i)
                            merged_in_this_pass = True
                            break
                            
            if not merged_in_this_pass:
                break
                
        # Final pass to strictly enforce <= 512 for Semantic Chunker
        strictly_final = []
        for c in final_chunks:
            toks = self.tokenizer.encode(c, add_special_tokens=False)
            if len(toks) > 512:
                for idx in range(0, len(toks), 512):
                    slice_ids = toks[idx:idx+512]
                    chunk_text = self.tokenizer.decode(slice_ids)
                    while len(self.tokenizer.encode(chunk_text, add_special_tokens=False)) > 512:
                        slice_ids = slice_ids[:-1]
                        chunk_text = self.tokenizer.decode(slice_ids)
                    strictly_final.append(chunk_text)
            else:
                strictly_final.append(c)
                
        return strictly_final
