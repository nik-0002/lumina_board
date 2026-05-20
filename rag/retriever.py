import faiss
import numpy as np
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """
    Retrieval Augmented Generation system for contextual farming data
    Integrates: Government APIs, Pest Surveillance, Weather, Best Practices
    """
    
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 index_path="./data/vector_indices"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index_path = index_path
        self.faiss_index = None
        self.documents = []
        self.metadata = []
    
    def build_index(self, documents: List[Dict]):
        """
        Build FAISS vector index from documents
        Each document should have: text, metadata (crop, region, type, etc.)
        """
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.documents = documents
        self.metadata = [doc.get('metadata', {}) for doc in documents]
        
        # Save index
        faiss.write_index(self.faiss_index, f"{self.index_path}/rag_index.faiss")
    
    def retrieve_context(self, crop: str, region: str, current_stage: str = None, 
                        query_type: str = "pest_management", k: int = 5) -> Dict:
        """
        Retrieve relevant context for content generation
        """
        # Build query
        query = f"{crop} {region} {current_stage or ''} {query_type}"
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # Compile results
        context = {
            'government_advisory': '',
            'best_practice': '',
            'pest_alerts': [],
            'weather_insight': '',
            'product_recommendations': [],
            'sources': []
        }
        
        for idx in indices[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                meta = self.metadata[idx]
                
                if meta.get('type') == 'government_advisory':
                    context['government_advisory'] = doc['text']
                elif meta.get('type') == 'best_practice':
                    context['best_practice'] = doc['text']
                elif meta.get('type') == 'pest_alert':
                    context['pest_alerts'].append(doc['text'])
                elif meta.get('type') == 'weather':
                    context['weather_insight'] = doc['text']
                
                context['sources'].append(meta.get('source', 'Unknown'))
        
        return context