"""
Lumina Board - CSV RAG Engine
Builds a searchable index from all CSV files using TF-IDF + cosine similarity.
No external vector database required — pure pandas + sklearn.
Grounding: all responses are anchored to actual CSV data rows.
"""

import os
import glob
import logging
import json
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("lumina.rag")


class CSVRagEngine:
    """
    CSV-based Retrieval Augmented Generation engine.
    
    Workflow:
      1. Load all CSV files from data directory
      2. Convert each row to a textual document (key: value pairs)
      3. Build a TF-IDF index across all documents
      4. At query time: find top-k most similar rows and return as context
    
    No hallucination: LLM only sees actual row data.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.documents: List[Dict] = []   # raw text documents
        self.tfidf_matrix = None
        self.is_built = False
        self._csv_metadata: Dict[str, Dict] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # INDEX BUILDING
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self):
        """Load all CSVs and build TF-IDF index."""
        logger.info(f"Building RAG index from {self.data_dir}")
        self.documents = []

        csv_paths = self._discover_csvs()
        if not csv_paths:
            logger.warning(f"No CSV files found in {self.data_dir}")
            self.is_built = True
            return

        for path in csv_paths:
            self._index_csv(path)

        if not self.documents:
            logger.warning("No documents to index.")
            self.is_built = True
            return

        # Build TF-IDF matrix
        texts = [d["text"] for d in self.documents]
        logger.info(f"Fitting TF-IDF on {len(texts)} documents from {len(csv_paths)} CSVs")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_built = True
        logger.info(f"RAG index built: {len(self.documents)} docs, matrix shape {self.tfidf_matrix.shape}")

    def _discover_csvs(self) -> List[str]:
        paths = set()
        for pattern in ["*.csv", "**/*.csv"]:
            for p in glob.glob(os.path.join(self.data_dir, pattern), recursive=True):
                paths.add(os.path.abspath(p))
        return sorted(paths)

    def _index_csv(self, path: str):
        """Convert CSV rows to searchable text documents."""
        try:
            df = pd.read_csv(path)
            source = os.path.basename(path).replace(".csv", "")
            df = df.fillna("unknown")
            rows_indexed = 0

            # Store CSV metadata
            self._csv_metadata[source] = {
                "path": path,
                "columns": list(df.columns),
                "row_count": len(df)
            }

            for idx, row in df.iterrows():
                # Build rich text representation of the row
                parts = [f"Dataset: {source}"]
                for col, val in row.items():
                    if str(val) not in ("unknown", "nan", ""):
                        parts.append(f"{col.replace('_', ' ')}: {val}")
                text = " | ".join(parts)

                self.documents.append({
                    "text": text,
                    "source": source,
                    "row_idx": idx,
                    "data": row.to_dict()
                })
                rows_indexed += 1

            logger.info(f"Indexed {rows_indexed} rows from {source}")

        except Exception as e:
            logger.error(f"Failed to index {path}: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # QUERYING
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        csv_filter: Optional[str] = None,
        min_score: float = 0.01
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Returns list of dicts with: text, source, row_idx, score, data
        """
        if not self.is_built:
            self.build_index()

        if not self.documents or self.tfidf_matrix is None:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply CSV filter
        if csv_filter:
            filter_name = csv_filter.replace(".csv", "")
            mask = np.array([
                1.0 if doc["source"] == filter_name else 0.0
                for doc in self.documents
            ])
            scores = scores * mask

        # Get top-k
        top_indices = scores.argsort()[::-1][:top_k * 3]  # over-fetch then filter

        results = []
        seen_rows = {}  # deduplicate: keep best score per (source, row_idx)

        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            doc = self.documents[idx]
            key = (doc["source"], doc["row_idx"])
            if key not in seen_rows or seen_rows[key]["score"] < score:
                seen_rows[key] = {**doc, "score": score}

        # Sort by score and return top_k
        results = sorted(seen_rows.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return results

    def query_by_filters(
        self,
        filters: Dict[str, str],
        csv_name: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Direct structured query: filter CSV rows by column values.
        Returns a DataFrame of matching rows.
        """
        csv_paths = self._discover_csvs()
        results = []

        for path in csv_paths:
            source = os.path.basename(path).replace(".csv", "")
            if csv_name and source != csv_name:
                continue
            try:
                df = pd.read_csv(path)
                mask = pd.Series([True] * len(df))
                for col, val in filters.items():
                    if col in df.columns:
                        mask &= df[col].astype(str).str.lower() == str(val).lower()
                matched = df[mask].head(limit)
                if len(matched) > 0:
                    matched["_source"] = source
                    results.append(matched)
            except Exception as e:
                logger.error(f"Filter query error on {path}: {e}")

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # STATUS
    # ──────────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "is_built": self.is_built,
            "total_documents": len(self.documents),
            "csv_files": list(self._csv_metadata.keys()),
            "csv_metadata": self._csv_metadata,
            "matrix_shape": list(self.tfidf_matrix.shape) if self.tfidf_matrix is not None else None
        }