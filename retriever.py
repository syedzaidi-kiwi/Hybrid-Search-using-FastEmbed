import numpy as np
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, NamedVector, NamedSparseVector, SparseVector, ScoredPoint
import os
import json
from typing import List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, NamedSparseVector, NamedVector, SparseVector, PointStruct, SparseIndexParams,
    SparseVectorParams, VectorParams, ScoredPoint
)
import requests
from transformers import AutoTokenizer
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from PyPDF2 import PdfReader

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="XXX", 
    api_key="XXX",
)

# Models
sparse_model_name = "prithvida/Splade_PP_en_v1"
dense_model_name = "mixedbread-ai/mxbai-embed-large-v1"
sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)

# Hybrid search function
def search(query_text: str, top_k: int = 10):
    query_sparse_vectors: List[SparseEmbedding] = list(sparse_model.embed([query_text]))
    query_dense_vector: List[np.ndarray] = list(dense_model.embed([query_text]))

    search_results = qdrant_client.search_batch(
        collection_name="quasar_jsonl",
        requests=[
            SearchRequest(
                vector=NamedVector(
                    name="text-dense",
                    vector=query_dense_vector[0],
                ),
                limit=top_k,
                with_payload=True,
            ),
            SearchRequest(
                vector=NamedSparseVector(
                    name="text-sparse",
                    vector=SparseVector(
                        indices=query_sparse_vectors[0].indices.tolist(),
                        values=query_sparse_vectors[0].values.tolist(),
                    ),
                ),
                limit=top_k,
                with_payload=True,
            ),
        ],
    )

    return search_results

# Example usage
query = "What are the projected U.S. wheat production and yield for 2024/25?"
results = search(query, top_k=5)

def rank_list(search_result: List[ScoredPoint]):
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

def rrf(rank_lists, alpha=60, default_rank=1000):
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
    item_to_index = {item: idx for idx, item in enumerate(all_items)}
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)
    sorted_indices = np.argsort(-rrf_scores)
    sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]
    return sorted_items

dense_rank_list, sparse_rank_list = rank_list(results[0]), rank_list(results[1])
rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])

def find_point_by_id(client: QdrantClient, collection_name: str, rrf_rank_list: List[Tuple[int, float]]):
    return client.retrieve(collection_name=collection_name, ids=[item[0] for item in rrf_rank_list])

retrieved_points = find_point_by_id(qdrant_client, "quasar_jsonl", rrf_rank_list)
data = retrieved_points[0].payload['text'][:1000]


import cohere
co = cohere.Client(api_key="XXX")

response = co.chat(
  model="command-r-plus",
  message=f"""Use this data:{retrieved_points[0].payload['text'][:1000]} to answer this query:{query}. Provide a detailed summary. Donot give any placeholders.
  When user asks any question related or current year then give answers from 2022 and onwards and donot give answers from before 2022."""
)

print(response.text)

