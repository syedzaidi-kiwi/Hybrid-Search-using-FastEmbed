import os
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, NamedSparseVector, NamedVector, SparseVector, PointStruct, SparseIndexParams,
    SparseVectorParams, VectorParams, ScoredPoint
)
from transformers import AutoTokenizer
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding


# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="XXX", 
    api_key="XXX",
    timeout=120
)

# Paths
jsonl_directory = "/Users/kiwitech/Documents/labelled_wasde_json"

# Models
sparse_model_name = "prithvida/Splade_PP_en_v1"
dense_model_name = "mixedbread-ai/mxbai-embed-large-v1"
sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)

# Function to read texts from JSONL files
def read_texts_from_jsonl(directory: str) -> List[str]: 
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                for line in file:
                    data = json.loads(line)
                    texts.append(data.get("full_text", ""))
    return texts

# Read texts from JSONL files
jsonl_texts = read_texts_from_jsonl(jsonl_directory)

# Create embeddings
def make_sparse_embedding(texts: List[str]):
    return list(sparse_model.embed(texts, batch_size=32))

def make_dense_embedding(texts: List[str]):
    return list(dense_model.embed(texts))

sparse_embeddings = make_sparse_embedding(jsonl_texts)
dense_embeddings = make_dense_embedding(jsonl_texts)

# Create and upload the points to Qdrant
collection_name = "quasar_jsonl"
qdrant_client.create_collection(
    collection_name,
    vectors_config={
        "text-dense": VectorParams(
            size=1024,  # BGE-Large-en-v1.5 embedding size
            distance=Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=True,
            )
        )
    },
)

def make_points(texts: List[str], sparse_embeddings: List[SparseEmbedding], dense_embeddings: List[np.ndarray]) -> List[PointStruct]:
    points = []
    for idx, (text, sparse_embedding, dense_embedding) in enumerate(zip(texts, sparse_embeddings, dense_embeddings)):
        sparse_vector = SparseVector(indices=sparse_embedding.indices.tolist(), values=sparse_embedding.values.tolist())
        point = PointStruct(
            id=idx,
            payload={"text": text},
            vector={
                "text-sparse": sparse_vector,
                "text-dense": dense_embedding,
            },
        )
        points.append(point)
    return points

points = make_points(jsonl_texts, sparse_embeddings, dense_embeddings)
qdrant_client.upsert(collection_name, points)

print("Metadata extraction and JSONL file processing complete.")
