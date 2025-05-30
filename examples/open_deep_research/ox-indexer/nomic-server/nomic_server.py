#!/usr/bin/env python3
"""
OpenAI-compatible REST API server for nomic-ai/nomic-embed-code
Supports multi-GPU loading across CUDA devices 0-3

Requirements:
pip install torch transformers fastapi uvicorn numpy accelerate
"""

import os
import time
import asyncio
from typing import List, Union, Optional
from datetime import datetime
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Nomic Embed Code API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device_map = None

# Request/Response models matching OpenAI's API
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "nomic-embed-code"
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nomic-ai"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    # Ensure both tensors are on the same device
    device = token_embeddings.device
    attention_mask = attention_mask.to(device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

async def load_model():
    """Load the model across multiple GPUs"""
    global model, tokenizer, device_map
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-code')
    
    print("Loading model across GPUs...")
    # Define device map for 4 GPUs
    # Auto device map will handle the distribution better
    device_map = "auto"
    
    # Alternative: If auto doesn't work well, use balanced strategy
    # from accelerate import infer_auto_device_map
    # device_map = infer_auto_device_map(model, max_memory={0: "10GB", 1: "10GB", 2: "10GB", 3: "10GB"})
    
    # Load model with device map
    model = AutoModel.from_pretrained(
        'nomic-ai/nomic-embed-code',
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print("Model loaded successfully across GPUs!")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    await load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "model": "nomic-embed-code"}

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models"""
    return ModelList(
        data=[
            ModelInfo(
                id="nomic-embed-code",
                created=int(time.time())
            )
        ]
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings - OpenAI compatible endpoint"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    # Add code prefix for better code embedding
    texts = [f"search_document: {text}" for text in texts]
    
    try:
        # Tokenize
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors='pt'
        )
        
        # Don't manually move to device - let the model handle it
        # The model with device_map will handle device placement automatically
        
        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Get the last hidden state
        if hasattr(model_output, 'last_hidden_state'):
            last_hidden_state = model_output.last_hidden_state
        else:
            last_hidden_state = model_output[0]
        
        # Mean pooling with proper device handling
        attention_mask = encoded_input['attention_mask']
        embeddings = mean_pooling((last_hidden_state,), attention_mask)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to list
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        # Calculate token usage
        total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
        
        # Format response
        data = [
            EmbeddingData(
                embedding=embedding,
                index=i
            ) for i, embedding in enumerate(embeddings_list)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU.")
    else:
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs")
        if gpu_count < 4:
            print(f"WARNING: Expected 4 GPUs but found {gpu_count}")
        
        for i in range(min(4, gpu_count)):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8001)
