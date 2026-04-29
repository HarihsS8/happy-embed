# happy-embed

A simple text embedding model for LLMs using sentence-transformers.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```python
from embed_model import TextEmbeddingModel

model = TextEmbeddingModel()
embeddings = model.encode(["Hello world", "Hi there"])
similarity = model.similarity("Hello world", "Hi there")
```

## Features

- Load pre-trained embedding models
- Encode single or multiple texts
- Compute cosine similarity between texts 
