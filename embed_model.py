import torch
from sentence_transformers import SentenceTransformer

class TextEmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the text embedding model.

        Args:
            model_name (str): Name of the pre-trained model to use.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, convert_to_tensor=False):
        """
        Encode texts into embeddings.

        Args:
            texts (str or list[str]): Text(s) to encode.
            convert_to_tensor (bool): Whether to return tensors or numpy arrays.

        Returns:
            Embeddings as numpy arrays or tensors.
        """
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts.

        Args:
            text1 (str): First text.
            text2 (str): Second text.

        Returns:
            float: Cosine similarity score.
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return torch.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2), dim=0).item()

if __name__ == "__main__":
    # Example usage
    model = TextEmbeddingModel()
    texts = ["Hello world", "Hi there", "Goodbye"]
    embeddings = model.encode(texts)
    print("Embeddings shape:", embeddings.shape)
    print("Similarity between 'Hello world' and 'Hi there':", model.similarity("Hello world", "Hi there"))