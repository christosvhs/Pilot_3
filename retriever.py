import os
import numpy as np
import torch
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
DEFAULT_MODEL_DIR = "/scratch/NLU/cvlachos/SCQA/Models_of_Samu_XLSR_finetuning/fine_tuned_text_implicit_model_doc2dialsplit_0_500"
CHROMA_DB_PATH = "/scratch/NLU/cvlachos/SCQA/Samu_XLSR_finetuning/data/chroma_db"
COLLECTION_NAME = "propositions_VS"

_device = "cuda" if torch.cuda.is_available() else "cpu"


class Retriever:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, n_results: int = 10):
        self.n_results = n_results

        self.model = SentenceTransformer(LABSE_MODEL_NAME).to(_device)
        self.model.tokenizer.truncation_side = "left"

        if "best.pt" in os.listdir(model_dir):
            checkpoint = torch.load(
                os.path.join(model_dir, "best.pt"),
                map_location=torch.device(_device),
            )
            self.model.load_state_dict(checkpoint["state_dict"])

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LABSE_MODEL_NAME
        )
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
        )

    def generate_embedding(self, text: list[str]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            features = self.model.tokenize(text)
            features = batch_to_device(features, _device)
            embedding = self.model(features)["sentence_embedding"]
        return embedding.detach().cpu().squeeze().numpy()

    def retrieve(self, dialog_history: list[str], n_results: int = None) -> dict:
        text = [" [SEP] ".join(dialog_history)]
        embedding = self.generate_embedding(text)
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results or self.n_results,
        )
