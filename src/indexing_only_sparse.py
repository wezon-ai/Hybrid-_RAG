import logging
from dotenv import load_dotenv
import os
import json
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# Load environment variables
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Sparse_Collection_Name')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseIndexing:
    """
    A class for indexing documents using only sparse embeddings in Qdrant.
    """

    def __init__(self) -> None:
        self.data_path = "./data_/processed_nodes.json"
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(url=Qdrant_URL, api_key=Qdrant_API_KEY)
        self.metadata = []
        self.documents = []
        logging.info("SparseIndexing object initialized.")

    def load_nodes(self, input_file):
        with open(input_file, 'r') as file:
            self.nodes = json.load(file)

        for node in self.nodes:
            self.metadata.append(node['metadata'])
            self.documents.append(node['text'])

        logging.info(f"Loaded {len(self.nodes)} nodes from JSON file.")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if not self.qdrant_client.collection_exists(collection_name=Collection_Name):
            self.qdrant_client.create_collection(
                collection_name=Collection_Name,
                vectors_config={},  # 빈 vectors_config 전달
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        ),
                    )
                }
            )
            logging.info(f"Created sparse collection '{Collection_Name}' in Qdrant.")



    def create_sparse_vector(self, text):
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")

    def documents_insertion(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            sparse_vector = self.create_sparse_vector(doc)

            point = models.PointStruct(
                id=i,
                vector={'sparse': sparse_vector},
                payload={'text': doc, **metadata}
            )
            points.append(point)

        if points:
            self.qdrant_client.upsert(
                collection_name=Collection_Name,
                points=points
            )
            logging.info(f"Upserted {len(points)} points with sparse vectors into Qdrant.")

if __name__ == '__main__':
    sparse_indexing = SparseIndexing()
    sparse_indexing.load_nodes(sparse_indexing.data_path)
    sparse_indexing.client_collection()
    sparse_indexing.documents_insertion()
