## indexing_only_dense.py

import logging
from dotenv import load_dotenv
import os
import json
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import openai

# Load environment variables
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Dense_Collection_Name')
OpenAI_API_KEY = os.getenv('OpenAI_API_KEY')

# Set OpenAI API key
openai.api_key = OpenAI_API_KEY

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenseIndexing:
    """
    A class for indexing documents using only dense embeddings in Qdrant.
    """

    def __init__(self) -> None:
        self.data_path = "./data_/processed_nodes.json"
        self.qdrant_client = QdrantClient(url=Qdrant_URL, api_key=Qdrant_API_KEY)
        self.metadata = []
        self.documents = []
        logging.info("DenseIndexing object initialized.")

    def load_nodes(self, input_file):
        with open(input_file, 'r') as file:
            self.nodes = json.load(file)

        for node in self.nodes:
            self.metadata.append(node['metadata'])
            self.documents.append(node['text'])

        logging.info(f"Loaded {len(self.nodes)} nodes from JSON file.")

    def client_collection(self):
        # 기존 컬렉션이 있을 경우 삭제 (필요에 따라 주석 처리 가능)
        if self.qdrant_client.collection_exists(collection_name=Collection_Name):
            self.qdrant_client.delete_collection(collection_name=Collection_Name)
            logging.info(f"Deleted existing collection '{Collection_Name}' in Qdrant.")
        
        # 새 컬렉션 생성
        self.qdrant_client.create_collection(
            collection_name=Collection_Name,
            vectors_config={
                'dense': models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE,
                )
            }
        )
        logging.info(f"Created dense collection '{Collection_Name}' in Qdrant.")

    def create_dense_vector(self, text):
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            dense_vector = response.data[0].embedding
            return dense_vector

        except Exception as e:
            logging.error(f"Error generating OpenAI embedding: {e}")
            return None

    def documents_insertion(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            dense_embedding = self.create_dense_vector(doc)
            if dense_embedding is None:
                continue

            point = models.PointStruct(
                id=i,
                vector={'dense': dense_embedding},
                payload={'text': doc, **metadata}
            )
            points.append(point)

        if points:
            self.qdrant_client.upsert(
                collection_name=Collection_Name,
                points=points
            )
            logging.info(f"Upserted {len(points)} points with dense vectors into Qdrant.")

            # # 강제 인덱싱 실행
            # self.qdrant_client.force_index(collection_name=Collection_Name)
            # logging.info(f"Forced indexing for collection '{Collection_Name}'")

if __name__ == '__main__':
    dense_indexing = DenseIndexing()
    dense_indexing.load_nodes(dense_indexing.data_path)
    dense_indexing.client_collection()
    dense_indexing.documents_insertion()
