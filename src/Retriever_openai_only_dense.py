## DenseSearch.py

import logging
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from typing import List, Union
import openai
from rerank import reranking  # 리랭킹 함수 가져오기

# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Dense_Collection_Name')
OpenAI_API_KEY = os.getenv('OpenAI_API_KEY')

# Set OpenAI API key
openai.api_key = OpenAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenseSearch:
    def __init__(self) -> None:
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

    def get_openai_embedding(self, text: str) -> List[float]:
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return []

    def metadata_filter(self, file_names: Union[str, List[str]]) -> models.Filter:
        if isinstance(file_names, str):
            file_name_condition = models.FieldCondition(
                key="file_path",
                match=models.MatchValue(value=file_names)
            )
        else:
            file_name_condition = models.FieldCondition(
                key="file_path",
                match=models.MatchAny(any=file_names)
            )
        return models.Filter(must=[file_name_condition])

    def query_dense_search(self, query, metadata_filter=None, limit=5):
        dense_query = self.get_openai_embedding(query)
        if not dense_query:
            logger.error("Failed to retrieve dense embedding. Aborting search.")
            return []

        results = self.qdrant_client.search(
            collection_name=Collection_Name,
            query_vector=models.NamedVector(name="dense", vector=dense_query),
            query_filter=metadata_filter,
            limit=limit,
        )
        documents = [{"text": point.payload.get('text'), "source": point.payload.get('source')} for point in results]
        return documents

if __name__ == '__main__':
    search = DenseSearch()
    # query = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
    query = "What are the potential hazards when documenting electrical hazards in high locations?"
    file_names = "ADM_04-00-003"
    metadata_filter = search.metadata_filter(file_names)
    
    results = search.query_dense_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for dense query: {query}")
    
    # 리랭킹 수행 후 상위 2개 문서 선택
    if results:
        reranked_texts = reranking().rerank_documents(query, [doc['text'] for doc in results])[:2]
        reranked_documents = [
            next(doc for doc in results if doc['text'] == text) for text in reranked_texts
        ]
        for idx, document in enumerate(reranked_documents, start=1):
            text_content = document.get('text', 'No text found')
            source_info = document.get('source', 'No source found')
            logger.info(f"Document {idx} (Source {source_info}): {text_content}")
    else:
        logger.info("No documents found for reranking.")
