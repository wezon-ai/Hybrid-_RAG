## Retriever_openai.py

import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from typing import List, Union
from rerank import reranking
import openai

# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')
OpenAI_API_KEY = os.getenv('OpenAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = OpenAI_API_KEY

class Hybrid_search():
    """
    A class for performing hybrid search using dense and sparse embeddings.
    """

    def __init__(self) -> None:
        """
        Initialize the Hybrid_search object with OpenAI's dense embedding model and a sparse embedding model.
        """
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

    def get_openai_embedding(self, text: str) -> List[float]:
        """
        Get dense embedding using OpenAI's embedding model.
        """
        try:
            response = openai.embeddings.create(
                input=text, 
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            
            # Dense embedding shape and sample values for inspection
            logger.info(f"Dense Embedding Shape: {len(embedding)}")
            logger.info(f"Dense Embedding Values (Sample): {embedding[:10]}")
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return []

    def metadata_filter(self, file_names: Union[str, List[str]]) -> models.Filter:
        """
        Create a metadata filter based on the file names provided.
        """
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

        return models.Filter(
            must=[file_name_condition]
        )
    
    def query_hybrid_search(self, query, metadata_filter=None, limit=5):
        """
        Perform a hybrid search using dense and sparse embeddings.
        """
        # OpenAI의 dense embedding 모델 사용
        dense_query = self.get_openai_embedding(query)
        if not dense_query:
            logger.error("Failed to retrieve dense embedding. Aborting search.")
            return []

        # Sparse embedding 모델 사용
        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

        # Sparse embedding shape and values for inspection
        logger.info(f"Sparse Embedding Indices: {sparse_query.indices[:10]}")
        logger.info(f"Sparse Embedding Values (Sample): {sparse_query.values[:10]}")
        
        results = self.qdrant_client.query_points(
            collection_name=Collection_Name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                    using="sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=limit,
                ),
            ],
            query_filter=metadata_filter,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )
        
        documents = [point.payload['text'] for point in results.points[:limit]]

        return documents

    # def query_hybrid_search(self, query, metadata_filter=None, limit=5):
    #     """
    #     Perform a hybrid search using dense and sparse embeddings.
    #     """
    #     # OpenAI의 dense embedding 모델 사용
    #     dense_query = self.get_openai_embedding(query)
    #     if not dense_query:
    #         logger.error("Failed to retrieve dense embedding. Aborting search.")
    #         return []

    #     # Sparse embedding 모델 사용
    #     sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

    #     # Sparse embedding shape and values for inspection
    #     logger.info(f"Sparse Embedding Indices: {sparse_query.indices[:10]}")
    #     logger.info(f"Sparse Embedding Values (Sample): {sparse_query.values[:10]}")
        
    #     results = self.qdrant_client.query_points(
    #         collection_name=Collection_Name,
    #         prefetch=[
    #             models.Prefetch(
    #                 query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
    #                 using="sparse",
    #                 limit=limit,
    #             ),
    #             models.Prefetch(
    #                 query=dense_query,
    #                 using="dense",
    #                 limit=limit,
    #             ),
    #         ],
    #         query_filter=metadata_filter,
    #         query=models.FusionQuery(fusion=models.Fusion.RRF),
    #     )
        
    #     # 'text'와 'source' 정보 포함하여 반환
    #     documents = [{"text": point.payload['text'], "source": point.payload.get('source', 'No source')} for point in results.points[:limit]]

    #     return documents

if __name__ == '__main__':    
    search = Hybrid_search()
    
    # 사용자가 영어로 쿼리를 입력

    # query = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
    query = "What are the potential hazards when documenting electrical hazards in high locations?"
    # 파일 이름에 대한 메타데이터 필터 설정
    file_names = "ADM_04-00-003"
    metadata_filter = search.metadata_filter(file_names)
    
    # 하이브리드 검색 수행
    results = search.query_hybrid_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for query: {query}")
    
    # 결과가 있는지 확인 후 리랭킹
    if results:
        reranked_documents = reranking().rerank_documents(query, results)
        logger.info(f"Reranked Documents: {reranked_documents}")

    # if results:
    #     # 리랭킹 수행 (텍스트만 전달)
    #     reranked_texts = reranking().rerank_documents(query, [doc['text'] for doc in results])[:2]
        
    #     # 리랭킹된 텍스트와 원래의 `results` 매칭하여 `text`와 `source` 출력
    #     reranked_documents = [
    #         next(doc for doc in results if doc['text'] == text) for text in reranked_texts
    #     ]
        
    #     for idx, document in enumerate(reranked_documents, start=1):
    #         text_content = document.get('text', 'No text found')
    #         source_info = document.get('source', 'No source found')
    #         logger.info(f"Document {idx} (Source: {source_info}): {text_content}")

    else:
        logger.info("No documents found for reranking.")
