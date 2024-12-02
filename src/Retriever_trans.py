## Retriever_trans.py

import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding, TextEmbedding
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
        Initialize the Hybrid_search object with dense and sparse embedding models and a Qdrant client.
        """
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

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
    
    # def translate_query_to_english(self, query: str) -> str:
    #     """
    #     Translate a Korean query to English using OpenAI API.
    #     """
    #     try:
    #         response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system", "content": "Translate the following Korean text to English:"},
    #                 {"role": "user", "content": query}
    #             ]
    #         )
    #         translated_query = response.choices[0].message.content.strip()
    #         return translated_query
        
    #     except Exception as e:
    #         logger.error(f"Error translating query: {e}")
    #         return query  # Return the original query if translation fails
        
    def translate_query_to_english(self, query: str) -> str:
        """
        Translate a Korean query to English using OpenAI API, ensuring only the translation is returned.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Translate the following Korean text to English. Please only provide the translation without any additional explanation."},
                    {"role": "user", "content": query}
                ]
            )
            translated_query = response.choices[0].message.content.strip()
            print(f"Translated query: {translated_query}")  # 번역된 쿼리를 출력하여 확인
            return translated_query
        
        except Exception as e:
            logger.error(f"Error translating query: {e}")
            return query  # Return the original query if translation fails


    # def query_hybrid_search(self, query, metadata_filter=None, limit=5):
    #     """
    #     Perform a hybrid search using dense and sparse embeddings.
    #     """
    #     # 먼저 한국어 쿼리를 영어로 번역
    #     translated_query = self.translate_query_to_english(query)
    #     logger.info(f"Translated query: {translated_query}")

    #     # Embed the query using the dense embedding model
    #     dense_query = list(self.embedding_model.embed([translated_query]))[0].tolist()

    #     # Embed the query using the sparse embedding model
    #     sparse_query = list(self.sparse_embedding_model.embed([translated_query]))[0]

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
        
    #     documents = [point.payload['text'] for point in results.points]

    #     return documents

    ## 각각 임베딩 형태 확인
    def query_hybrid_search(self, query, metadata_filter=None, limit=5):
        """
        Perform a hybrid search using dense and sparse embeddings.
        """
        # 먼저 한국어 쿼리를 영어로 번역
        translated_query = self.translate_query_to_english(query)
        logger.info(f"Translated query: {translated_query}")

        # Embed the query using the dense embedding model
        dense_query = list(self.embedding_model.embed([translated_query]))[0].tolist()
        logger.info(f"Dense Embedding Shape: {len(dense_query)}")  # Dense 임베딩 길이 확인
        logger.info(f"Dense Embedding: {dense_query}")  # Dense 임베딩 값 확인

        # Embed the query using the sparse embedding model
        sparse_query = list(self.sparse_embedding_model.embed([translated_query]))[0]
        logger.info(f"Sparse Embedding Indices: {sparse_query.indices}")  # Sparse 임베딩 인덱스 확인
        logger.info(f"Sparse Embedding Values: {sparse_query.values}")  # Sparse 임베딩 값 확인

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
        
        documents = [point.payload['text'] for point in results.points]
        return documents



if __name__ == '__main__':
    search = Hybrid_search()
    # 사용자가 한국어로 쿼리를 입력
    query = "유리 섬유 I형 Gr2의 평균 온도 204℃에서의 열전도도는 얼마입니까?"
    
    # 파일 이름에 대한 메타데이터 필터 설정
    file_names = "Technical Guidelines for Calculating and Installing Discharge Capacity of Safety Valves, etc."
    metadata_filter = search.metadata_filter(file_names)
    
    # 하이브리드 검색 수행
    results = search.query_hybrid_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for query: {query}")
    
    # 결과가 있는지 확인 후 리랭킹
    if results:
        reranked_documents = reranking().rerank_documents(query, results)
        logger.info(f"Reranked Documents: {reranked_documents}")
    else:
        logger.info("No documents found for reranking.")
