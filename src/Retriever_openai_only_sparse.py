# ## SparseSearch.py

# import logging
# from dotenv import load_dotenv
# import os
# from fastembed import SparseTextEmbedding
# from qdrant_client import QdrantClient, models
# from qdrant_client.http.models import SparseVector, NamedSparseVector  # NamedSparseVector 추가
# from typing import List, Union


# # Load environment variables
# load_dotenv()
# Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
# Qdrant_URL = os.getenv('Qdrant_URL')
# Collection_Name = os.getenv('Sparse_Collection_Name')

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SparseSearch:
#     """
#     A class for performing sparse search using BM42 embeddings.
#     """

#     def __init__(self) -> None:
#         self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
#         self.qdrant_client = QdrantClient(
#             url=Qdrant_URL,
#             api_key=Qdrant_API_KEY,
#             timeout=30
#         )

#     def metadata_filter(self, file_names: Union[str, List[str]]) -> models.Filter:
#         """
#         Create a metadata filter based on the file names provided.
#         """
#         if isinstance(file_names, str):
#             file_name_condition = models.FieldCondition(
#                 key="file_path",
#                 match=models.MatchValue(value=file_names)
#             )
#         else:
#             file_name_condition = models.FieldCondition(
#                 key="file_path",
#                 match=models.MatchAny(any=file_names)
#             )

#         return models.Filter(must=[file_name_condition])

    

#     # def query_sparse_search(self, query, metadata_filter=None, limit=5):
#     #     """
#     #     Perform a sparse search using BM42 embeddings.
#     #     """
#     #     sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

#     #     results = self.qdrant_client.search(
#     #         collection_name=Collection_Name,
#     #         query_vector=NamedSparseVector(
#     #             name="sparse",  # 컬렉션 생성 시 정의된 이름
#     #             vector={"indices": sparse_query.indices.tolist(), "values": sparse_query.values.tolist()}
#     #         ),
#     #         query_filter=metadata_filter,
#     #         limit=limit,
#     #     )

#     #     documents = [point.payload['text'] for point in results]
#     #     return documents
    
#     def query_sparse_search(self, query, metadata_filter=None, limit=5):
#         """
#         Perform a sparse search using BM42 embeddings.
#         """
#         sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

#         results = self.qdrant_client.search(
#             collection_name=Collection_Name,
#             query_vector=NamedSparseVector(
#                 name="sparse",  # 컬렉션 생성 시 정의된 이름
#                 vector={"indices": sparse_query.indices.tolist(), "values": sparse_query.values.tolist()}
#             ),
#             query_filter=metadata_filter,
#             limit=limit,
#         )

#         # 검색 결과에서 payload 정보 반환
#         documents = [{"text": point.payload.get('text'), "source": point.payload.get('source')} for point in results]
#         return documents




# # if __name__ == '__main__':
# #     search = SparseSearch()
# #     query = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
# #     file_names = "ADM_04-00-003"
# #     metadata_filter = search.metadata_filter(file_names)
    
# #     results = search.query_sparse_search(query, metadata_filter)
# #     logger.info(f"Found {len(results)} results for sparse query: {query}")
    
# #     # # 검색된 문서 내용 출력
# #     # for idx, document in enumerate(results, start=1):
# #     #     logger.info(f"Document {idx}: {document}")

# #  # 검색된 문서의 내용과 source 필드를 출력
# #     for idx, document in enumerate(results, start=1):
# #         text_content = document.get('text', 'No text found')
# #         source_info = document.get('source', 'No source found')
# #         logger.info(f"Document {idx} (Source {source_info}): {text_content}")
# if __name__ == '__main__':
#     search = SparseSearch()
#     query = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
#     file_names = "ADM_04-00-003"
#     metadata_filter = search.metadata_filter(file_names)
    
#     results = search.query_sparse_search(query, metadata_filter)
#     logger.info(f"Found {len(results)} results for sparse query: {query}")
    
#     # 검색된 문서의 내용과 source 필드를 출력
#     for idx, document in enumerate(results, start=1):
#         text_content = document.get('text', 'No text found')
#         source_info = document.get('source', 'No source found')
#         logger.info(f"Document {idx} (Source {source_info}): {text_content}")

import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import NamedSparseVector
from typing import List, Union
from rerank import reranking  # 리랭킹 함수 가져오기

# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Sparse_Collection_Name')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseSearch:
    def __init__(self) -> None:
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

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

    def query_sparse_search(self, query, metadata_filter=None, limit=5):
        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]
        results = self.qdrant_client.search(
            collection_name=Collection_Name,
            query_vector=NamedSparseVector(
                name="sparse",
                vector={"indices": sparse_query.indices.tolist(), "values": sparse_query.values.tolist()}
            ),
            query_filter=metadata_filter,
            limit=limit,
        )
        documents = [{"text": point.payload.get('text'), "source": point.payload.get('source')} for point in results]
        return documents

if __name__ == '__main__':
    search = SparseSearch()
    query = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
    file_names = "ADM_04-00-003"
    metadata_filter = search.metadata_filter(file_names)
    
    results = search.query_sparse_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for sparse query: {query}")
    
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
