## indexing.py

'''
이 코드는 Qdrant 벡터 데이터베이스에 문서를 임베딩(embedding)하여 삽입하기 위한 파이프라인을 구축합니다. 
dense(밀집) 및 sparse(희소) 임베딩을 모두 생성하여, Qdrant에 하이브리드 인덱스를 만듭니다.
'''

## 1. 모듈 임포트

import logging
from dotenv import load_dotenv
import os
import json
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm import tqdm

'''
logging: 로그 메시지를 기록합니다.
dotenv: .env 파일에서 환경 변수를 로드합니다.
os: 파일 경로 및 환경 변수 접근에 사용됩니다.
json: JSON 파일에서 데이터를 로드합니다.
fastembed: 밀집(dense) 및 희소(sparse) 임베딩을 생성합니다.
qdrant_client: Qdrant 데이터베이스에 데이터 삽입 및 인덱스 관리에 사용됩니다.
tqdm: 진행 상황을 시각화하는 프로그레스 바를 제공합니다.
'''

## 2. 환경 변수 로드
# Load environmental variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

'''
load_dotenv(): .env 파일에서 환경 변수를 로드합니다.
API 키 및 Qdrant URL은 환경 변수로 관리됩니다.
(보안상 중요한 정보는 .env 파일에 따로 관리합니다.)
'''

## 3. QdrantIndexing 클래스

class QdrantIndexing:
    """
    A class for indexing documents using Qdrant vector database.
    """

    def __init__(self) -> None:
        """
        Initialize the QdrantIndexing object.
        """
        self.data_path = "./data_/translated_nodes.json"
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
                            url=Qdrant_URL,
                            api_key=Qdrant_API_KEY)
        self.metadata = []
        self.documents = []
        logging.info("QdrantIndexing object initialized.")

        '''
        데이터 경로: nodes.json 파일 경로입니다.
        밀집 임베딩 모델: all-MiniLM-L6-v2 모델을 사용합니다.
        희소 임베딩 모델: BM42 접근을 사용하는 Qdrant/bm42-all-minilm-l6-v2-attentions 모델입니다.
        Qdrant 클라이언트: Qdrant API와 상호작용하는 클라이언트를 초기화합니다.
        메타데이터 및 문서 리스트 초기화: 나중에 문서를 삽입할 때 사용됩니다.
        '''

    ## 4. 노드 로드 (load_nodes)

    def load_nodes(self, input_file):
        """
        Load nodes from a JSON file and extract metadata and documents.

        Args:
            input_file (str): The path to the JSON file.
        """
        with open(input_file, 'r') as file:
            self.nodes = json.load(file)

        for node in self.nodes:
            self.metadata.append(node['metadata'])
            self.documents.append(node['text'])

        logging.info(f"Loaded {len(self.nodes)} nodes from JSON file.")

        '''
        nodes.json 파일에서 데이터를 불러옵니다.
        각 노드의 메타데이터와 텍스트를 별도의 리스트에 저장합니다.
        '''

## 5. Qdrant 컬렉션 생성 (client_collection)
    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if not self.qdrant_client.collection_exists(collection_name=f"{Collection_Name}"): 
            self.qdrant_client.create_collection(
                collection_name= Collection_Name,
                vectors_config={
                     'dense': models.VectorParams(
                         size=384,
                         distance = models.Distance.COSINE,
                     )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                              index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            logging.info(f"Created collection '{Collection_Name}' in Qdrant vector database.")

            '''
            컬렉션 확인 및 생성: Qdrant에 동일한 컬렉션이 없을 경우 새로 생성합니다.
            밀집 벡터 설정: 벡터 차원은 384로 설정되고, 코사인 거리를 사용합니다.
            희소 벡터 설정: 희소 벡터는 BM42 인덱스로 관리됩니다.
            '''

## 6. 희소 벡터 생성 (create_sparse_vector)
    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using SPLADE.
        """
        # Generate the sparse vector using SPLADE model
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]

        # Check if embeddings has indices and values attributes
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")
        
        '''
        BM42 임베딩 생성: 희소 벡터를 생성합니다.
        벡터 객체가 indices와 values 속성을 포함하지 않으면 예외를 발생시킵니다.
        '''

## 7. 문서 삽입 (documents_insertion)
    def documents_insertion(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            # Generate both dense and sparse embeddings
            dense_embedding = list(self.embedding_model.embed([doc]))[0]
            sparse_vector = self.create_sparse_vector(doc)

            # Create PointStruct
            point = models.PointStruct(
                id=i,
                vector={
                    'dense': dense_embedding.tolist(),
                    'sparse': sparse_vector,
                },
                payload={
                    'text': doc,
                    **metadata  # Include all metadata
                }
            )
            points.append(point)

        # Upsert points
        self.qdrant_client.upsert(
            collection_name=Collection_Name,
            points=points
        )

        logging.info(f"Upserted {len(points)} points with dense and sparse vectors into Qdrant vector database.")

'''
각 문서에 대해 밀집 임베딩과 희소 임베딩을 생성합니다.
PointStruct 객체에 벡터와 메타데이터를 포함하여 Qdrant에 삽입합니다.
upsert: Qdrant에 데이터를 삽입하고, 기존 데이터가 있을 경우 업데이트합니다.
'''

## 8. 메인 실행 블록 
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    indexing = QdrantIndexing()
    indexing.load_nodes(indexing.data_path)
    indexing.client_collection()
    indexing.documents_insertion()

'''
로깅 설정: 로그 메시지를 출력합니다.
QdrantIndexing 객체 생성 및 메서드 호출:
노드 데이터 로드
Qdrant 컬렉션 생성
문서 삽입
'''

## 9. 주요 기능 요약
'''
밀집(dense) 및 희소(sparse) 벡터를 동시에 사용해 하이브리드 검색을 지원합니다.
Qdrant 벡터 데이터베이스에 데이터를 삽입하고 업데이트합니다.
.env 파일을 사용해 API 키와 URL을 안전하게 관리합니다.
'''

### 실행 방법
'''
.env 파일 생성:
Qdrant_API_KEY=your_api_key
Qdrant_URL=https://your-qdrant-instance-url
Collection_Name=your_collection_name

필요한 패키지 설치:
pip install qdrant-client fastembed tqdm python-dotenv

스크립트 실행:
python your_script_name.py
'''