## document_pre_processing.py

'''
이 코드는 문서 데이터를 로드, 전처리, 분할한 다음 JSON 파일로 저장하는 파이프라인을 구축하는 모듈입니다. 
'''
## 1. 모듈 임포트

'''
os: 파일 및 디렉터리 조작에 사용됩니다.
json: 노드 데이터를 JSON 형식으로 저장합니다.
re: 정규 표현식을 사용해 텍스트를 전처리합니다.

llama_index 라이브러리의 클래스들:
Document: 각 문서를 나타내는 객체입니다.
SentenceSplitter: 문서를 청크(Chunk) 단위로 나누는 클래스입니다.
SimpleDirectoryReader: 디렉터리에서 파일을 읽어 문서 객체로 변환합니다.
'''

import os
import json
import re
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader


## 2. CustomTransformation 클래스

'''
역할:
문서 텍스트를 전처리하여 소문자화, 공백 정리, 특수문자 제거 작업을 수행합니다.
새롭게 전처리된 텍스트와 원래의 메타데이터를 포함한 Document 객체를 생성합니다.

__call__ 메서드:
이 클래스는 호출 가능한 객체처럼 사용되며, documents 리스트를 입력받아 각 문서를 변환합니다.
'''

class CustomTransformation:
    def __call__(self, documents):
        transformed_documents = []
        for doc in documents:
            transformed_content = doc.get_content().lower()
            transformed_content = re.sub(r'\s+', ' ', transformed_content)
            transformed_content = re.sub(r'[^\w\s]', '', transformed_content)
            transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
        return transformed_documents

# class CustomTransformation:
#     def __call__(self, documents):
#         transformed_documents = []
#         for doc in documents:
#             transformed_content = doc.get_content()

#             # 인코딩 문제 해결을 위해 UTF-8로 디코딩 시도
#             if isinstance(transformed_content, bytes):
#                 transformed_content = transformed_content.decode('utf-8', 'ignore')
            
#             # 소문자화, 공백 정리, 특수문자 제거
#             transformed_content = transformed_content.lower()
#             transformed_content = re.sub(r'\s+', ' ', transformed_content)
#             transformed_content = re.sub(r'[^\w\s]', '', transformed_content)

#             transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
#         return transformed_documents



## 3. Sentence_Splitter_docs_into_nodes 함수

'''
역할:
문서 객체를 "노드(Node)"로 분할합니다. 각 노드는 텍스트의 작은 청크와 메타데이터를 포함합니다.
chunk_size와 chunk_overlap을 설정해 청크의 크기와 겹침을 조절합니다.
'''

def Sentence_Splitter_docs_into_nodes(all_documents):
    try:
        splitter = SentenceSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        nodes = splitter.get_nodes_from_documents(all_documents)

        return nodes

    except Exception as e:
        print(f"Error splitting documents into nodes: {e}")
        return []


## 4. save_nodes 함수

'''
역할:
노드 리스트를 JSON 파일로 저장합니다.
디렉터리가 존재하지 않을 경우 자동 생성하고, 각 노드를 딕셔너리로 변환한 후 JSON 파일로 기록합니다.
'''

def save_nodes(nodes, output_file):
    try:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert the TextNode objects to dictionaries
        nodes_dict = [node.dict() for node in nodes]

        with open(output_file, 'w') as file:
            json.dump(nodes_dict, file, indent=4)
        print(f"Saved nodes to {output_file}")
    except Exception as e:
        print(f"Error saving nodes to file: {e}")

from llama_index.readers.file import PyMuPDFReader
from pathlib import Path

# PDF 파일을 PyMuPDFReader로 읽어오기
def load_pdfs_with_pymupdf(directory_path):
    try:
        # PyMuPDFReader 객체 생성
        pdf_reader = PyMuPDFReader()
        documents = []

        # 디렉터리에서 PDF 파일을 검색하여 읽어옴
        for pdf_file in Path(directory_path).glob("*.pdf"):
            # PyMuPDFReader를 사용하여 PDF 파일 읽기
            document = pdf_reader.load_data(pdf_file)
            documents.extend(document)

        print(f"Loaded {len(documents)} documents from PDF files.")
        return documents

    except Exception as e:
        print(f"Error loading PDF files: {e}")
        return []

import pdfplumber
from pathlib import Path

def load_pdfs_with_pdfplumber(directory_path):
    documents = []
    for pdf_file in Path(directory_path).glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # 텍스트를 UTF-8로 변환
                    text = text.encode('utf-8', 'ignore').decode('utf-8')
                    documents.append(Document(text=text, metadata={"file_name": pdf_file.name}))
    return documents


## 5. 메인 실행 블록

'''
__name__ == '__main__' 블록:

스크립트가 직접 실행될 때만 실행됩니다.
디렉터리에서 문서 데이터를 로드하고, 커스텀 전처리를 적용하며, 노드로 분할한 후 JSON 파일로 저장합니다.
문서 로드:
SimpleDirectoryReader를 사용해 \data 디렉터리에서 문서를 불러옵니다.

전처리 및 노드 생성:
각 문서에 전처리를 수행한 후 노드로 분할합니다.

노드 저장:
노드를 JSON 파일로 저장합니다.
'''
# documents = SimpleDirectoryReader(input_dir=os.path.abspath("./data_")).load_data()
# print(f"Loaded {len(documents)} documents")

if __name__ == '__main__':
    try:
        # Load data from directory
        # reader = SimpleDirectoryReader(input_dir="./data_", encoding="utf-8-sig")
        # documents = reader.load_data()
        # documents = SimpleDirectoryReader(input_dir=os.path.abspath("./data_")).load_data()
        # PDF 파일이 있는 디렉터리 설정
        input_dir = "./"
        documents = load_pdfs_with_pymupdf(input_dir)

        
        print(f"Loaded {len(documents)} documents")

        if documents:
            # Apply custom transformation
            custom_transform = CustomTransformation()
            documents = custom_transform(documents)

            # Split documents into nodes
            nodes = Sentence_Splitter_docs_into_nodes(documents)

            print(f"Created {len(nodes)} nodes")

            # Save nodes to a single JSON file
            output_file = os.path.abspath("./data_/nodes.json")
            save_nodes(nodes, output_file)

        else:
            print("No documents to process.")

    except Exception as e:
        print(f"Error processing documents: {e}")
        

## 6. 실행 과정 요약
'''
디렉터리에서 문서 로드:
SimpleDirectoryReader로 문서 데이터를 불러옵니다.

문서 전처리:
텍스트를 소문자로 변환하고 불필요한 공백 및 특수 문자를 제거합니다.

문서를 노드로 분할:
문서를 1500자 크기의 청크로 분할하며, 200자씩 중복된 부분이 포함됩니다.

노드 저장:
생성된 노드들을 JSON 형식으로 파일에 저장합니다.
'''