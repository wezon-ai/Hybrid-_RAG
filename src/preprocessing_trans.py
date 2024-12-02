## preprocessing.py

import os
import json
import openai
import pymupdf4llm
import re
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path
import fitz  # PyMuPDF for total pages extraction

# OpenAI API Key 설정

# 파일 이름 번역 함수
def translate_filename(file_name):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 사용할 모델 지정
            messages=[
                {"role": "system", "content": "Translate the following file name from Korean to English:"},
                {"role": "user", "content": file_name}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during file name translation: {e}")
        return "[Translation Error]"

# 전처리 클래스 (CustomTransformation) - 번역된 텍스트에 적용
class CustomTransformation:
    def __call__(self, documents):
        transformed_documents = []
        for doc in documents:
            # 소문자화 및 공백 정리
            transformed_content = doc.get_content().lower()
            transformed_content = re.sub(r'\s+', ' ', transformed_content)
            
            # '|' 기호는 살리기 위해 다른 특수 문자만 제거
            transformed_content = re.sub(r'[^\w\s|]', '', transformed_content)

            # 유니코드 문제 처리
            transformed_content = self.clean_unicode(transformed_content)

            # 페이지 번호 제거
            transformed_content = self.remove_page_numbers(transformed_content)

            transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
        return transformed_documents

    def clean_unicode(self, text):
        try:
            # 유니코드 특수 문자 디코딩
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            print(f"Unicode decoding error: {e}")
        return text

    def remove_page_numbers(self, text):
        # "- 숫자" 형식의 페이지 번호 제거
        return re.sub(r'- \d+', '', text)

# 텍스트를 일정 길이로 나누는 함수 (토큰 수 제한을 고려하여 나눔)
def split_text(text, max_tokens=2000):
    sentences = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_length += len(sentence.split())
        if current_length <= max_tokens:
            current_chunk.append(sentence)
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence.split())

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# 텍스트 번역 함수 (청크 단위로 처리)
def translate_text(text_chunk):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 사용할 모델 지정
            messages=[
                {"role": "system", "content": "Translate the following text to English:"},
                {"role": "user", "content": text_chunk}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error during translation: {e}")
        return "[Translation Error]"

# 전체 텍스트를 청크로 나누고 번역 후 결합하는 함수
def translate_large_text(text):
    text_chunks = split_text(text)
    translated_chunks = [translate_text(chunk) for chunk in text_chunks]
    return "\n".join(translated_chunks)

# PDF 파일에서 텍스트와 테이블을 추출하고 번역 및 전처리하는 함수
def extract_translate_and_preprocess_text_by_page(file_path, translated_file_name):
    print(f"Processing {file_path}...")

    # PyMuPDF로 PDF 파일 열고 페이지 수 확인
    pdf_document = fitz.open(file_path)
    total_pages = pdf_document.page_count
    pdf_document.close()

    try:
        # 페이지별로 마크다운 형식으로 텍스트 추출
        page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

    translated_pages = []
    for page_num, chunk in enumerate(page_chunks, start=1):
        markdown_text = chunk['text']  # 마크다운 형식의 텍스트

        # 텍스트 번역 (마크다운 형식 유지)
        translated_markdown = translate_large_text(markdown_text)

        # 마크다운 형식을 유지한 채로 Document 객체로 저장
        processed_document = [Document(
            text=translated_markdown,
            metadata={
                "total_pages": total_pages,  # 총 페이지 수
                "file_path": translated_file_name,  # 번역된 파일 이름 사용
                "source": page_num  # 현재 페이지 번호
            }
        )]

        # 페이지 결과 저장
        translated_pages.extend(processed_document)

    return translated_pages

# 문서를 노드로 분할하는 함수
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

# 노드 리스트를 JSON 파일로 저장하는 함수
def save_nodes(nodes, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        nodes_dict = [node.dict() for node in nodes]

        with open(output_file, 'w') as file:
            json.dump(nodes_dict, file, indent=4)
        print(f"Saved nodes to {output_file}")
    except Exception as e:
        print(f"Error saving nodes to file: {e}")

# PDF 파일 로드 함수 (PyMuPDF4LLM를 사용하여 PDF 파일을 로드하고 처리)
def load_pdfs_with_pymupdf4llm(directory_path):
    try:
        documents = []
        for pdf_file in Path(directory_path).glob("*.pdf"):
            # 파일 이름 추출 및 번역
            original_file_name = pdf_file.stem  # 확장자 제거한 파일 이름 추출
            translated_file_name = translate_filename(original_file_name)  # 파일 이름 번역

            document = extract_translate_and_preprocess_text_by_page(pdf_file, translated_file_name)
            if document:
                documents.extend(document)

        print(f"Loaded and processed {len(documents)} documents from PDF files.")
        return documents

    except Exception as e:
        print(f"Error loading and processing PDF files: {e}")
        return []

# 메인 실행 블록
if __name__ == '__main__':
    try:
        input_dir = "./data_"
        documents = load_pdfs_with_pymupdf4llm(input_dir)

        if documents:
            # CustomTransformation 적용
            custom_transform = CustomTransformation()
            documents = custom_transform(documents)

            # Split documents into nodes
            nodes = Sentence_Splitter_docs_into_nodes(documents)

            print(f"Created {len(nodes)} nodes")

            # Save nodes to a single JSON file
            output_file = os.path.abspath("./data_/translated_nodes.json")
            save_nodes(nodes, output_file)

        else:
            print("No documents to process.")

    except Exception as e:
        print(f"Error processing documents: {e}")
