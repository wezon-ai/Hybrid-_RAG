## translate.py

import pymupdf4llm
import openai
import os

# OpenAI API Key 설정

# PDF 파일 경로 설정
file_path = "./data_/D-18-2020 안전밸브 등의 배출용량 산정 및 설치 등에 관한 기술지침.pdf"

# 테이블 메타데이터를 제거하는 함수
def format_table_content(tables):
    formatted_tables = []
    for table in tables:
        # 테이블 내용을 문자열로 변환하는 로직 (메타데이터 제거)
        table_content = f"Rows: {table['rows']}, Columns: {table['columns']}\n"
        formatted_tables.append(table_content)
    return "\n".join(formatted_tables)

# 페이지별로 텍스트와 테이블을 추출하고 번역하는 함수
def extract_and_translate_text_by_page(file_path):
    print(f"Processing {file_path}...")

    # 모든 페이지의 텍스트와 테이블을 추출
    try:
        page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)  # 페이지 단위로 청크화 추출
        print("Extraction complete.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

    translated_pages = []
    for page_num, chunk in enumerate(page_chunks, start=1):
        text_chunk = chunk['text']  # 텍스트만 추출
        tables = chunk['tables']  # 테이블만 추출

        # 텍스트 번역
        translated_text = translate_large_text(text_chunk)

        # 테이블 포맷팅
        formatted_tables = format_table_content(tables)

        # 페이지 결과를 저장
        translated_pages.append(f"## Page {page_num}\n\n{translated_text}\n\n### Tables\n{formatted_tables}")

    return "\n".join(translated_pages)

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

# 텍스트 파일로 번역 결과를 저장하는 함수
def save_to_txt(translated_text, output_txt_path):
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write(translated_text)

# 텍스트와 테이블을 추출 및 번역
translated_text = extract_and_translate_text_by_page(file_path)

if translated_text:
    save_to_txt(translated_text, "translated_output.md")
    print("Translation saved to translated_output.md")

