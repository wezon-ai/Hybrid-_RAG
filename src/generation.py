## generation.py

import openai
import os
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from rerank import reranking
from Retriever_openai import Hybrid_search
from llama_index.core.response_synthesizers import BaseSynthesizer
from dotenv import load_dotenv

# Load environment variables (including OpenAI API key)
load_dotenv()

## 1. prompt_template_generation 클래스
class prompt_template_generation():
    def __init__(self) -> None:
        self.search = Hybrid_search()
        self.reranker = reranking()
        # self.prompt_str = """You are an AI assistant specializing in explaining complex topics related to Retrieval-Augmented Generation(RAG). Your task is to provide a clear, concise, and informative explanation based on the following context and query.

        # Context:
        # {context_str}

        # Query: {query_str}

        # Please follow these guidelines in your response:
        # 1. Start with a brief overview of the concept mentioned in the query.
        # 2. Provide at least one concrete example or use case to illustrate the concept.
        # 3. If there are any limitations or challenges associated with this concept, briefly mention them.
        # 4. Conclude with a sentence about the potential future impact or applications of this concept.

        # Your explanation should be informative yet accessible, suitable for someone with a basic understanding of RAG. If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer, and only respond based on the given context.

        # Response:
        # """
        self.prompt_str = """You are an AI assistant specializing in retrieving and interpreting data from structured information. Your task is to accurately extract relevant information from the provided context and answer the query based on that data.

        Context:
        {context_str}

        Query: {query_str}

        Please follow these guidelines in your response:
        1. First, extract the specific data relevant to the query from the context.
        2. Based on the extracted data, provide a clear and concise answer to the query.
        3. If the context does not contain enough information to answer the query, mention that there is insufficient information and only respond based on the given context.
        4. Your response should be concise and easy to understand.

        Response:
        """
        self.prompt_tmpl = PromptTemplate(self.prompt_str)

        # 4. **Your response must be in Korean. Under no circumstances should your response be in English.**
        

    def prompt_generation(self, query: str, filename: str):
        metadata_filter = self.search.metadata_filter(filename)
        results = self.search.query_hybrid_search(query, metadata_filter)
        
        reranked_documents = self.reranker.rerank_documents(query, results)
        
        context = "/n/n".join(reranked_documents)

        # 프롬프트 생성 확인
        # translated_query = self.search.translate_query_to_english(query)
        prompt_templ = self.prompt_tmpl.format(context_str=context, query_str=query)
        print("Generated prompt:", prompt_templ)  # 터미널에 로그 출력

        return prompt_templ

    # def prompt_generation(self, query: str, filename: str):
    #     metadata_filter = self.search.metadata_filter(filename)
        
    #     # 쿼리를 번역한 후 검색에 사용
    #     translated_query = self.search.translate_query_to_english(query)
        
    #     # 번역된 쿼리를 사용하여 하이브리드 검색 수행
    #     results = self.search.query_hybrid_search(translated_query, metadata_filter)
        
    #     # 검색된 문서 리랭킹
    #     reranked_documents = self.reranker.rerank_documents(translated_query, results)
        
    #     context = "/n/n".join(reranked_documents)

    #     # 번역된 쿼리를 프롬프트에 포함
    #     prompt_templ = self.prompt_tmpl.format(context_str=context, query_str=translated_query)
    #     print("Generated prompt:", prompt_templ)  # 터미널에 로그 출력

    #     return prompt_templ

## 2. RAGStringQueryEngine 클래스
class RAGStringQueryEngine(CustomQueryEngine):
    api_key: str
    response_synthesizer: BaseSynthesizer

    def __init__(self, api_key: str, response_synthesizer: BaseSynthesizer):
        # self.api_key = api_key
        # self.response_synthesizer = response_synthesizer
        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'response_synthesizer', response_synthesizer)
        

    def custom_query(self, prompt: str) -> str:
        openai.api_key = self.api_key

        # API 키 출력
        print(f"Using API Key: {openai.api_key}")  # 터미널에 API 키 출력
        
        # GPT-4o API 호출 (openai.ChatCompletion.create() 사용)
        response = openai.chat.completions.create(
            model="gpt-4o",  # GPT-4o 모델 사용
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in explaining complex topics."},
                {"role": "user", "content": prompt}
            ],
            # max_tokens=1000,  # 최대 토큰 수
            # temperature=0.7   # 창의성 및 답변 다양성을 조정
        )

         # API 응답 확인
        print("OpenAI API response:", response)  # 터미널에 로그 출력
        
        # completion_text = response.choices[0].message['content'].strip()
        completion_text = response.choices[0].message.content.strip()
        # 응답을 요약하는 로직 (TreeSummarize 사용)
        summary = self.response_synthesizer.get_response(query_str=completion_text, text_chunks=prompt)
        return str(summary)

        # return completion_text

def create_query_engine(prompt: str):
    api_key = os.environ.get('OPENAI_API_KEY')  # OpenAI API 키를 환경변수에서 불러옴
    response_synthesizer = TreeSummarize()  # TreeSummarize 응답 합성기

    query_engine = RAGStringQueryEngine(
        api_key=api_key,
        response_synthesizer=response_synthesizer
    )
    
    response = query_engine.custom_query(prompt)
    return response


if __name__ == '__main__':
   
    # query_str = "What are the required electrical hazards when documenting the energized condition in a panel with exposed parts?"
    query_str = "What are the potential hazards when documenting elevated electrical hazards in high locations?"
    
    filename ="ADM_04-00-003"

    prompt_gen = prompt_template_generation()
    prompt = prompt_gen.prompt_generation(query=query_str, filename=filename)
    response = create_query_engine(prompt)
    print(response)
