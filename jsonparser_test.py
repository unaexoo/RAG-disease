import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate

# 출력 모델 정의
class Output(BaseModel):
    disease: str = Field(default="알 수 없음", description="The name of the disease.")
    symptoms: str = Field(default="N/A", description="Symptoms related to the disease.")
    treatment: str = Field(default="N/A", description="Suggested treatment methods.")

def create_medical_assistant_chain():
    # JSON 출력 파서
    parser = JsonOutputParser(pydantic_object=Output)

    # ChatPromptTemplate 생성
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "의학 보조로 환자에게 질병과 증상, 치료 방법에 대해 간략히 소개해주는 전문가입니다. 답변은 json 형식으로 해주세요."),
        ("user", "{query}")
    ])
    
    # HuggingFace 모델 정의
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-v0.1",
        model_kwargs={"temperature": 0.2, "max_length": 512},
    )

    # Chain 구성
    chain = chat_prompt | model | parser
    return chain

import json

def test_medical_assistant_chain():
    # Chain 생성
    chain = create_medical_assistant_chain()

    # 테스트용 쿼리
    test_query = "열이 나고 기침을 동반하는 증상이 나타납니다. 어떤 질병일까요?"

    try:
        # Chain 실행
        raw_response = chain.invoke({"query": test_query})

        # 응답 확인 및 출력
        print("Test Query:", test_query)
        print("Response:", raw_response)

    except Exception as e:
        # 예외 처리
        response = {"error": str(e)}
        print("Error:", response)

if __name__ == "__main__":
    test_medical_assistant_chain()


