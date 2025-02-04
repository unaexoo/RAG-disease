import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'api_key'

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Output(BaseModel):
    disease: str = Field(default="질병 이름 (확실하지 않은 경우 '알 수 없음')", description="The name of the disease.")
    symptoms: str = Field(default="질병과 관련된 증상 (없을 경우 'N/A')", description="Symptoms related to the disease.")
    treatment: str = Field(default="제안된 치료 방법 (없을 경우 'N/A')", description="Suggested treatment methods.")

def create_medical_assistant_chain():
    parser = JsonOutputParser(pydantic_object=Output)
    prompt = PromptTemplate(
        template="""
        Answer the user's query in the following JSON format. Only return the JSON object.
        Please do not create new querie.

        Schema:
        {{
            "disease": "질병 이름 (확실하지 않은 경우 '알 수 없음')",
            "symptoms": "질병과 관련된 증상 (없을 경우 'N/A')",
            "treatment": "제안된 치료 방법 (없을 경우 'N/A')"
        }}

        Example Output:
        {{
            "disease": "위염",
            "symptoms": "복통, 설사",
            "treatment": "위산 억제제 복용, 식사 조절"
        }}

        Query: {query}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-v0.1",
        model_kwargs={"temperature": 0.2, "max_length": 512},
    )

    chain = prompt | model | parser
    return chain

import json
def test_medical_assistant_chain():
    # Chain 생성
    chain = create_medical_assistant_chain()

    # 테스트용 쿼리
    test_query = "열이 나고 기침을 동반하는 증상이 나타납니다. 어떤 질병일까요?"

    try:
        # Chain 실행 (invoke를 사용)
        raw_response = chain.invoke({"query": test_query})

        # JSON만 추출
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError:
            # 응답이 JSON 형식이 아닐 경우 예외 처리
            response = {"error": "Model did not return valid JSON", "raw_output": raw_response}

    except Exception as e:
        # 기타 오류 처리
        response = {"error": str(e)}

    # 결과 출력
    print("Test Query:", test_query)
    print("Response:", response)
    
if __name__ == "__main__":
    test_medical_assistant_chain()
