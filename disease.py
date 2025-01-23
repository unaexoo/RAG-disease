# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# %%
# 단계 1 : 문서 로드
loader = CSVLoader(r"C:\LangGraph\Diseases_Symptoms.csv")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")
print(docs[0].metadata)

# %%
# CSV 파일 경로
loader = CSVLoader(
    file_path="./Diseases_Symptoms.csv",
    csv_args={
        "delimiter": ",",  # 구분자
        "quotechar": '"',  # 인용 부호 문자
        "fieldnames": [
            "Code",
            "Name",
            "Symptoms",
            "Treatments",
        ],  # 필드 이름
    },
)

# 데이터 로드
docs = loader.load()

# 데이터 출력
print(docs[1].page_content)


# %%
# 단계 2 : 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의 수 : {len(split_documents)}")

# %%
# 단계 3 : 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})

# %%
# 단계 4 : DB 생성 및 저장
vectorstore = Chroma.from_documents(documents=split_documents, embedding=embeddings)

# %%
# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()

# %%
# 관련 문서를 검색
docs = retriever.invoke("나 콧물 나와")

for doc in docs:
    print(doc.page_content)
    print("=========================================================")


# %%
# 단계 7: 언어 모델 생성
llm=ChatGroq(temperature=0, model="gemma2-9b-it", api_key="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")

# %%
llm.invoke("나 콧물 나와")

# %%
prompt = PromptTemplate.from_template(
    """You are an assistant in the question-and-answer task. However, this content is for medical information only. Use the following retrieved context fragment to answer the question. 
If you don't know the answer, just say you don't know. Please answer in Korean. Displays up to three related diseases along with their symptoms, treatments, and match rates. The matching rate is extracted as a percentage of the searched value.

#question: 
{question} 
#context: 
{context} 

#answer:
1. 질병명: 
   매칭률: 
   증상: 
   치료법: 

2. 질병명: 
   매칭률: 
   증상: 
   치료법: 

3. 질병명: 
   매칭률: 
   증상: 
   치료법: 
"""
)

# %%
# 단계 8: 체인(chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
def invoke(question: str):
    response = chain.invoke(question)
    return response


# %%
question = "나 콧물 나와"
response = invoke(question)
print(response)


# %%



