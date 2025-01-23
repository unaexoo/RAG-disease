import gradio as gr
import disease  

def user_greeting(question):
    response = disease.invoke(question)
    return response

app = gr.Interface(
    fn=user_greeting,  # 함수 연결
    inputs="text",  # 입력 타입
    outputs="text",  # 출력 타입
    title="질병 진단 어시스턴트",  # 제목 설정
    description="질문을 입력하면 관련된 질병 정보를 알려줍니다.",  # 설명
)

if __name__ == "__main__":
    app.launch()
