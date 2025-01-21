from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from .model import create_medical_assistant_chain
import logging
import json

# 로그 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

main = Blueprint('main', __name__)
llm_chain = create_medical_assistant_chain()  # LangChain 객체 생성

@main.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    """
    Process patient symptoms and generate a response using LangChain.
    """
    try:
        # JSON 데이터 수신
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # 사용자 입력 처리
        patient_input = request_data.get('patient_input') or request_data.get('symptoms')
        if not patient_input or not patient_input.strip():
            return jsonify({"error": "No input provided or input is empty"}), 400
        patient_input = patient_input.strip()

        # LangChain 실행
        try:
            # Chain 실행 (invoke 사용)
            raw_response = llm_chain.invoke({"query": patient_input})
            logging.debug(f"LangChain raw response: {raw_response}")

            # JSON 형식 검증 및 추출
            try:
                response_data = json.loads(raw_response)
            except json.JSONDecodeError:
                # 응답이 JSON 형식이 아닐 경우 예외 처리
                logging.error(f"Invalid JSON response: {raw_response}")
                response_data = {
                    "error": "Model did not return valid JSON",
                    "raw_output": raw_response
                }

        except Exception as e:
            # LangChain 실행 중 발생한 오류 처리
            logging.error(f"LangChain processing failed: {e}")
            response_data = {
                "disease": "알 수 없음",
                "symptoms": "N/A",
                "treatment": "N/A"
            }

        # 세션에 사용자 입력 및 결과 저장
        session['patient_input'] = patient_input
        session['generated_response'] = json.dumps([response_data], ensure_ascii=False)

        return redirect(url_for('main.result'))

    except Exception as e:
        # 상위 레벨 예외 처리
        logging.exception("Error during prediction")
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


@main.route('/result')
def result():
    """
    Render the result page with LangChain's response.
    """
    patient_input = session.get('patient_input', '증상이 입력되지 않았습니다.')
    generated_response = json.loads(session.get('generated_response', '[]'))

    logging.debug(f"Patient Input: {patient_input}")
    logging.debug(f"Generated Response: {generated_response}")

    return render_template(
        'result.html',
        patient_input=patient_input,
        generated_response=generated_response
    )

