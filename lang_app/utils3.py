import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Hugging Face Hub API 토큰 설정
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# 로그인
login()

# 모델 및 토크나이저 로딩
repo_id = 'Bllossom/llama-3-Korean-Bllossom-70B'
model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto")  # GPU 활용 자동 설정
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# 번역 함수 정의
def translate(input_text, source_lang='ko', target_lang='en'):
    """
    입력 텍스트를 source_lang에서 target_lang로 번역.
    
    Args:
        input_text (str): 번역할 텍스트
        source_lang (str): 입력 텍스트 언어 코드 (기본값: 'ko')
        target_lang (str): 출력 텍스트 언어 코드 (기본값: 'en')
    
    Returns:
        str: 번역된 텍스트
    """
    prompt = f"Translate this text from {source_lang} to {target_lang}: {input_text}"
    
    # 입력 텍스트 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 모델을 사용하여 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_length=512,  # 출력 길이 제한
        temperature=0.2,  # 생성 다양성 조절
        num_beams=5       # 빔 검색 사용
    )
    
    # 결과 디코딩 및 후처리
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return translated_text