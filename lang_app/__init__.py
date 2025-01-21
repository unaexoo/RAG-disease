from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)
    app.secret_key = "16"  # 세션에 필요한 secret key 설정
    app.register_blueprint(main, url_prefix="/")  # Blueprint 등록
    return app

