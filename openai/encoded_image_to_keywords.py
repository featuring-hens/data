"""
OPENAI vision

https://platform.openai.com/docs/guides/vision
"""

import os
import time
import argparse
import base64
import requests

from openai import OpenAI
from dotenv import load_dotenv
from featuring_keywords import dict_keywords


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser()
parser.add_argument("--image-path", nargs="*", type=str, help="Image File Path", default=[])
args = parser.parse_args()

# 이미지를 base64로 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_keywords(image_path: str):
    """
    OpenAI API를 사용하여 입력된 이미지와 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{dict_keywords}에서 주어진 이미지와 가장 연관성이 높은 키워드(한국어)를 3개 추출해 주세요. \
                                키워드 추출 결과는 ['키워드_1', '키워드_2', '키워드_3']의 형태로 작성해 주세요."
                                # 그리고 왜 그러한 키워드가 주어진 이미지와 연관성이 높다고 판단했는지에 대해 간략히 설명해 주세요."
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
        }
        ],
        "max_tokens": 300
    }

    start_time = time.time()

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    end_time = time.time()

    print("TOP 3 Keywords:", response.json()['choices'][0]['message']['content'])
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    if args.image_path:
        for image_path in args.image_path:
            image_to_keywords(image_path)
    else:
        print("Error: Image File URL Required")


"""
명령어 실행 예시 및 결과


> python openai/encoded_image_to_keywords.py --image-path images/artist.jpeg images/baseball-stadium.jpeg images/tiger.jpg
TOP 3 Keywords: ['취미/문화', '일상', '홈/리빙']
Execution Time: 2.6784980297088623

TOP 3 Keywords: ['스포츠/운동', '일상', '여행']
Execution Time: 5.59466028213501

TOP 3 Keywords: ['반려동물', '여행', '취미/문화']
Execution Time: 5.311323881149292
"""