"""
OPENAI vision

https://platform.openai.com/docs/guides/vision
"""

import os
import time
import argparse

from openai import OpenAI
from dotenv import load_dotenv
from featuring_keywords import dict_keywords


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("--image-url", nargs="*", type=str, help="Image File URL", default=[])
args = parser.parse_args()

def image_to_keywords(image_url: str):
    """
    OpenAI API를 사용하여 입력된 이미지와 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
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
                            # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    end_time = time.time()

    print("TOP 3 Keywords:", response.choices[0].message.content)
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    if args.image_url:
        for image_url in args.image_url:
            image_to_keywords(image_url)
    else:
        print("Error: Image File URL Required")


"""
명령어 실행 예시 및 결과


> python openai/url_image_to_keywords.py --image-url https://as1.ftcdn.net/v2/jpg/02/45/68/40/1000_F_245684006_e55tOria5okQtKmiLLbY30NgEHTIB0Og.jpg
TOP 3 Keywords: ['여행', '일상', '취미/문화']
Execution Time: 4.484991073608398
"""