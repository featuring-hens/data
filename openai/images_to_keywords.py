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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def images_to_keywords(image_paths: list):
    """
    OpenAI API를 사용하여 입력된 이미지들과 관련된 피처링 정의 키워드(3개)를 출력하는 함수

    API를 한 번 호출하여 입력된 모든 이미지에 대한 결과를 출력합니다.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"주어진 이미지는 인스타그램(instagram)에 모바일로 접속했을 때 보여지는 화면이고, \
                        해당 계정의 프로필 일부와 최근 게시물(일반적으로 9개)이 나온 사진입니다. \
                        {dict_keywords}에서 주어진 이미지와 가장 연관성이 높은 키워드(한국어)를 3개 추출해 주세요. \
                        키워드 추출 결과에 대한 응답은 아래와 같이 작성해 주세요. \
                        Image_(번호): ['키워드_1', '키워드_2', '키워드_3'] \
                        예시: Image_1: ['패션', '스타/연예인', '일상']"
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                }
            }
            for image_path in image_paths
        ]
    }]

    payload = {
        "model": "gpt-4-turbo",
        "messages": messages,
        "max_tokens": 600
    }

    start_time = time.time()

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    end_time = time.time()

    print("TOP 3 Keywords:\n", response.json()['choices'][0]['message']['content'])
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    if args.image_path:
        images_to_keywords(args.image_path)
    else:
        print("Error: Image File URL Required")


"""
명령어 실행 예시 및 결과 (1)


> python openai/images_to_keywords.py --image-path images/ig/amottivation.jpg images/ig/jbkwak.jpg images/ig/risabae_art.jpg
TOP 3 Keywords:
Image_1: ['스포츠/운동', '일상', '패션']
Image_2: ['여행', '일상', '미디어/엔터테인먼트']
Image_3: ['뷰티', '패션', '스타/연예인']
Execution Time: 15.57243800163269
"""


"""
명령어 실행 예시 및 결과 (2)


> python openai/images_to_keywords.py --image-path images/ig_prof_post/1.jpg images/ig_prof_post/2.jpg images/ig_prof_post/3.jpg images/ig_prof_post/4.jpg images/ig_prof_post/5.jpg images/ig_prof_post/6.jpg images/ig_prof_post/7.jpg images/ig_prof_post/8.jpg images/ig_prof_post/9.jpg images/ig_prof_post/10.jpg
TOP 3 Keywords:
Image_1: ['스포츠/운동', '일상', '여행']
Image_2: ['스포츠/운동', 'F&B', '일상']
Image_3: ['홈/리빙', '일상', '육아/키즈']
Image_4: ['만화/애니/툰', '일상', '취미/문화']
Image_5: ['반려동물', '일상', '여행']
Image_6: ['뷰티', '일상', '육아/키즈']
Image_7: ['F&B', '일상', '여행']
Image_8: ['여행', '일상', '스타/연예인']
Image_9: ['패션', '뷰티', '일상']
Image_10: ['뷰티', '패션', '일상']
Execution Time: 29.09743094444275
"""