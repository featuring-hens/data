"""
인스타그램 테스트

프로필 일부 및 최근 게시물(일반적으로 9개)이 나온 사진에 대해,
해당 사진 분석을 통해 인플루언서 카테고리 파악 정확도를 테스트합니다.
"""

import os
import argparse

from openai import OpenAI
from dotenv import load_dotenv
from encoded_image_to_keywords import image_to_keywords


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser()
parser.add_argument("--image-path", nargs="*", type=str, help="Image File Path", default=[])
args = parser.parse_args()

if args.image_path:
    for image_path in args.image_path:
        image_to_keywords(image_path)
else:
    print("Error: Image File URL Required")


"""
명령어 실행 예시 및 결과 (1)


> python openai/images_to_keywords.py --image-path images/ig/1.jpg images/ig/2.jpg images/ig/3.jpg images/ig/4.jpg images/ig/5.jpg images/ig/6.jpg images/ig/7.jpg images/ig/8.jpg images/ig/9.jpg images/ig/10.jpg
TOP 3 Keywords:
Image_1: ['스포츠/운동', '일상', '여행']
Image_2: ['스포츠/운동', '일상', '여행']
Image_3: ['홈/리빙', '일상', '기타']
Image_4: ['만화/애니/툰', '일상', '취미/문화']
Image_5: ['반려동물', '일상', '기타']
Image_6: ['뷰티', '일상', '기타']
Image_7: ['뷰티', '일상', '기타']
Image_8: ['F&B', '일상', '기타']
Image_9: ['여행', '일상', '기타']
Image_10: ['패션', '뷰티', '일상']
Execution Time: 27.863433837890625
"""


"""
명령어 실행 예시 및 결과 (2)


> python openai/images_to_keywords.py --image-path images/ig_post/1.jpg images/ig_post/2.jpg images/ig_post/3.jpg images/ig_post/4.jpg images/ig_post/5.jpg images/ig_post/6.jpg images/ig_post/7.jpg images/ig_post/8.jpg images/ig_post/9.jpg images/ig_post/10.jpg
TOP 3 Keywords:
Image_1: ['패션', '스타/연예인', '스포츠/운동']
Image_2: ['스포츠/운동', '일상', '여행']
Image_3: ['홈/리빙', '일상', '기타']
Image_4: ['만화/애니/툰', '취미/문화', '기타']
Image_5: ['반려동물', '일상', '육아/키즈']
Image_6: ['육아/키즈', '일상', '기타']
Image_7: ['F&B', '일상', '기타']
Image_8: ['여행', '일상', '스포츠/운동']
Image_9: ['패션', '스타/연예인', '뷰티']
Execution Time: 26.284602165222168
"""


"""
명령어 실행 예시 및 결과 (3)


> python openai/images_to_keywords.py --image-path images/ig_post/1.jpg images/ig_post/2.jpg images/ig_post/3.jpg images/ig_post/4.jpg images/ig_post/5.jpg images/ig_post/6.jpg images/ig_post/7.jpg images/ig_post/8.jpg images/ig_post/9.jpg images/ig_post/10.jpg
TOP 3 Keywords:
Image_1: ['스포츠/운동', '일상', '스타/연예인']
Image_2: ['스포츠/운동', '일상', '여행']
Image_3: ['홈/리빙', '일상', '기타']
Image_4: ['만화/애니/툰', '취미/문화', '일상']
Image_5: ['반려동물', '일상', '여행']
Image_6: ['육아/키즈', '일상', '기타']
Image_7: ['F&B', '일상', '기타']
Image_8: ['여행', '일상', '스타/연예인']
Image_9: ['패션', '스타/연예인', '일상']
Image_10: ['뷰티', '스타/연예인', '일상']
Execution Time: 29.259170055389404
"""


"""
명령어 실행 예시 및 결과 (4)


> python openai/test_ig.py --image-path images/yt/닥신TV.jpg      
TOP 3 Keywords: ['자동차/모빌리티', '미디어/엔터테인먼트', '일상']
Execution Time: 6.768593788146973
"""
