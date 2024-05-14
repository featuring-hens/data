"""
CLIP

*https://github.com/openai/CLIP
"""

import clip
import torch
import numpy as np
import time
import argparse
import os

from PIL import Image
from featuring_keywords import dict_keywords


# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
# parser.add_argument("image_path", nargs="+", type=str, help="Image File Path")
parser.add_argument("--image-path", nargs="*", type=str, help="Image File Path", default=[])
args = parser.parse_args()

# 피처링 정의 키워드 추출
# file = pandas.read_excel('keywords.xlsx', header=None)
# keywords = file.iloc[1:].values.flatten().tolist()
# keywords = [str(kw).strip() for kw in keywords if str(kw) != "NaN"]
keywords = list(dict_keywords.keys())

def image_to_keywords(image_path: str):
    """
    CLIP 모델을 사용하여 입력된 이미지와 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    start_time = time.time()

    # CLIP 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # CLIP 모델에 맞게 이미지 전처리
    preprocessed_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 키워드를 텍스트로 변환하고 CLIP에 맞게 전처리
    text_tokens = clip.tokenize(keywords).to(device)

    # 이미지-키워드 유사성 계산
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        text_features = model.encode_text(text_tokens)

        # 유사성 점수 계산
        similarities = torch.matmul(text_features, image_features.T).squeeze()

        # 가장 높은 점수를 가진 키워드 3개 추출
        top_keywords_eng = [keywords[i] for i in similarities.argsort(descending=True)[:3]]
        top_keywords_kor = [dict_keywords[keyword] for keyword in top_keywords_eng]

    end_time = time.time()

    print("Image File Name:", os.path.basename(image_path))
    print("TOP 3 Keywords (ENG):", top_keywords_eng)
    print("TOP 3 Keywords (KOR):", top_keywords_kor)
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    print("Image File Path:", args.image_path)
    if args.image_path:
        for image_path in args.image_path:
            image_to_keywords(image_path)
    else:
        print("Error: Image File Path Required")


"""
명령어 실행 예시 및 결과 (1)


> python clip/image_to_keywords.py --image-path images/artist.jpeg images/baseball-stadium.jpeg images/tiger.jpg

Image File Name: artist.jpeg
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Hobbies/Culture', 'Celebrities/Entertainment']
TOP 3 Keywords (KOR): ['만화/애니/툰', '취미/문화', '스타/연예인']
Execution Time: 3.266166925430298

Image File Name: baseball-stadium.jpeg
TOP 3 Keywords (ENG): ['Beauty', 'Home/Living', 'Sports/Fitness']
TOP 3 Keywords (KOR): ['뷰티', '홈/리빙', '스포츠/운동']
Execution Time: 3.322561025619507

Image File Name: tiger.jpg
TOP 3 Keywords (ENG): ['Beauty', 'Travel', 'Fashion']
TOP 3 Keywords (KOR): ['뷰티', '여행', '패션']
Execution Time: 3.312255859375
"""


"""
명령어 실행 예시 및 결과 (2)


> python clip/image_to_keywords.py --image-path images/ig/1.jpg images/ig/2.jpg images/ig/3.jpg images/ig/4.jpg images/ig/5.jpg images/ig/6.jpg images/ig/7.jpg images/ig/8.jpg images/ig/9.jpg images/ig/10.jpg
Image File Path: ['images/ig/1.jpg', 'images/ig/2.jpg', 'images/ig/3.jpg', 'images/ig/4.jpg', 'images/ig/5.jpg', 'images/ig/6.jpg', 'images/ig/7.jpg', 'images/ig/8.jpg', 'images/ig/9.jpg', 'images/ig/10.jpg']
Image File Name: 1.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Media/Entertainment', 'Daily Life']
TOP 3 Keywords (KOR): ['스포츠/운동', '미디어/엔터테인먼트', '일상']
Execution Time: 3.783710241317749

Image File Name: 2.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Others', 'Daily Life']
TOP 3 Keywords (KOR): ['스포츠/운동', '기타', '일상']
Execution Time: 3.492805004119873

Image File Name: 3.jpg
TOP 3 Keywords (ENG): ['Food & Beverage', 'Media/Entertainment', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['F&B', '미디어/엔터테인먼트', '만화/애니/툰']
Execution Time: 3.659393787384033

Image File Name: 4.jpg
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['만화/애니/툰', '일상', '결혼/연애']
Execution Time: 3.4755001068115234

Image File Name: 5.jpg
TOP 3 Keywords (ENG): ['Daily Life', 'Pets', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['일상', '반려동물', '만화/애니/툰']
Execution Time: 3.383308172225952

Image File Name: 6.jpg
TOP 3 Keywords (ENG): ['Daily Life', 'Celebrities/Entertainment', 'Others']
TOP 3 Keywords (KOR): ['일상', '스타/연예인', '기타']
Execution Time: 3.3157958984375

Image File Name: 7.jpg
TOP 3 Keywords (ENG): ['Food & Beverage', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['F&B', '일상', '기타']
Execution Time: 3.3431050777435303

Image File Name: 8.jpg
TOP 3 Keywords (ENG): ['Travel', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['여행', '일상', '기타']
Execution Time: 3.34493088722229

Image File Name: 9.jpg
TOP 3 Keywords (ENG): ['Celebrities/Entertainment', 'Fashion', 'Daily Life']
TOP 3 Keywords (KOR): ['스타/연예인', '패션', '일상']
Execution Time: 3.6524240970611572

Image File Name: 10.jpg
TOP 3 Keywords (ENG): ['Media/Entertainment', 'Celebrities/Entertainment', 'Sports/Fitness']
TOP 3 Keywords (KOR): ['미디어/엔터테인먼트', '스타/연예인', '스포츠/운동']
Execution Time: 3.5192861557006836
"""
