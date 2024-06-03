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

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('RN101', device=device)
model, preprocess = clip.load("ViT-B/32", device=device)

def image_to_keywords(image_path: str):
    """
    CLIP 모델을 사용하여 입력된 이미지와 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    start_time = time.time()

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


> python clip/image_to_keywords.py --image-path images/ig_post_9/1.png images/ig_post_9/2.png images/ig_post_9/3.png images/ig_post_9/4.png images/ig_post_9/5.png images/ig_post_9/6.png images/ig_post_9/7.png images/ig_post_9/8.png images/ig_post_9/9.png images/ig_post_9/10.png
Image File Path: ['images/ig_post_9/1.png', 'images/ig_post_9/2.png', 'images/ig_post_9/3.png', 'images/ig_post_9/4.png', 'images/ig_post_9/5.png', 'images/ig_post_9/6.png', 'images/ig_post_9/7.png', 'images/ig_post_9/8.png', 'images/ig_post_9/9.png', 'images/ig_post_9/10.png']
Image File Name: 1.png
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Daily Life', 'Travel']
TOP 3 Keywords (KOR): ['스포츠/운동', '일상', '여행']
Execution Time: 0.5491058826446533

Image File Name: 2.png
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['스포츠/운동', '일상', '기타']
Execution Time: 1.0257189273834229

Image File Name: 3.png
TOP 3 Keywords (ENG): ['Hobbies/Culture', 'Daily Life', 'Home/Living']
TOP 3 Keywords (KOR): ['취미/문화', '일상', '홈/리빙']
Execution Time: 0.644136905670166

Image File Name: 4.png
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['만화/애니/툰', '일상', '기타']
Execution Time: 0.691483736038208

Image File Name: 5.png
TOP 3 Keywords (ENG): ['Daily Life', 'Pets', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['일상', '반려동물', '만화/애니/툰']
Execution Time: 0.8347578048706055

Image File Name: 6.png
TOP 3 Keywords (ENG): ['Parenting/Kids', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['육아/키즈', '일상', '결혼/연애']
Execution Time: 0.9594709873199463

Image File Name: 7.png
TOP 3 Keywords (ENG): ['Food & Beverage', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['F&B', '일상', '취미/문화']
Execution Time: 0.6524143218994141

Image File Name: 8.png
TOP 3 Keywords (ENG): ['Travel', 'Hobbies/Culture', 'Daily Life']
TOP 3 Keywords (KOR): ['여행', '취미/문화', '일상']
Execution Time: 0.7809238433837891

Image File Name: 9.png
TOP 3 Keywords (ENG): ['Fashion', 'Celebrities/Entertainment', 'Daily Life']
TOP 3 Keywords (KOR): ['패션', '스타/연예인', '일상']
Execution Time: 0.6968801021575928

Image File Name: 10.png
TOP 3 Keywords (ENG): ['Fashion', 'Beauty', 'Daily Life']
TOP 3 Keywords (KOR): ['패션', '뷰티', '일상']
Execution Time: 0.6631419658660889
"""


"""
명령어 실행 예시 및 결과 (2)


> python clip/image_to_keywords.py --image-path images/ig_post/1.jpg images/ig_post/2.jpg images/ig_post/3.jpg images/ig_post/4.jpg images/ig_post/5.jpg images/ig_post/6.jpg images/ig_post/7.jpg images/ig_post/8.jpg images/ig_post/9.jpg images/ig_post/10.jpg
Image File Path: ['images/ig_post/1.jpg', 'images/ig_post/2.jpg', 'images/ig_post/3.jpg', 'images/ig_post/4.jpg', 'images/ig_post/5.jpg', 'images/ig_post/6.jpg', 'images/ig_post/7.jpg', 'images/ig_post/8.jpg', 'images/ig_post/9.jpg', 'images/ig_post/10.jpg']
Image File Name: 1.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['스포츠/운동', '일상', '취미/문화']
Execution Time: 0.4062638282775879

Image File Name: 2.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Celebrities/Entertainment', 'Media/Entertainment']
TOP 3 Keywords (KOR): ['스포츠/운동', '스타/연예인', '미디어/엔터테인먼트']
Execution Time: 0.48108696937561035

Image File Name: 3.jpg
TOP 3 Keywords (ENG): ['Home/Living', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['홈/리빙', '일상', '취미/문화']
Execution Time: 0.5586161613464355

Image File Name: 4.jpg
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['만화/애니/툰', '일상', '결혼/연애']
Execution Time: 0.7358639240264893

Image File Name: 5.jpg
TOP 3 Keywords (ENG): ['Daily Life', 'Pets', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['일상', '반려동물', '취미/문화']
Execution Time: 0.5884578227996826

Image File Name: 6.jpg
TOP 3 Keywords (ENG): ['Parenting/Kids', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['육아/키즈', '일상', '결혼/연애']
Execution Time: 0.5739848613739014

Image File Name: 7.jpg
TOP 3 Keywords (ENG): ['Food & Beverage', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['F&B', '일상', '취미/문화']
Execution Time: 0.6256668567657471

Image File Name: 8.jpg
TOP 3 Keywords (ENG): ['Travel', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['여행', '일상', '기타']
Execution Time: 0.713547945022583

Image File Name: 9.jpg
TOP 3 Keywords (ENG): ['Fashion', 'Celebrities/Entertainment', 'Others']
TOP 3 Keywords (KOR): ['패션', '스타/연예인', '기타']
Execution Time: 0.5839757919311523

Image File Name: 10.jpg
TOP 3 Keywords (ENG): ['Fashion', 'Beauty', 'Daily Life']
TOP 3 Keywords (KOR): ['패션', '뷰티', '일상']
Execution Time: 0.5879721641540527
"""


"""
명령어 실행 예시 및 결과 (3)


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


"""
명령어 실행 예시 및 결과 (4)


> python clip/image_to_keywords.py --image-path images/ig/1.jpg images/ig/2.jpg images/ig/3.jpg images/ig/4.jpg images/ig/5.jpg images/ig/6.jpg images/ig/7.jpg images/ig/8.jpg images/ig/9.jpg images/ig/10.jpg
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


"""
명령어 실행 예시 및 결과 (5)


> python clip/image_to_keywords.py --image-path images/ig_post/1.jpg images/ig_post/2.jpg images/ig_post/3.jpg images/ig_post/4.jpg images/ig_post/5.jpg images/ig_post/6.jpg images/ig_post/7.jpg images/ig_post/8.jpg images/ig_post/9.jpg images/ig_post/10.jpg
Image File Path: ['images/ig_post/1.jpg', 'images/ig_post/2.jpg', 'images/ig_post/3.jpg', 'images/ig_post/4.jpg', 'images/ig_post/5.jpg', 'images/ig_post/6.jpg', 'images/ig_post/7.jpg', 'images/ig_post/8.jpg', 'images/ig_post/9.jpg', 'images/ig_post/10.jpg']
Image File Name: 1.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['스포츠/운동', '일상', '취미/문화']
Execution Time: 3.8931541442871094

Image File Name: 2.jpg
TOP 3 Keywords (ENG): ['Sports/Fitness', 'Celebrities/Entertainment', 'Media/Entertainment']
TOP 3 Keywords (KOR): ['스포츠/운동', '스타/연예인', '미디어/엔터테인먼트']
Execution Time: 3.5437159538269043

Image File Name: 3.jpg
TOP 3 Keywords (ENG): ['Home/Living', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['홈/리빙', '일상', '취미/문화']
Execution Time: 3.3530430793762207

Image File Name: 4.jpg
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['만화/애니/툰', '일상', '결혼/연애']
Execution Time: 3.3914973735809326

Image File Name: 5.jpg
TOP 3 Keywords (ENG): ['Daily Life', 'Pets', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['일상', '반려동물', '취미/문화']
Execution Time: 3.4652960300445557

Image File Name: 6.jpg
TOP 3 Keywords (ENG): ['Parenting/Kids', 'Daily Life', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['육아/키즈', '일상', '결혼/연애']
Execution Time: 3.510446071624756

Image File Name: 7.jpg
TOP 3 Keywords (ENG): ['Food & Beverage', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['F&B', '일상', '취미/문화']
Execution Time: 3.469528913497925

Image File Name: 8.jpg
TOP 3 Keywords (ENG): ['Travel', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['여행', '일상', '기타']
Execution Time: 3.481348991394043

Image File Name: 9.jpg
TOP 3 Keywords (ENG): ['Fashion', 'Celebrities/Entertainment', 'Others']
TOP 3 Keywords (KOR): ['패션', '스타/연예인', '기타']
Execution Time: 3.3355510234832764

Image File Name: 10.jpg
TOP 3 Keywords (ENG): ['Fashion', 'Beauty', 'Daily Life']
TOP 3 Keywords (KOR): ['패션', '뷰티', '일상']
Execution Time: 3.5700368881225586
"""
