"""
USE + EfficientNet

USE: 텍스트에 대한 특성 추출
EfficientNet: 이미지에 대한 특성 추출

*https://github.com/tensorflow/hub
*https://github.com/tensorflow/models
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import os
import time

from PIL import Image
from featuring_keywords import dict_keywords

# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("--image-path", nargs="*", type=str, help="Image File Path", default=[])
args = parser.parse_args()

# USE(Universal Sentence Encoder) 로드
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# EfficientNet 로드
image_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', pooling='max')

# 피처링 정의 키워드 추출
keywords = list(dict_keywords.keys())

def image_to_keywords(image_path: str):
    start_time = time.time()

    # 이미지 전처리
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))    # EfficientNet의 일반적인 입력 이미지 크기
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 이미지 특성 추출 및 차원 축소
    # 첫 번째 512개의 특성(주요 특성)만 사용
    image_features = image_model.predict(img_array).flatten()[:512]

    # USE로 키워드를 벡터로 변환
    # 임베딩 후 numpy 배열로 변환
    text_features = embed(keywords).numpy()

    # 각 키워드 벡터와 축소된 이미지 특성 간 유사도 계산
    similarities = np.array([np.inner(image_features, text_feature) for text_feature in text_features])

    # 상위 키워드 3개 추출
    top_indices = np.argsort(similarities)[::-1][:3]
    top_keywords_eng = [keywords[i] for i in top_indices]
    top_keywords_kor = [dict_keywords[keyword] for keyword in top_keywords_eng]

    end_time = time.time()

    print("Image File Name:", os.path.basename(image_path))
    print("TOP 3 Keywords (ENG):", top_keywords_eng)
    print("TOP 3 Keywords (KOR):", top_keywords_kor)
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    if args.image_path:
        for image_path in args.image_path:
            image_to_keywords(image_path)
    else:
        print("Error: Image File Path")


"""
명령어 실행 예시 및 결과


> python use-efficientnet/image_to_keywords.py --image-path images/artist.jpeg images/baseball-stadium.jpeg images/tiger.jpg

1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step
Image File Name: artist.jpeg
TOP 3 Keywords (ENG): ['Others', 'Parenting/Kids', 'Travel']
TOP 3 Keywords (KOR): ['기타', '육아/키즈', '여행']
Execution Time: 1.3453710079193115

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
Image File Name: baseball-stadium.jpeg
TOP 3 Keywords (ENG): ['Parenting/Kids', 'Comics/Animation/Cartoons', 'Fashion']
TOP 3 Keywords (KOR): ['육아/키즈', '만화/애니/툰', '패션']
Execution Time: 0.0814657211303711

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
Image File Name: tiger.jpg
TOP 3 Keywords (ENG): ['Beauty', 'Hobbies/Culture', 'Parenting/Kids']
TOP 3 Keywords (KOR): ['뷰티', '취미/문화', '육아/키즈']
Execution Time: 0.0662689208984375
"""
