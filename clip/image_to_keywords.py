"""
CLIP

*https://github.com/openai/CLIP
"""

import clip
import torch
import pandas
import time
import argparse
import os

from PIL import Image
from featuring_keywords import dict_keywords


# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Image File Path")
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

    # return top_keywords_eng, top_keywords_kor, end_time - start_time


if __name__ == "__main__":
    image_to_keywords(args.image_path)


"""
명령어 실행 예시 및 결과

[1] python clip/image_to_keywords.py images/artist.jpeg
Image File Name: artist.jpeg
TOP 3 Keywords (ENG): ['Comics/Animation/Cartoons', 'Hobbies/Culture', 'Celebrities/Entertainment']
TOP 3 Keywords (KOR): ['만화/애니/툰', '취미/문화', '스타/연예인']
Execution Time: 4.06693696975708

-

[2] python clip/image_to_keywords.py images/baseball-stadium.jpeg
Image File Name: baseball-stadium.jpeg
TOP 3 Keywords (ENG): ['Beauty', 'Home/Living', 'Sports/Fitness']
TOP 3 Keywords (KOR): ['뷰티', '홈/리빙', '스포츠/운동']
Execution Time: 4.002578020095825

-

[3] python clip/image_to_keywords.py images/tiger.jpg
Image File Name: tiger.jpg
TOP 3 Keywords (ENG): ['Beauty', 'Travel', 'Fashion']
TOP 3 Keywords (KOR): ['뷰티', '여행', '패션']
Execution Time: 3.6416969299316406
"""
