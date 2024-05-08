import clip
import torch
import pandas
import time

from PIL import Image
from featuring_keywords import dict_keywords


# 피처링 정의 키워드 추출
# file = pandas.read_excel('keywords.xlsx', header=None)
# keywords = file.iloc[1:].values.flatten().tolist()
# keywords = [str(kw).strip() for kw in keywords if str(kw) != "NaN"]
keywords = list(dict_keywords.keys())

def image_to_keywords(image_path: str):
    # CLIP 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # CLIP 모델에 맞게 이미지 전처리
    preprocessed_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 키워드를 텍스트로 변환하고 CLIP에 맞게 전처리
    text_tokens = clip.tokenize(keywords).to(device)

    # 모델 실행 시작
    start_time = time.time()

    # 이미지-키워드 유사성 계산
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        text_features = model.encode_text(text_tokens)

        # 유사성 점수 계산
        similarities = torch.matmul(text_features, image_features.T).squeeze()

        # 가장 높은 점수를 가진 키워드 3개 추출
        top_keywords_eng = [keywords[i] for i in similarities.argsort(descending=True)[:3]]

    # 모델 실행 종료
    end_time = time.time()

    top_keywords_kor = []
    for top_keyword in top_keywords_eng:
        top_keywords_kor.append(dict_keywords[top_keyword])

    print("TOP 3 Keywords (ENG):", top_keywords_eng)
    print("TOP 3 Keywords (KOR):", top_keywords_kor)
    print("Execution Time:", end_time - start_time)

    # return top_keywords_eng, top_keywords_kor, end_time - start_time


if __name__ == "__main__":
    print("1. baseball-stadium")
    image_to_keywords("images/baseball-stadium.jpeg")
    print()

    print("2. artist")
    image_to_keywords("images/artist.jpeg")
    print()

    print("3. tiger")
    image_to_keywords("images/tiger.jpg")
    print()