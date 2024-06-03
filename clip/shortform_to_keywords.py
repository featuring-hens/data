"""
CLIP

*https://github.com/openai/CLIP
"""

import clip
import torch
import time
import argparse
import os
import cv2

from PIL import Image
from featuring_keywords import dict_keywords


# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("--shortform-path", nargs="*", type=str, help="Shortform File Path", default=[])
args = parser.parse_args()

# 피처링 정의 키워드 추출
keywords = list(dict_keywords.keys())

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("RN50", device=device)

def shortform_to_keywords(shortform_path: str):
    """
    CLIP 모델을 사용하여 입력된 숏폼과 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    start_time = time.time()

    # 숏폼 파일 로드
    cap = cv2.VideoCapture(shortform_path)
    if not cap.isOpened():
        print("Error: Could not open shortform.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // 10)

    all_similarities = torch.zeros(len(keywords)).to(device)

    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # OpenCV 이미지를 PIL 이미지로 변환
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(preprocessed_image)
            text_tokens = clip.tokenize(keywords).to(device)
            text_features = model.encode_text(text_tokens)

            similarities = torch.matmul(text_features, image_features.T).squeeze()
            all_similarities += similarities

    # 해제
    cap.release()

    # 상위 3개의 키워드 추출
    top_indices = all_similarities.argsort(descending=True)[:3]
    top_keywords_eng_final = [keywords[i] for i in top_indices]
    top_keywords_kor_final = [dict_keywords[keyword] for keyword in top_keywords_eng_final]

    end_time = time.time()

    print("Shortform File Name:", os.path.basename(shortform_path))
    print("TOP 3 Keywords (ENG):", top_keywords_eng_final)
    print("TOP 3 Keywords (KOR):", top_keywords_kor_final)
    print("Execution Time:", end_time - start_time)
    print()


if __name__ == "__main__":
    print("Shortform File Path:", args.shortform_path)
    if args.shortform_path:
        for shortform_path in args.shortform_path:
            shortform_to_keywords(shortform_path)
    else:
        print("Error: Shortform File Path Required")


"""
명령어 실행 예시 및 결과 (1)


> python clip/shortform_to_keywords.py --shortform-path /Users/je/desktop/여행.mp4 /Users/je/desktop/화장.mp4 /Users/je/desktop/강아지.mp4 /Users/je/desktop/아기.mp4 /Users/je/desktop/차.mp4 /Users/je/desktop/음식.mp4 /Users/je/desktop/홈.mp4  
Shortform File Path: ['/Users/je/desktop/여행.mp4', '/Users/je/desktop/화장.mp4', '/Users/je/desktop/강아지.mp4', '/Users/je/desktop/아기.mp4', '/Users/je/desktop/차.mp4', '/Users/je/desktop/음식.mp4', '/Users/je/desktop/홈.mp4']
Shortform File Name: 여행.mp4
TOP 3 Keywords (ENG): ['Travel', 'Daily Life', 'Others']
TOP 3 Keywords (KOR): ['여행', '일상', '기타']
Execution Time: 9.171317100524902

Shortform File Name: 화장.mp4
TOP 3 Keywords (ENG): ['Beauty', 'Daily Life', 'Hobbies/Culture']
TOP 3 Keywords (KOR): ['뷰티', '일상', '취미/문화']
Execution Time: 8.599191188812256

Shortform File Name: 강아지.mp4
TOP 3 Keywords (ENG): ['Pets', 'Home/Living', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['반려동물', '홈/리빙', '만화/애니/툰']
Execution Time: 11.893131017684937

Shortform File Name: 아기.mp4
TOP 3 Keywords (ENG): ['Parenting/Kids', 'Daily Life', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['육아/키즈', '일상', '만화/애니/툰']
Execution Time: 13.051569938659668

Shortform File Name: 차.mp4
TOP 3 Keywords (ENG): ['Automotive/Mobility', 'Daily Life', 'Celebrities/Entertainment']
TOP 3 Keywords (KOR): ['자동차/모빌리티', '일상', '스타/연예인']
Execution Time: 9.190489053726196

Shortform File Name: 음식.mp4
TOP 3 Keywords (ENG): ['Food & Beverage', 'Home/Living', 'Marriage/Dating']
TOP 3 Keywords (KOR): ['F&B', '홈/리빙', '결혼/연애']
Execution Time: 11.898238897323608

Shortform File Name: 홈.mp4
TOP 3 Keywords (ENG): ['Home/Living', 'Daily Life', 'Comics/Animation/Cartoons']
TOP 3 Keywords (KOR): ['홈/리빙', '일상', '만화/애니/툰']
Execution Time: 11.232323169708252
"""