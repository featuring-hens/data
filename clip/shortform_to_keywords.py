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
