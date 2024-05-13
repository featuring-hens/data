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
명령어 실행 예시 및 결과


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
프롬프트 반영된 코드!

import clip
import torch
from PIL import Image
import numpy as np
import time
import argparse
import os

# argparse로 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("--image-path", nargs="*", type=str, help="Image File Path", default=[])
parser.add_argument("--prompt-file", type=str, help="File containing prompts for each image", default="")

args = parser.parse_args()

# Function to read prompts from a file
def read_prompts(prompt_file):
    with open(prompt_file, "r") as file:
        prompts = [line.strip() for line in file.readlines()]
    return prompts

# Function to perform inference
def image_to_keywords(image_path: str, prompt: str):
    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    preprocessed_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([prompt]).to(device)  # Tokenizing the custom prompt
    
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        text_features = model.encode_text(text_tokens)
        
        similarities = torch.matmul(text_features, image_features.T).squeeze()
        top_indices = similarities.argsort(descending=True)[:3]
        
        top_keywords = [prompt.split()[i] for i in top_indices]  # Simplified for example

    end_time = time.time()
    
    print("Image File Name:", os.path.basename(image_path))
    print("TOP 3 Keywords (ENG):", top_keywords)
    print("Execution Time:", end_time - start_time)
    print()

if __name__ == "__main__":
    if args.image_path and args.prompt_file:
        prompts = read_prompts(args.prompt_file)
        for image_path, prompt in zip(args.image_path, prompts):
            image_to_keywords(image_path, prompt)
    else:
        print("Error: Image File Path or Prompt File is missing")
"""

"""
코사인 유사도를 이용한 코드


import torch
from PIL import Image
import torchvision.transforms as transforms
from clip import clip

# CLIP 모델 및 전처리기 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 미리 정의된 키워드 리스트
keywords = ["dog", "cat", "bird", "car", "tree", "flower", "building", "sky", "ocean", "mountain"]

# 입력 이미지 로드 및 전처리
image = Image.open("input_image.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)

# 텍스트 인코딩
text_inputs = torch.cat([clip.tokenize(keyword) for keyword in keywords]).to(device)

# 이미지와 텍스트 간 유사도 계산
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    # 코사인 유사도 계산
    similarity = text_features @ image_features.T

# 상위 3개의 유사도 값과 해당 키워드 출력
top_indices = similarity.squeeze().argsort(descending=True)[:3]
for idx in top_indices:
    print(f"Keyword: {keywords[idx.item()]}, Similarity: {similarity.squeeze()[idx]:.4f}")
"""