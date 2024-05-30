# CLIP

## CLIP (Contrastive Language-Image Pre-Training)
- OpenAI에서 개발
- 텍스트와 이미지를 연결하여 텍스트 설명과 이미지 간의 유사성을 학습

<br/><br/>

## 모델 종류

### ResNet 기반 모델
- **RN50**: ResNet-50 아키텍처 사용

- **RN101**: ResNet-101 아키텍처 사용

- **RN50x4**: ResNet-50 아키텍처 4배 확장

- **RN50x16**: ResNet-50 아키텍처 16배 확장

- **RN50x64**: ResNet-50 아키텍처 64배 확장

<br/>

### Vision Transformer 기반 모델
- **ViT-B/32**: 32x32 패치 크기를 사용하는 Base 크기의 Vision Transformer

- **ViT-B/16**: 16x16 패치 크기를 사용하는 Base 크기의 Vision Transformer

- **ViT-L/14**: 14x14 패치 크기를 사용하는 Large 크기의 Vision Transformer

<br/><br/>

## 모델 특징

### ResNet 기반 모델
- **RN50**: 
  - 기본적인 ResNet-50 아키텍처
  - 상대적으로 가벼운 연산량과 메모리 사용량

- **RN101**: 
  - ResNet-101 아키텍처
  - RN50에 비해 더 깊은 네트워크로, 높은 표현력 제공

- **RN50x4**: 
  - ResNet-50 아키텍처의 4배 확장 버전
  - 더 많은 매개변수를 통해 높은 성능 제공

- **RN50x16**: 
  - ResNet-50 아키텍처의 16배 확장 버전
  - 매우 높은 성능 제공하지만, 연산량과 메모리 사용량이 큼

- **RN50x64**: 
  - ResNet-50 아키텍처의 64배 확장 버전
  - 최고의 성능 제공, 매우 큰 연산량과 메모리 사용량 요구

<br/>

### Vision Transformer 기반 모델
- **ViT-B/32**: 
  - Base 크기의 Vision Transformer
  - 32x32 패치 크기 사용
  - 상대적으로 가벼운 연산량

- **ViT-B/16**: 
  - Base 크기의 Vision Transformer
  - 16x16 패치 크기 사용
  - 더 높은 해상도의 패치 처리 가능

- **ViT-L/14**: 
  - Large 크기의 Vision Transformer
  - 14x14 패치 크기 사용
  - 매우 높은 표현력 제공
