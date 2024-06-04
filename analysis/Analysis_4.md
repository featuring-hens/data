## 모델 테스트 결과 분석
### 목적
사용한 모델은 CLIP ViT-B/32이며,
진행 및 비교 분석한 테스트는 아래와 같습니다.

<br/>

**1. 숏폼(프레임 5개) > 키워드 추출<br/>**
- Input: 숏폼 프레임 이미지 5개<br/>
- Output: 피처링 정의 카테고리 3개

**2. 숏폼(섬네일) > 키워드 추출<br/>**
- Input: 숏폼 섬네일 이미지 1개<br/>
- Output: 피처링 정의 카테고리 3개

<br/>

### 결과
**특정 영상**
|                              | CLIP: 숏폼(프레임) > 키워드                                         | CLIP: 숏폼(섬네일) > 키워드                     | 
|------------------------------|----------------------------------------------|------------------------------------|
| **숏폼당 평균 소요 시간(초)**             | 5.545                                        | 0.817                             |
| 여행(https://www.videvo.net/video/girl-on-top-with-mobile-phone/884533/#rs=video-box)                        | ['여행', '일상', '기타']                         | ['여행', '일상', '자동차/모빌리티']    |
| 뷰티(https://www.videvo.net/video/a-young-beautiful-woman-in-front-of-a-smartphone-in-video-mode-is-in-the-middle-of-a-social-media-makeup-tutorial/1736965/#rs=video-box) | ['뷰티', '일상', '취미/문화']   | ['일상', '뷰티', '취미/문화']       |
| 반려동물(https://www.videvo.net/video/little-girl-holding-pet-bowl-with-food-and-feeding-dog-at-home/1166918/#rs=video-box)   | ['반려동물', '만화/애니/툰', '홈/리빙']                  | ['반려동물', '결혼/연애', '일상']          |
| 육아/키즈(https://www.videvo.net/video/front-view-of-a-baby-sitting-on-sofa-in-living-room-at-home-while-bitting-a-wooden-bracelet/1116870/#rs=video-box)  | ['육아/키즈', '일상', '만화/애니/툰']              | ['육아/키즈', '일상', '결혼/연애']    |
| 자동차(https://www.videvo.net/video/cars-pass-through-busy-lane-in-intersection/1511381/#rs=video-box)   | ['자동차/모빌리티', '일상', '스타/연예인']                 | ['자동차/모빌리티', '일상', '만화/애니/툰']           |
| F&B(https://www.videvo.net/video/close-up-view-of-a-man-hand-cutting-a-meat-fillet-from-a-plate-with-vegetables-and-potatoes-during-an-outdoor-party-in-the-park/1166861/#rs=video-box)   | ['F&B', '스포츠/운동', '홈/리빙']                | ['F&B', '스포츠/운동', '결혼/연애']           |
| 홈(https://www.videvo.net/video/dolly-in-shot-of-young-woman-peeling-a-potato/595702/#rs=video-box)   | ['홈/리빙', '일상', '취미/문화']                    | ['홈/리빙', '일상', '만화/애니/툰']               |
