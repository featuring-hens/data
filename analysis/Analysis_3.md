## 모델 테스트 결과 분석
### 목적
숏폼에서 카테고리를 추출하는 CLIP 테스트 결과입니다.

사용한 모델은 ViT-B/32이며,
입력으로 들어온 영상을 10개의 프레임으로 추출해 테스트를 진행했습니다.

인스타그램, 틱톡, 유튜브 등에서 숏폼이 어떤 형태(이미지, 영상 등)로 들어오냐에 따라 실행 시간은 달라질 수 있을 것 같습니다.

<br/>

### 결과
**특정 영상**
|                              | CLIP                                         | 영상 길이 / 실행 시간                     | 
|------------------------------|----------------------------------------------|------------------------------------|
| 여행(https://www.videvo.net/video/girl-on-top-with-mobile-phone/884533/#rs=video-box)                        | ['여행', '일상', '기타']                         | 21초 / 9.17초    |
| 뷰티(https://www.videvo.net/video/a-young-beautiful-woman-in-front-of-a-smartphone-in-video-mode-is-in-the-middle-of-a-social-media-makeup-tutorial/1736965/#rs=video-box) | ['뷰티', '일상', '취미/문화']   | 12초 / 8.60초       |
| 반려동물(https://www.videvo.net/video/little-girl-holding-pet-bowl-with-food-and-feeding-dog-at-home/1166918/#rs=video-box)   | ['반려동물', '홈/리빙', '만화/애니/툰']                  | 20초 / 11.89초          |
| 육아/키즈(https://www.videvo.net/video/front-view-of-a-baby-sitting-on-sofa-in-living-room-at-home-while-bitting-a-wooden-bracelet/1116870/#rs=video-box)  | ['육아/키즈', '일상', '만화/애니/툰']              | 20초 / 13.05초    |
| 자동차(https://www.videvo.net/video/cars-pass-through-busy-lane-in-intersection/1511381/#rs=video-box)   | ['자동차/모빌리티', '일상', '스타/연예인']                 | 16초 / 9.19초           |
| F&B(https://www.videvo.net/video/close-up-view-of-a-man-hand-cutting-a-meat-fillet-from-a-plate-with-vegetables-and-potatoes-during-an-outdoor-party-in-the-park/1166861/#rs=video-box)   | ['F&B', '홈/리빙', '결혼/연애']                | 13초 / 11.90초           |
| 홈(https://www.videvo.net/video/dolly-in-shot-of-young-woman-peeling-a-potato/595702/#rs=video-box)   | ['홈/리빙', '일상', '만화/애니/툰']                    | 17초 / 11.23초               |

*영상당 평균 소요 시간: 10.72초