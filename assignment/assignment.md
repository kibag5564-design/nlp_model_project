# assignment.md
제공된 네이버 영화 리뷰 데이터 ratings.txt 를 로드하여
기존에 작성된 lstm_imdb_lightning.py 의 코드를 커스터마이징 하여
./assignment/LSTM_movie_review.py 로 작성하시오.

# 영화 리뷰 감성 분류 모델 만들어 학습 및 평가

# 깃허브에 nlp_model_project 리포지토리 생성하고, 프로젝트 최초 커밋함
# 해당 리포지토리에서 브렌치 2개 추가 : team01, team02

# 해당 프로젝트 터미널 (또는 Git 메뉴 또는 Git 뷰)에서  현재 브렌치 확인 : *main
git branch
# 깃허브 리포지토리에서 추가한 브렌치 목록 확인
git branch -a
출력:
*main
remotes/origin/team01
remotes/origin/team02

# team01 브렌치로 변경 (checkout)
git checkout -b team01 origin/team01
또는
git switch -c team01 origin/teamo1

# 현재 브렌치 확인 : team01
git branch
# team01 브렌치에서 아래의 추가 작업을 진행

[추가 작업 진행]
1. 프로젝트 루트에 assignment 패키지 폴더 만듦
2. assignment 폴더 아래에 LSTM_movie_review.py 만듦  <= lstm_imdb_lightning.py 파일 복사해서 rename 함
3. 네이버 영화 리뷰 데이터로 감성 분석 모델로 수정해서 학습 및 평가하고 실행 완료함
4. 브렌치에서 작업한 내용을 깃허브 리포지토리로 커밋 (메세지 추가) > 푸시함

git commit -m "LSTM 네이버 영화 리뷰 감성 분석 모델 추가"
git push origin team01

5. 깃허브 리포지토리에 브렌치 이름 선택하면, 푸시한 정보 확인할 수 있음
6. 