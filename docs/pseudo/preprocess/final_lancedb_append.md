# final lancedb append 단계 스도코드

## 목적
이 문서는 dataset별 Lance dataset 디렉터리들을 하나의 최종 LLM 학습용 LanceDB로 합치는 실행 코드의 자연어 의사코드다.

## 사용자 설정값
1. `IN_ROOT = "data/korean_processed/lance_by_dataset"`
2. `OUT_ROOT = "data/korean_processed/final_lancedb"`
3. `MANIFEST_ROOT = "data/korean_processed/final_lancedb/_manifests"`
4. `INDICES_ROOT = "data/korean_processed/final_lancedb/_indices"`

## 전체 실행 절차
1. `IN_ROOT` 아래의 dataset별 Lance dataset 디렉터리를 안정적인 순서로 수집한다.
2. 첫 dataset으로 최종 LanceDB를 생성한다.
3. 이후 dataset들은 schema 검증 후 순차 append한다.
4. append가 끝나면 source별 row 수와 token 수를 집계한다.
5. data_usage별 row 수, split별 row 수, `content/messages` 보유 row 수를 집계한다.
6. manifest를 저장한다.
7. indices를 생성한다.

## manifest 기록
최소한 아래를 저장한다.
- 실행 시각
- 입력 dataset 디렉터리 목록
- source별 row 수
- source별 총 token 수
- data_usage별 row 수
- split별 row 수
- `content` 보유 row 수
- `messages` 보유 row 수
- dataset별 shard 수

## 최종 검증
1. append된 모든 row가 canonical schema를 만족하는지 확인한다.
2. dataset별 row 합계와 최종 row 합계가 일치하는지 확인한다.
3. `namu` 통계는 하나의 source로 묶어서 별도 집계한다.
4. indices와 manifest가 모두 생성되었는지 확인한다.
