# namu 전처리 스도코드

## 목적
`namu` 데이터를 PT용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source`는 원본 row의 source를 유지하되 통계 집계는 `namu`로 묶는다.
2. `data_usage = "PT"`
3. `messages = null`
4. 입력 경로는 `data/korean_raw/namu/*.json`

## parquet 생성 규칙
1. JSON 파일을 읽고, 파일 안이 row 배열이면 각 원소를 처리하고 단일 row면 하나만 처리한다.
2. `split_key`는 원본 row source 우선, 없으면 `file_stem + row_index`다.
3. 기존 row가 이미 canonical 구조와 유사하더라도 `source`, `data_usage`, `split`, `content`, `messages`, `token_count`만 다시 맞춰 저장한다.
4. `content`가 비면 제외한다.
5. 최종 split은 기존 train 라벨을 그대로 쓰지 않고 공통 99:1 helper로 다시 부여한다.

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 `source`가 `namu.`로 시작하는 row를 읽는다.
2. `content` 기준 `token_count`를 채운다.
3. dataset별 Lance는 개별 row source를 유지하되, 통계 요약은 `namu`로 묶는다.

## 품질 이벤트
- json parse failure
- malformed canonical-like row
- empty content
