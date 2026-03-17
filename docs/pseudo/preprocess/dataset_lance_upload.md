# dataset별 Lance 업로드 단계 스도코드

## 목적
이 문서는 minhash dedup 완료 parquet를 dataset별 Lance dataset으로 올리는 실행 코드의 자연어 의사코드다.
이 단계에서 토크나이저를 사용해 `content` 기준 `token_count`를 채운다.

## 사용자 설정값
1. `IN_ROOT = "data/korean_processed/_staging/minhash_dedup"`
2. `TOKENIZER_PATH = "data/tokenizers/korean_bbpe_v1/tokenizer.json"`
3. `OUT_ROOT = "data/korean_processed/lance_by_dataset"`
4. `TARGET_SHARD_BYTES = 1GB`
5. `NUM_WORKERS = 7`

## 전체 실행 절차
1. minhash dedup 완료 parquet를 source별로 읽는다.
2. 각 row의 `content`를 tokenizer로 인코딩해 `token_count`를 채운다.
3. `token_count`가 null이거나 1 미만이면 오류로 처리한다.
4. source별로 독립 Lance dataset을 만든다.
5. shard 크기가 1GB를 넘기면 방금 추가한 row까지 포함해 shard를 닫는다.
6. dataset별 적재 통계와 shard 메타를 저장한다.

## token_count 채우기
1. `TOKENIZER_PATH`를 한 번만 로드한다.
2. row의 `content`를 그대로 인코딩한다.
3. 인코딩 결과 토큰 수를 `token_count`에 넣는다.
4. `messages`는 token_count 계산에 사용하지 않는다.

## dataset별 분리
1. `source`를 기준으로 row를 묶는다.
2. 한 source는 하나의 Lance dataset 디렉터리로 올린다.
3. 저장 경로명은 source를 안전한 파일명으로 바꿔 사용한다.

## 산출물
1. source별 Lance dataset 디렉터리
2. source별 shard 통계 JSON
3. source별 row 수, token 수, shard 수 요약 JSON
