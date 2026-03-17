# HAERAE-HUB-HR-Instruct-Math-v0.1 전처리 스도코드

## 목적
`HAERAE-HUB-HR-Instruct-Math-v0.1`을 REASONING용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source = "HAERAE-HUB-HR-Instruct-Math-v0.1"`
2. `data_usage = "REASONING"`
3. `system = "사용자의 질문을 읽고 단계 별로 사고하여 논리적인 답변을 제시합니다."`
4. 입력 경로는 `data/korean_raw/HAERAE-HUB-HR-Instruct-Math-v0.1/*.parquet`

## parquet 생성 규칙
1. parquet row 하나를 canonical row 하나로 본다.
2. `split_key`는 원본 id 우선, 없으면 `file_path + row_index`다.
3. `instruction`을 user, `response`를 assistant로 읽는다.
4. 둘 중 하나라도 비면 제외한다.
5. `messages = [system, user(instruction), assistant(response)]`
6. `content`는 `instruction + "\n" + response` 형태로 만든다.

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 이 source row만 읽는다.
2. `content` 기준 `token_count`를 채운다.
3. REASONING Lance dataset으로 저장한다.

## 품질 이벤트
- missing instruction
- missing response
- parquet read failure
