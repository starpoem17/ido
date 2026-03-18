# HAERAE-HUB-KOREAN-WEBTEXT 전처리 스도코드

## 목적
`HAERAE-HUB-KOREAN-WEBTEXT`를 PT용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source = "HAERAE-HUB-KOREAN-WEBTEXT"`
2. `data_usage = "PT"`
3. `messages = null`
4. 입력 경로는 `data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/*.parquet`

## parquet 생성 규칙
1. parquet row 하나를 canonical row 하나로 본다.
2. `split_key`는 원본 row id 우선, 없으면 `file_path + row_index`다.
3. `text` 필드만 읽는다.
4. `text`가 비면 제외한다.
5. `content = text`
6. 원본 `token_count` 필드는 학습 row 값으로 사용하지 않는다.

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 이 source row만 읽는다.
2. `content` 기준 `token_count`를 채운다.
3. PT-only Lance dataset으로 저장한다.

## 품질 이벤트
- parquet read failure
- missing text
- empty text
