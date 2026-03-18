# novel24 전처리 스도코드

## 목적
`novel24`를 PT용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source = "novel24"`
2. `data_usage = "PT"`
3. `messages = null`
4. 입력 경로는 `data/korean_raw/novel24/*.txt`
5. 한 txt 파일 내부는 train/val로 나누지 않는다.

## parquet 생성 규칙
1. 숨김 파일과 `.DS_Store`를 제외한 txt 파일 목록을 수집한다.
2. 파일 단위 `split_key`는 `file_stem`을 사용한다.
3. 최종 split은 파일 단위로 99:1 helper를 적용하고, 같은 파일의 모든 chunk는 같은 split을 가진다.
4. txt 내용은 줄 단위로 읽는다.
5. 줄을 누적해 하나의 chunk를 만든다.
6. 줄 중간이나 문장 중간은 자르지 않는다.
7. chunk가 비어 있지 않으면 `content`로 저장한다.

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 `source == novel24` row만 읽는다.
2. 최종 tokenizer로 `content` 기준 `token_count`를 채운다.
3. source별 Lance dataset으로 저장한다.

## 품질 이벤트
- txt read failure
- empty chunk
- chunking helper missing
