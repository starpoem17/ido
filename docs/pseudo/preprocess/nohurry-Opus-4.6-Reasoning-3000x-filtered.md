# nohurry-Opus-4.6-Reasoning-3000x-filtered 전처리 스도코드

## 목적
`nohurry-Opus-4.6-Reasoning-3000x-filtered`를 REASONING용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source = "nohurry-Opus-4.6-Reasoning-3000x-filtered"`
2. `data_usage = "REASONING"`
3. `system = "사용자의 질문을 읽고 단계 별로 사고하여 논리적인 답변을 제시합니다."`
4. 입력 경로는 `data/korean_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/*.jsonl`

## parquet 생성 규칙
1. jsonl 각 줄을 row 단위로 처리한다.
2. `split_key`는 `id` 우선, 없으면 `file_path + line_index`다.
3. `problem`을 user 입력으로 사용한다.
4. `thinking + solution`을 assistant 응답으로 이어 붙인다.
5. 핵심 필드가 비면 제외한다.
6. `messages = [system, user(problem), assistant(thinking + solution)]`
7. `content = problem + "\n" + thinking + "\n" + solution`

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 이 source row만 읽는다.
2. `content` 기준 `token_count`를 채운다.
3. REASONING Lance dataset으로 저장한다.

## 품질 이벤트
- jsonl parse failure
- missing problem
- missing reasoning text
