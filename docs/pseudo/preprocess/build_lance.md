# Lance 적재 단계 스도코드

## 목적
이 문서는 `near_dedup` 완료 parquet를 읽어 Lance 적재 직전 규칙을 수행하고,
source별 shard 작성과 final Lance append까지 처리하는 실행 코드의 자연어 의사코드다.
상위 기준은 [docs/plans/lance.md](/Users/nojonghyeon/Documents/GitHub/ido/docs/plans/lance.md)다.

## 사용자 설정값
1. `INPUT_ROOT = "data/korean_processed/near_dedup"`
2. `OUTPUT_ROOT = "data/korean_processed/final_lancedb"`
3. `TEMP_SOURCE_ROOT = "data/korean_processed/lance_by_source"`
4. `TOKENIZER_JSON_PATH`
5. `SOURCE_SHARD_TARGET_BYTES = 1GB`
6. `SFT_MAX_TOKENS = 1024`
7. `PT_TARGET_TOKENS = 960`
8. `PT_MAX_TOKENS = 1024`
9. `READ_BATCH_ROWS`
10. `OVERWRITE_OUTPUT`
11. `ENABLE_TQDM`, `ENABLE_DEBUG_LOG`, `TQDM_MININTERVAL_SEC`

## 입력과 출력 경로
1. 입력 parquet는 `data/korean_processed/near_dedup/part-*.parquet`를 사용한다.
2. source별 임시 shard는 `data/korean_processed/lance_by_source/<source_hash>/part-*.parquet`에 저장한다.
3. 최종 Lance dataset은 `data/korean_processed/final_lancedb/dataset.lance`에 저장한다.
4. manifest는 `data/korean_processed/final_lancedb/_meta/build_lance_manifest.json`에 저장한다.
5. progress 메타는 `data/korean_processed/final_lancedb/_meta/build_lance_progress.json`에 저장한다.

## 전체 실행 절차
1. 입력 root, 출력 root, tokenizer 경로, shard 목표 크기를 stdout 로그로 출력한다.
2. 입력 shard 목록을 `part-*.parquet` 기준 안정 순서로 수집한다.
3. 출력 root와 임시 source root를 준비한다. 기존 결과가 있으면 `OVERWRITE_OUTPUT` 정책으로 처리한다.
4. tokenizer를 `TOKENIZER_JSON_PATH`에서 한 번 로드한다.
5. 첫 입력 shard에 대해 `pyarrow` probe를 시도한다.
6. `pyarrow` probe가 성공하면 `pyarrow` reader를 사용하고, 실패하면 `duckdb read_parquet(...)` reader를 사용한다.
7. 전체 입력에서 distinct `source` 목록을 안정 순서로 수집한다.
8. 각 `source`를 순서대로 처리하며 source별 Lance row를 만든다.
9. source 처리 중 변환된 row의 예상 byte 크기를 누적하고, 다음 row를 추가하면 목표 shard 크기를 넘길 때 flush한다.
10. flush된 source shard를 임시 parquet로 저장하고 progress 메타를 갱신한다.
11. 모든 source shard 작성이 끝나면 source 순서대로 임시 shard를 다시 읽어 final Lance dataset에 append한다.
12. append가 끝나면 manifest와 최종 progress 메타를 저장하고 임시 source root를 정리한다.

## reader 선택 절차
1. 입력 shard 전체를 안정 순서로 순회하며 `pyarrow.ParquetFile(shard).iter_batches(columns=canonical_names, batch_size=1)`를 한 번씩 시도한다.
2. 모든 shard probe가 성공하면 `reader_backend = "pyarrow"`로 고정한다.
3. shard 하나라도 실패하면 오류 로그를 남기고 `reader_backend = "duckdb"`로 고정한다.
4. `pyarrow` reader는 입력 shard를 순서대로 열고 batch를 읽어 `source`가 맞는 row만 통과시킨다.
5. `duckdb` reader는 `SELECT source, data_usage, split, content, messages, token_count FROM read_parquet(...) WHERE source = ?`로 source별 row를 batch 단위로 읽는다.
6. `duckdb`가 반환한 nested `messages`는 파이썬 list-of-dict 구조로 정규화한다.

## row 변환 절차

### 1. 공통 분기
1. row의 `data_usage`를 읽는다.
2. `SFT`면 SFT 변환 절차로 보낸다.
3. `PT`면 PT 변환 절차로 보낸다.
4. `REASONING`이면 REASONING 변환 절차로 보낸다.
5. 지원하지 않는 `data_usage`는 즉시 실패한다.

### 2. SFT 변환
1. `messages`가 비어 있지 않은 list인지 확인한다.
2. 선행 `system` 메시지들을 분리하고, 나머지 dialogue 메시지를 `user-assistant` pair로 묶는다.
3. dialogue가 비었거나 마지막 turn이 `assistant`가 아니면 실패한다.
4. pair를 평탄화해 `system + dialogue` 순서의 message list를 만든다.
5. 아래 형식으로 직렬화한다.
   - `<|bos|>` + 각 message의 `<|role|>content<|eot_id|>` + `<|eos|>`
6. serialized `messages` 기준 token count를 계산한다.
7. `token_count <= 1024`이면 Lance row 1개를 생성한다.
8. `token_count > 1024`이면 `ceil(token_count / 1024)`를 기준으로 목표 chunk 수를 잡는다.
9. pair 개수를 최대한 균등하게 나눠 chunk 목록을 만든다.
10. 각 chunk에 대해 system 메시지를 다시 앞에 붙여 독립 직렬화한다.
11. 각 chunk의 `content`는 해당 chunk에 포함된 `user`, `assistant` 발화를 공백 한 칸으로 이어 붙인다.
12. chunk가 여전히 `1024`를 넘으면 같은 규칙으로 재귀 분할한다.

### 3. PT 변환
1. `content`가 비어 있지 않은 문자열인지 확인한다.
2. `content` 기준 token count를 계산한다.
3. `token_count <= 1024`이면 Lance row 1개를 생성한다.
4. `token_count > 1024`이면 `ceil(token_count / 960)`를 기준으로 목표 chunk 수를 잡는다.
5. `.` `?` `!` `\n` 경계 우선으로 text unit 목록을 만든다.
6. target chunk 수를 참고해 unit들을 순서대로 묶으며 각 chunk가 대략 `960~1024` 범위에 들어가게 조정한다.
7. 문장 경계만으로도 `1024` 이하가 되지 않으면 더 작은 단어 또는 문자 단위로 hard split한다.
8. 분할된 각 row는 `messages = null`로 저장한다.
9. 각 row의 `token_count`는 분할된 `content` 기준으로 다시 계산한다.

### 4. REASONING 변환
1. `content`와 `messages`가 모두 유효한지 확인한다.
2. `PT` 파생 row를 만든다.
   - `data_usage = PT`
   - `content = 원본 content`
   - `messages = null`
   - PT 변환 절차를 그대로 적용한다.
3. `REASONING` 파생 row를 만든다.
   - `data_usage = REASONING`
   - `content = null`
   - `messages = SFT와 같은 직렬화`
   - 별도 분할 없이 `token_count(messages)`를 계산한다.
4. 두 파생 row는 모두 원본 `source`, `split`을 유지한다.

## shard 작성 절차
1. source별 buffer row 목록과 현재 추정 byte 크기를 유지한다.
2. 변환된 output row 하나가 생길 때마다 문자열 필드들의 UTF-8 byte 길이 합으로 예상 byte를 계산한다.
3. 현재 buffer가 비어 있지 않고, 다음 row를 넣으면 `SOURCE_SHARD_TARGET_BYTES`를 넘는다면 현재 buffer를 flush한다.
4. flush 시 Lance 최종 schema의 parquet table로 변환해 `part-{shard_index:06d}.parquet` 이름으로 저장한다.
5. shard별 `row_count`, `byte_size`, `path`를 메타에 기록한다.
6. flush 후 새 shard buffer를 연다.

## final Lance append 절차
1. source별 임시 shard 디렉터리를 source 안정 순서로 순회한다.
2. 각 source의 `part-*.parquet`를 shard 순서대로 읽는다.
3. 첫 shard는 `mode="create"`로 Lance dataset을 생성한다.
4. 이후 shard는 `mode="append"`로 Lance dataset에 추가한다.
5. shard append가 끝날 때마다 progress 메타를 갱신한다.

## row 검증 규칙
1. `data_usage`는 `PT`, `SFT`, `REASONING`만 허용한다.
2. `split`은 `train`, `val`만 허용한다.
3. `content`와 `messages`는 둘 다 null일 수 없다.
4. `PT` row는 `content != null`, `messages == null`이어야 한다.
5. `SFT`, `REASONING` row는 `messages != null`이어야 한다.
6. `token_count`는 PT는 `content`, SFT/REASONING은 serialized `messages` 기준 재계산 값과 일치해야 한다.

## manifest와 로그
1. manifest에는 `reader_backend`, `input_shard_count`, `input_row_count`, `output_row_count`, `source_count`, source별 통계, data_usage별 통계, split별 통계, shard별 메타를 남긴다.
2. progress 메타에는 `phase`, `last_completed_source`, `last_completed_shard_index`, `final_dataset_uri`, `temp_source_root`, `updated_at`를 남긴다.
3. 로그에는 reader probe 성공/실패, source 시작/종료, shard flush, final append 진행, manifest 저장 완료를 남긴다.

## 테스트 시나리오
1. SFT row가 1024 이하일 때 직렬화만 하고 분할하지 않는지 확인한다.
2. SFT row가 1024 초과일 때 `user-assistant` pair 기준으로 균등 분할되는지 확인한다.
3. SFT chunk의 `content`가 system을 제외한 user/assistant만 포함하는지 확인한다.
4. PT row가 1024 초과일 때 문장 경계 우선으로 분할되는지 확인한다.
5. REASONING row가 `PT` 파생 row와 `REASONING` 파생 row로 나뉘는지 확인한다.
6. 작은 shard byte 설정에서 source별 flush와 progress 파일 갱신이 일어나는지 확인한다.
7. `pyarrow` probe 실패 상황에서 `duckdb` reader가 선택되는지 확인한다.
