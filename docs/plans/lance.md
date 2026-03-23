# Korean Lance 적재 계획

## 0. 문서 목적
이 문서는 [docs/personal/preprocess/lance에 넣기전.md](/Users/nojonghyeon/Documents/GitHub/ido/docs/personal/preprocess/lance에%20넣기전.md),
[docs/personal/preprocess/lance에 추가.md](/Users/nojonghyeon/Documents/GitHub/ido/docs/personal/preprocess/lance에%20추가.md),
[docs/plans/preprocess.md](/Users/nojonghyeon/Documents/GitHub/ido/docs/plans/preprocess.md)를 기준으로
`near_dedup` parquet를 최종 Lance 데이터셋으로 변환하는 계획을 고정한다.

핵심 원칙:
1. 입력은 `data/korean_processed/near_dedup/part-*.parquet`로 고정한다.
2. Lance 적재 직전 단계에서만 `messages`를 list에서 serialized string으로 바꾼다.
3. PT는 `content` 기준, SFT와 REASONING은 serialized `messages` 기준으로 최종 `token_count`를 계산한다.
4. SFT와 PT는 Lance 적재 직전 길이 제한에 따라 row를 분할한다.
5. REASONING row는 Lance 적재 직전 `PT` 파생 row와 `REASONING` 파생 row로 분리한다.
6. shard 목표 크기는 `1GB`다.
7. append 단위는 `source`별로 고정한다.
8. 실패 재시작 단위는 shard다.

---

## 1. 입력과 출력

### 1.1 입력
- 입력 root는 `data/korean_processed/near_dedup`
- 입력 파일은 `part-*.parquet`
- 입력 row는 canonical parquet schema를 따른다.
  1. `source: string`
  2. `data_usage: string`
  3. `split: string`
  4. `content: string | null`
  5. `messages: list<struct<role, content>> | null`
  6. `token_count: int32 | null`

### 1.2 출력
- 중간 source별 parquet shard는 `data/korean_processed/lance_by_source/<source>/part-*.parquet`에 저장한다.
- 최종 Lance dataset은 `data/korean_processed/final_lancedb/dataset.lance`에 저장한다.
- 실행 메타데이터는 `data/korean_processed/final_lancedb/_meta/` 아래에 저장한다.

### 1.3 Lance 최종 schema
최종 Lance row는 아래 schema를 사용한다.

1. `source: string`
2. `data_usage: string`
3. `split: string`
4. `content: string | null`
5. `messages: string | null`
6. `token_count: int32`

강제 규칙:
- `content`와 `messages` 중 최소 하나는 반드시 채운다.
- `PT` row는 `messages = null`
- `SFT`, `REASONING` row는 `messages = serialized string`
- 최종 Lance row의 `token_count`는 parquet 단계 값과 무관하게 다시 계산한다.

---

## 2. Lance 직전 변환 규칙

### 2.1 SFT
- parquet의 `messages` list를 아래 형식으로 직렬화한다.
  - `<|bos|><|system|>...<|eot_id|><|user|>...<|eot_id|><|assistant|>...<|eot_id|><|eos|>`
- 직렬화된 `messages` 기준으로 `token_count`를 계산한다.
- `token_count <= 1024`이면 그대로 1 row로 유지한다.
- `token_count > 1024`이면 `ceil(token_count / 1024)`로 목표 row 수를 잡는다.
- 분할 단위는 토큰 greedy가 아니라 `user-assistant pair` 개수 기준 균등 분할을 우선한다.
- 각 chunk는 반드시 `assistant` turn으로 끝나야 한다.
- 균등 분할 후에도 `1024` 초과 chunk가 있으면 같은 규칙으로 재귀 분할한다.
- 분할된 각 row의 `content`는 해당 chunk에 포함된 `user`, `assistant` 발화만 원순서대로 공백 한 칸으로 이어 붙인다.
- 분할된 각 row의 `messages`는 원래 system 메시지를 다시 앞에 붙여 독립 직렬화한다.

### 2.2 PT
- `content` 기준으로 `token_count`를 계산한다.
- `token_count <= 1024`이면 그대로 유지한다.
- `token_count > 1024`이면 `ceil(token_count / 960)`를 기준으로 목표 chunk 수를 계산한다.
- 분할은 온점, 물음표, 느낌표, 줄바꿈 경계를 우선 사용한다.
- 목표는 각 chunk가 대략 `960~1024` 범위에 들어가도록 하는 것이다.
- 문장 경계만으로 해결되지 않으면 더 작은 텍스트 단위로 추가 분할한다.
- 분할된 각 row는 `messages = null`이고, `token_count`는 분할된 `content` 기준으로 다시 계산한다.

### 2.3 REASONING
- 원본 1 row를 Lance 적재 직전에 2계열 row로 나눈다.
- `PT` 파생 row:
  - `data_usage = PT`
  - `content = 원본 content`
  - `messages = null`
  - PT와 동일한 `1024 / 960` 분할 규칙을 적용한다.
  - `token_count`는 `content` 기준으로 계산한다.
- `REASONING` 파생 row:
  - `data_usage = REASONING`
  - `content = null`
  - `messages = SFT와 같은 serialized string`
  - 별도 분할하지 않는다.
  - `token_count`는 serialized `messages` 기준으로 계산한다.
- 두 파생 row 모두 원본 `source`, `split`을 유지한다.

---

## 3. 읽기, 샤딩, append 정책

### 3.1 읽기 정책
- 기본 reader 시도 순서는 `pyarrow`다.
- 입력 shard 전체에 대한 `pyarrow` probe가 하나라도 실패하면 전체 입력을 `duckdb read_parquet(...)` 경로로 읽는다.
- reader 선택은 실행 시작 시 1회 결정하고 manifest에 남긴다.
- 현재 known risk는 nested `messages` 컬럼이 있는 일부 parquet에서 `pyarrow`가 `Repetition level histogram size mismatch`를 낼 수 있다는 점이다.

### 3.2 source별 shard 작성
- 입력 row를 `source` 기준으로 그룹화해 처리한다.
- 변환된 Lance row는 source별 임시 parquet shard로 먼저 저장한다.
- row를 하나씩 변환하면서 예상 byte 크기를 계산한다.
- 현재 shard에 row를 계속 담다가 `1GB`를 넘기기 직전에 flush한다.
- row 중간 분할은 허용하지 않는다.
- flush 시 source별 shard 메타데이터를 즉시 기록한다.

### 3.3 final Lance append
- source별 임시 parquet shard를 안정 순서로 읽는다.
- 첫 shard는 `create`, 이후 shard는 `append` 모드로 Lance dataset에 기록한다.
- append 완료 후 임시 source shard 디렉터리는 정리한다.
- 재시작 시에는 progress 메타파일을 기준으로 마지막 완료 source와 shard를 확인할 수 있어야 한다.

---

## 4. 메타데이터와 검증

### 4.1 메타데이터
`build_lance_manifest.json`에는 최소 아래를 남긴다.
- `input_shard_count`
- `input_row_count`
- `output_row_count`
- `reader_backend`
- `tokenizer_json_path`
- `source_stats`
- `data_usage_counts`
- `split_counts`
- `output_shards`
- `final_dataset_uri`
- `started_at`
- `finished_at`
- `duration_sec`

`build_lance_progress.json`에는 최소 아래를 남긴다.
- `phase`
- `last_completed_source`
- `last_completed_shard_index`
- `final_dataset_uri`
- `temp_source_root`
- `updated_at`

### 4.2 검증 기준
1. SFT row는 Lance 최종 row에서 `messages`가 list가 아니라 string이어야 한다.
2. SFT split row는 항상 `assistant` turn으로 끝나야 한다.
3. PT split row는 모두 `messages = null`이어야 한다.
4. REASONING row는 `PT` 파생 row와 `REASONING` 파생 row가 모두 생성되어야 한다.
5. 최종 `token_count`는 PT는 `content`, SFT/REASONING은 serialized `messages` 기준이어야 한다.
6. source별 shard byte 크기와 row 수가 manifest에 기록되어야 한다.
7. final Lance append는 source별 안정 순서로 수행되어야 한다.

---

## 5. 테스트 계획
- SFT 짧은 row가 분할 없이 직렬화되는지 검증
- SFT 긴 row가 `user-assistant pair` 기준 균등 분할되는지 검증
- SFT 분할 row의 `content`에서 system이 제외되는지 검증
- PT 긴 row가 문장 경계 우선으로 분할되는지 검증
- REASONING row가 `PT`와 `REASONING` 두 계열로 파생되는지 검증
- REASONING 파생 row의 `content`와 `messages` nullability가 맞는지 검증
- `pyarrow` probe 실패 시 `duckdb` reader가 선택되는지 검증
- 작은 shard target byte를 사용해 source별 shard flush가 동작하는지 검증
- progress/manifest 파일이 생성되는지 검증
