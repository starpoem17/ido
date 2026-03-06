# Korean Raw 전처리 마스터 플랜

## 0. 목적
이 문서는 `docs/personal/preprocess/preprocess.md`와 데이터셋별 personal 문서를 기준으로,
`data/korean_raw` 원천 데이터를 공통 스키마의 Lance 데이터셋으로 만들고 이후 데이터셋별 결과를 같은 저장소에 append할 수 있도록 상위 정책을 고정한다.

핵심 원칙:
1. `docs/personal`의 수기 규칙을 최우선으로 따른다.
2. 데이터셋별 전처리는 독립 실행 가능해야 한다.
3. 산출물은 모두 같은 6컬럼 스키마를 사용해야 한다.
4. 각 데이터셋 결과는 `data/korean_processed/chunks/`에 순차 append 가능해야 한다.

---

## 1. 공통 계약

### 1.1 Canonical Row Schema
최종 row는 아래 6개 컬럼만 사용한다.

1. `data_type: String`
2. `source: String`
3. `split: String`
4. `content: String | null`
5. `messages: List[Struct{role: String, content: String}] | null`
6. `token_count: int32`

강제 규칙:
- `PT` row는 `content != null`, `messages = null`
- `FT` row는 `messages != null`, `content = null`
- 한 row는 `PT` 또는 `FT` 중 하나만 만족해야 한다.
- 모든 데이터셋은 동일한 PyArrow schema로 캐스팅 가능해야 한다.

### 1.2 통합 저장 전략
출력 루트는 `data/korean_processed/`로 고정한다.

```text
data/
└── korean_processed/
    ├── chunks/
    │   ├── chunk-000001.lance
    │   ├── chunk-000002.lance
    │   └── ...
    ├── _manifests/
    └── _indices/
```

규칙:
- 데이터셋별 전처리 결과를 `data/korean_processed/chunks/`에 순차 append한다.
- 첫 실행 시 저장소가 없으면 생성하고, 이후는 동일 schema append만 허용한다.
- 데이터셋 구분은 경로가 아니라 `source` 컬럼으로 한다.
- `_indices`는 모든 대상 데이터셋 적재가 끝난 뒤 1회만 생성한다.

### 1.3 샤딩 정책
- 목표 크기: `1,073,741,824 bytes` (1GiB)
- 행 단위 샤딩만 허용한다.
- 행 중간 분할은 금지한다.
- 새 row를 추가한 뒤 임계값을 넘으면 그 row까지 포함해 현재 chunk를 닫는다.

### 1.4 공통 토크나이저 정책
- `token_count` 계산용 토크나이저는 `data/tokenizers/korean_bbpe_v1/tokenizer.json`을 사용한다.
- `token_count_version = korean_bbpe_v1`로 기록한다.
- 토크나이저가 바뀌면 기존 append와 혼용하지 않고 새 run으로 관리한다.

### 1.5 직렬화와 token_count 정책
`token_count`는 저장 row 자체가 아니라 실제 학습 입력으로 직렬화한 문자열의 토큰 길이로 계산한다.

공통 규칙:
- Lance에는 구조화된 `content` 또는 `messages`를 저장한다.
- `FT` row는 personal 문서 예시대로 `<|system|>`, `<|user|>`, `<|assistant|>`를 붙여 직렬화한다.
- `PT` row는 데이터셋별 personal 문서에 정의된 concat 규칙으로 직렬화한다.
- `<|bos|>`, `<|eos|>`, `<|eot_id|>`는 현재 personal 예시 직렬화에 포함되지 않으므로 기본 전처리 직렬화에는 넣지 않는다.
- 전처리의 `token_count` 계산 규칙과 학습 로더의 직렬화 규칙은 동일해야 한다.

### 1.6 공통 검증 항목
1. 스키마 검증
- 컬럼 6개와 타입 일치
- `split` 허용값 검증
- `PT/FT` 상호배타 규칙 검증

2. 수량 검증
- 입력 레코드 수 대비 출력 row 수, 제외 row 수, 보정 row 수 집계
- `split`별 집계 일치
- `source`별 집계 일치

3. 샤딩 검증
- chunk별 row 수 합계 = 전체 row 수
- 모든 row가 정확히 하나의 chunk에만 존재

4. 토큰 검증
- `token_count >= 1`
- 직렬화 재계산 결과와 저장된 `token_count` 일치

### 1.7 공통 산출물
1. `chunks/chunk-xxxxxx.lance`
2. `_manifests/manifest-<version>.json`
3. `_manifests/quality_report-<run_id>.json`
4. `_indices/*`

manifest 필수 항목:
- `schema_version`
- `token_count_version`
- run 시각
- chunk별 bytes/rows
- source별 rows
- split별 rows
- PT/FT rows

quality report 필수 항목:
- 제외 사유별 집계
- 보정 이벤트 집계
- 제외/보정 샘플

---

## 2. 멀티세션 계열 공통 규칙
대상:
- `009.전문분야_기술과학_한국어 멀티세션 데이터`
- `010.전문분야_사회과학_한국어 멀티세션 데이터`
- `011.일상대화 한국어 멀티세션 데이터`

### 2.1 입력 경로 패턴
- `data/korean_raw/<dataset>/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/<dataset>/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 2.2 split 매핑
- `Training -> train`
- `Validation -> val`

### 2.3 원본 필드 규칙
- 세션 배열: `sessionInfo`
- 발화 배열: `sessionInfo[*].dialog`
- 역할 키: `dialog[*].speaker`
- 텍스트 키: `dialog[*].utterance`
- role 매핑:
  - `speaker1 -> user`
  - `speaker2 -> assistant`

### 2.4 세션 정규화 규칙
- `utterance`는 좌우 공백을 `strip()`한 뒤 사용한다.
- 빈 문자열 발화가 하나라도 있으면 세션 전체 제외
- 시작 role이 `user`가 아니면 세션 전체 제외
- role 교대가 깨지면 세션 전체 제외
- 마지막이 `user`면 마지막 `user` 턴만 제거하고 세션은 유지
- 마지막 `user` 제거 후 assistant 턴이 1개도 없으면 세션 전체 제외

### 2.5 row 생성 규칙
- 정규화에 성공한 세션마다 Lance에는 `FT` row만 1개 생성한다.
- `data_type = FT`
- `content = null`
- `messages = [system, user, assistant, ...]`
- `system` 문장은 personal 문서에 적힌 고정 문구를 그대로 사용한다.

### 2.6 PT 사용 전략
- 이 계열 데이터의 `PT 시 사용 전략`은 Lance 저장 포맷이 아니라 downstream 학습 직렬화 규칙으로 해석한다.
- `system`을 제외한 user/assistant `content`를 원순서대로 이어 붙여 `PT` 입력 문자열을 만든다.
- 구분자는 단일 공백 `" "`이다.

### 2.7 FT token_count 규칙
- `FT` row는 `messages`를 특수 토큰 포함 문자열로 직렬화한 뒤 토큰화한다.
- 직렬화 순서는 personal 예시대로 `<|system|>...<|user|>...<|assistant|>...`를 따른다.
- `token_count`에는 특수 토큰도 포함한다.

---

## 3. 009.전문분야_기술과학_한국어 멀티세션 데이터

### 3.1 입력 경로
- `data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 3.2 source 및 system 문구
- `source = "009.전문분야_기술과학_한국어 멀티세션 데이터"`
- `system = "당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

### 3.3 출력 규칙
- 정규화에 성공한 세션마다 `FT` row 1개 생성
- `data_type = "FT"`
- `content = null`
- `messages`는 personal 예시와 같은 role 배열

---

## 4. 010.전문분야_사회과학_한국어 멀티세션 데이터

### 4.1 입력 경로
- `data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 4.2 source 및 system 문구
- `source = "010.전문분야_사회과학_한국어 멀티세션 데이터"`
- `system = "당신은 사회과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

### 4.3 출력 규칙
- 009와 동일한 정규화 규칙을 사용
- 정규화에 성공한 세션마다 `FT` row 1개 생성

---

## 5. 011.일상대화 한국어 멀티세션 데이터

### 5.1 입력 경로
- `data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 5.2 source 및 system 문구
- `source = "011.일상대화 한국어 멀티세션 데이터"`
- `system = "당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."`

### 5.3 출력 규칙
- 009와 동일한 정규화 규칙을 사용
- 정규화에 성공한 세션마다 `FT` row 1개 생성

---

## 6. 030.웹데이터 기반 한국어 말뭉치 데이터

### 6.1 입력 경로
- `data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/**/*.json`
- `data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json`

### 6.2 split 매핑
- `1.Training -> train`
- `2.Validation -> val`

### 6.3 row 단위
- `named_entity[*]` 아래의 `title[*]` 항목 하나를 row 단위로 본다.
- 즉 `title` 배열에 제목이 여러 개면 같은 기사 본문을 공유하는 row를 제목 수만큼 생성한다.

### 6.4 source 및 system 문구
- `source = "030.웹데이터 기반 한국어 말뭉치 데이터"`
- `system = "당신은 기사를 읽고 제목을 짓습니다. 내용을 요약하고, 사람들의 눈길을 끄는 제목을 작성합니다."`

### 6.5 row 생성 규칙
- `title[*]` 항목 하나당 Lance에는 `FT` row만 1개 생성한다.
- `data_type = "FT"`
- `content = null`
- `messages = [system, user, assistant]`
- `user` 텍스트는 `content[*].sentence`를 공백 `" "`으로 이어 붙인 문자열이다.
- `assistant` 텍스트는 해당 `title[*].sentence`다.
- 전처리 단계에서는 기사 본문을 자르거나 분할하지 않는다.

### 6.6 PT 사용 전략
- `PT 시 사용 전략`은 Lance 저장 포맷이 아니라 downstream 학습 직렬화 규칙으로 해석한다.
- `assistant` 제목을 앞에 두고, `user` 본문을 뒤에 둔다.
- 구분자는 줄바꿈 `\\n` 하나다.

### 6.7 이상치 규칙
- `named_entity` 누락 또는 비배열이면 파일 스킵
- `content`가 비어 있거나 본문 문장이 모두 빈 문자열이면 해당 제목 row 제외
- `title` 문장이 빈 문자열이면 해당 제목 row 제외
- 문자열이 아닌 `sentence`는 무효값으로 보고 해당 행에서 제외한 뒤, 최종 본문 또는 제목이 비면 row 제외

---

## 7. novel24

### 7.1 입력 경로
- `data/korean_raw/novel24/*.txt`
- 숨김 파일과 `.DS_Store`는 제외한다.

### 7.2 split 및 data_type
- 모든 row는 `split = train`
- 모든 row는 `data_type = PT`

### 7.3 row 단위
- txt 원문을 `\\n` 단위로 누적하면서 row를 만든다.
- 누적 결과의 토큰 길이가 `1024`를 넘기기 직전까지만 현재 row에 넣고, 넘기는 줄부터 다음 row로 넘긴다.
- 문장 중간이나 줄 중간을 자르지 않는다.

### 7.4 source
- `source = "novel24"`

---

## 8. 구현 순서
1. `docs/pseudo/preprocess/preprocess.md`에 공통 오케스트레이터, 직렬화, token_count, Lance append 의사코드를 정의한다.
2. `009`, `010`, `011`은 멀티세션 공통 헬퍼를 공유하는 개별 adapter 의사코드를 작성한다.
3. `030`은 personal 예시 row 형태대로 `FT` row를 생성하는 기사-제목 adapter 의사코드를 작성한다.
4. `novel24`는 `PT` 전용 분할 adapter 의사코드를 작성한다.
5. 구현 시 dataset별 결과를 같은 Lance 저장소에 append한다.

---

## 9. 수용 기준
1. `009`, `010`, `011`, `030`, `novel24`가 모두 personal 문서와 논리적으로 충돌하지 않는다.
2. 각 데이터셋 의사코드만 보고 바로 구현 가능한 수준으로 입력 경로, row 단위, 직렬화, `token_count`, 품질 규칙이 정의돼 있다.
3. 모든 데이터셋 결과가 같은 6컬럼 스키마로 append 가능하다.
4. `FT` 중심 데이터셋과 `PT` 중심 데이터셋이 같은 Lance 저장소에 함께 append 가능하다.
