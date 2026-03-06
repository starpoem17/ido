# Korean Raw 전처리 마스터 플랜 (확장형 문서 구조)

## 문서 목적
이 문서는 `data/korean_raw`의 다양한 데이터 형식을 공통 학습 포맷으로 변환하기 위한 전처리 계획서다.  
문서 구조를 데이터셋별로 분리해, `009`뿐 아니라 `010`, `011`, `019` 및 나머지 데이터셋 계획을 동일한 틀에서 계속 확장할 수 있도록 설계한다.
009(009.전문분야_기술과학_한국어 멀티세션 데이터)를 기본 틀로 잡고 이후 나머지 데이터들은 009와 같은 데이터 형식으로 진행되도록 방향을 잡는다

본 문서는 정책/논리/검증 기준을 정의하며, 코드 구현 내용은 포함하지 않는다.

---

## 1. 전체 데이터 공통 전략

### 1.1 공통 목표 스키마 (Canonical Row Schema v1)
`docs/personal/preprocess.md` 기준으로 최종 저장 스키마를 고정한다.

1. `data_type: String` (`PT` 또는 `FT`)
2. `source: String`
3. `split: String` (`train`, `val`)
4. `content: String | null`
5. `messages: List[Struct{role: String, content: String}] | null`
6. `token_count: int32`

공통 강제 조건:
- 모든 데이터셋은 하나의 통합 Lance 저장소로 병합할 수 있도록 동일한 PyArrow schema를 적용한다.
- `token_count`는 데이터셋별 예외 없이 동일한 단일 로직으로 계산한다.
- `token_count` 계산은 사전 생성된 공통 토크나이저 `data/tokenizers/korean_bbpe_v1/tokenizer.json`을 사용한다.
- 스키마/토큰 카운트 규칙 변경 시 `schema_version`, `token_count_version`을 함께 갱신한다.

규칙:
- `PT` row: `content` 사용, `messages = null`
- `FT` row: `messages` 사용, `content = null`
- 하나의 row는 `PT` 또는 `FT` 중 하나만 충족해야 한다.

### 1.2 공통 파이프라인 (데이터셋별 전처리 + Lance append 통합)
데이터셋은 개별 단위로 전처리한다.  
각 데이터셋의 Canonical row를 검증한 뒤, 기존 통합 Lance 저장소에 순차 append하여 최종적으로 하나의 저장소로 합친다.
즉, "전처리 단위는 데이터셋별", "저장소는 통합 1개" 원칙을 유지한다.

1. `Dataset Adapter Normalize` (데이터셋별 실행)
- 데이터셋별 Adapter가 원본 구조를 Canonical row 후보로 평탄화한다.
- split/source/data_type 매핑과 품질 필터는 Adapter 내부 규칙으로 처리한다.
- Adapter 출력은 공통 contract(`data_type, source, split, content, messages, token_count`)를 따른다.

2. `DuckDB/Arrow Canonicalize` (데이터셋별 검증)
- 현재 처리 중인 데이터셋 출력만 relation으로 구성한다.
- 공통 PyArrow schema와 null 규칙을 동일하게 검증한다.

3. `Lance Append Persist`
- 검증된 데이터셋 row를 통합 Lance 저장소(`data/korean_processed/chunks`)에 append한다.
- 첫 데이터셋 append 시 저장소가 없으면 생성하고, 이후 데이터셋은 같은 스키마로 append한다.
- chunk(shard)는 데이터셋 경계와 무관하게 바이트 기준으로만 마감한다.

### 1.3 공통 샤딩 정책
- 샤드 목표 크기: `1,073,741,824 bytes` (1GiB)
- 샤딩은 행 단위로만 수행한다.
- 행 중간 분할은 금지한다.
- 임계값 초과 직전이라도 다음 행 추가 후 초과되면 그 행까지 포함해 해당 샤드를 종료한다.

예시:
- 현재 `999MB`, 다음 row 반영 후 `1005MB`가 되면 `1005MB`로 shard 종료

### 1.4 공통 검증 항목
1. 스키마 검증
- 컬럼 6개 및 타입 일치
- `split` 허용값 검증
- `PT/FT` 상호 배타 규칙 검증

2. 수량 검증
- 입력 row 수 대비 출력 row 수 및 제외 row 수 일치
- split별 집계 일치

3. 샤딩 검증
- shard 합계 row 수 = 전체 row 수
- 행 원자성 위반 없음

### 1.5 공통 산출물
통합 출력 루트: `data/korean_processed/`

권장 구조:
```text
data/
└── korean_processed/
    ├── chunks/
    │   ├── chunk-000001.lance
    │   ├── chunk-000002.lance
    │   └── ...
    ├── _indices/
    └── _manifests/
```

1. `chunks/chunk-xxxxxx.lance`
- 1GiB 내외 행 단위 샤드
- 서로 다른 source 데이터가 동일 저장소에 공존 가능
2. `_manifests/manifest-<version>.json`
- `schema_version`, `token_count_version`, chunk별 bytes/rows, split/source 집계
3. `_manifests/quality_report-<run_id>.json`
- 제외 규칙/제외 건수/샘플 파일 경로
4. `_indices/*`
- 검색/필터용 인덱스 산출물
- 모든 대상 데이터셋 적재가 끝난 뒤 한 번만 생성한다.

### 1.6 데이터셋 전략 섹션 템플릿
아래 템플릿으로 각 데이터셋 섹션을 갱신한다.

1. 입력 경로 및 split 매핑
2. row 단위 정의
3. `PT/FT` 매핑 규칙
4. `messages/content` 구성 규칙
5. 이상치 정책
6. token_count 기준
7. 예상 산출량/리스크/확정 여부

### 1.7 공통 토크나이저 정책 (신규 반영)
- 토크나이저 아티팩트는 `data/tokenizers/korean_bbpe_v1/`를 기준으로 고정한다.
- 최소 사용 파일:
1. `tokenizer.json`
2. `vocab.json`
3. `merges.txt`
- `token_count_version`은 토크나이저 식별자(`korean_bbpe_v1`)와 결합해 관리한다.
- 후속 데이터셋(010/011/019 및 백로그 포함)도 동일 토크나이저를 사용해 `token_count` 메타데이터 일관성을 유지한다.

---

## 2. 009.전문분야_기술과학_한국어 멀티세션 데이터 전략 (확정)

### 2.1 입력 경로
- `data/korean_raw/009.../3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json`
- `data/korean_raw/009.../3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json`

### 2.2 split 매핑
- `Training -> train`
- `Validation -> val`

### 2.3 row 단위 정의
- `sessionInfo.dialog`의 세션 1개를 1 row로 생성한다.
- `messages`는 가변 길이이며 `system` 1개 뒤에 세션의 `user/assistant` 전체 턴을 원순서대로 넣는다.
- 정상 세션 기준 `messages` 순서는 `system -> user -> assistant -> ... -> assistant`를 따른다.

### 2.4 FT 구성 규칙
- `data_type = FT`
- `source = "009.전문분야_기술과학_한국어 멀티세션 데이터"`
- `content = null`
- `system` 고정 문구:  
`너는 기술과학 전문 비서야. 사용자의 질문에 친절하고 과학적으로 대답해.`

### 2.5 이상치 정책
아래 순서로 처리한다:
1. 빈 발화 포함
2. role 교대 위반(`user -> user` 또는 `assistant -> assistant`)
3. 세션 시작 role이 `user`가 아님
4. 세션 마지막이 `user`로 끝나면 마지막 `user` 턴만 제거하고 세션은 유지한다.
5. 마지막 `user` 제거 후 유효한 `assistant` 턴이 1개도 없으면 세션 전체 제외한다.

### 2.6 token_count 정책
- `messages`의 모든 `content`를 순서대로 concat한 단일 텍스트를 공통 토크나이저로 토큰화한다.
- 공통 토크나이저 경로: `data/tokenizers/korean_bbpe_v1/tokenizer.json`

### 2.7 현황 수치 (검증 기준)
- 전체 세션: `135,105`
- 이상 세션: `19` (`~0.014%`)
- 세션 1row 기준 row 산정은 `마지막 user 턴 제거` 정책 반영 후 재프로파일링 필요
- split별 row 수(`train`, `val`)는 세션 단위 재집계로 갱신 필요

### 2.8 통합 저장소 적재 규칙
- 009 전처리 결과는 `data/korean_processed/chunks/` 통합 저장소에 append한다.
- 데이터셋 구분은 경로 분리 대신 `source` 컬럼으로 식별한다.
- 후속 데이터셋(010, 011, 019 등)도 동일 schema를 유지한 채 같은 저장소에 순차 append한다.

---

## 3. 010.전문분야_사회과학_한국어 멀티세션 데이터 전략 (초안)

### 3.1 기본 방향
- 009와 동일한 멀티세션 계열로 간주하고, 1차 전략은 009 규칙 재사용
- 세션 1row + `messages` 다턴 가변 길이 전략을 우선 적용
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 순차 append

### 3.2 작성할 항목 (채움 대상)
1. 입력 경로/파일 포맷 확정 (`json`/`txt` 중 단일 소스 선택)
2. split 매핑 확정 (`Training/Validation`)
3. 발화자 키 매핑(`speaker1/2` 여부) 확인
4. 도메인 system 문구 확정(사회과학 전용)
5. 이상치 유형 및 제외 기준 확정
6. 예상 산출량(세션/row/제외 비율) 기록

### 3.3 승인 조건
- 009 대비 다른 스키마 필드가 있으면 Adapter 규칙만 분기하고 Canonical 스키마는 유지

---

## 4. 011.일상대화 한국어 멀티세션 데이터 전략 (초안)

### 4.1 기본 방향
- 멀티세션 대화형 데이터로 분류
- 009/010과 동일한 세션 직렬화 엔진(세션 1row, 다턴 messages)을 재사용 가능한 기준으로 설계
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 순차 append

### 4.2 작성할 항목 (채움 대상)
1. 입력 경로 확정
2. split 매핑 확정
3. speaker role 매핑 확정
4. 일상대화 도메인 system 문구 확정
5. 빈 발화/role 교대 위반/마지막 user 보정 정책 확정
6. 출력 row 수 및 제외 비율 산출

### 4.3 승인 조건
- 대화형인 경우 `FT` 기본
- PT 혼합이 확인되면 혼합 전략(PT/FT 동시 적재)으로 문서 갱신

---

## 5. 019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터 전략 (초안)

### 5.1 기본 방향
- 텍스트 분석 계열로 분류
- 대화형이 아닐 가능성을 전제로 `PT` 우선 검토, `FT` 혼합 여부는 구조 확인 후 결정
- 결과는 통합 저장소(`data/korean_processed/chunks`)에 순차 append

### 5.2 작성할 항목 (채움 대상)
1. 원본 레코드 단위 정의(문서/문단/질의응답)
2. `data_type` 결정 규칙
- 문서형이면 `PT(content)`
- 질의응답형이면 `FT(messages)`
3. split 경로/메타 매핑 확정
4. 긴 문서 분할 전략(문단/문장 단위) 확정
5. 법률 도메인 품질 규칙(빈 문서, 깨진 인코딩, 극단 길이) 확정

### 5.3 승인 조건
- 019는 009형 고정 규칙을 재사용하지 않고, 구조 확인 후 Adapter 분기 정책을 명시한다.

---

## 6. 나머지 데이터 형식 수정 계획 (확장 백로그)

### 6.1 대상 목록
- `020.주제별 텍스트 일상 대화 데이터`
- `021.용도별 목적대화 데이터`
- `023.국회 회의록 기반 지식검색 데이터`
- `030.웹데이터 기반 한국어 말뭉치 데이터`
- `045.지식검색 대화`
- `046.공감형 대화`
- `141.한국어 멀티세션 대화`
- `전문분야 말뭉치`
- `gsm8k`
- `novel24`

### 6.2 백로그 관리 표준
각 데이터셋은 아래 상태로 관리한다.
- `draft`: 경로/포맷만 파악
- `profiling`: 스키마/분포/이상치 측정 중
- `rule-locked`: PT/FT/품질 규칙 확정
- `ready`: 구현 가능 상태

### 6.3 데이터셋별 기록 템플릿
아래 형식으로 동일 문서에 섹션을 추가한다.

```md
## X. <dataset_name> 전략 (<status>)

### X.1 입력 경로 및 split
- ...

### X.2 row 단위 및 data_type
- ...

### X.3 변환 규칙
- ...

### X.4 이상치/품질 정책
- ...

### X.5 token_count 정책
- ...

### X.6 예상 산출량 및 리스크
- ...
```

### 6.4 우선순위 권장
1. 멀티세션 계열: `010`, `011`, `141`
2. 대화 계열: `020`, `021`, `045`, `046`
3. 문서/검색 계열: `019`, `023`, `030`, `전문분야 말뭉치`
4. 외부 코퍼스 계열: `gsm8k`, `novel24`

---

## 7. 문서 운영 규칙
이 문서는 구현 전 의사결정 문서로 유지하며, 데이터셋별 정책 확정 시 아래 순서로만 갱신한다.

1. 공통 전략 변경 필요 여부 검토
2. 해당 데이터셋 섹션에 규칙 확정값 반영
3. 백로그 상태 업데이트 (`draft -> profiling -> rule-locked -> ready`)
4. 산출량/제외 비율/리스크 수치 반영

이 구조를 유지하면 009 중심 문서에서 전체 데이터셋 로드맵 문서로 자연스럽게 확장할 수 있다.
또한 실제 구현은 데이터셋별 전처리 후 통합 Lance 저장소에 append하는 방식을 기본값으로 유지한다.
