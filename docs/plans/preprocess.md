# Korean Raw 전처리 마스터 플랜

## 0. 목적
이 문서는 최신 [docs/personal/preprocess](/home/hwajoong/projects/ido/docs/personal/preprocess) 문서를 기준으로
`data/korean_raw` 원천 데이터를 하나의 Lance 데이터셋으로 정리하는 상위 계획을 고정한다.

핵심 원칙:
1. `docs/personal/preprocess/*.md`를 최우선 기준으로 따른다.
2. 데이터셋별 전처리는 독립 실행 가능해야 한다.
3. 모든 데이터셋 결과는 같은 5개 feature 스키마로 append 가능해야 한다.
4. 데이터셋별 차이는 `source`, `content`, `messages` 생성 규칙으로만 구분한다.

---

## 1. 공통 계약

### 1.1 Canonical Row Schema
최종 row는 아래 5개 feature만 사용한다.

1. `source: String`
2. `split: String`
3. `content: String | null`
4. `messages: List[Struct{role: String, content: String}] | null`
5. `token_count: int32`

강제 규칙:
- `content`와 `messages`는 둘 다 nullable이지만 최소 하나는 반드시 채운다.
- PT 학습은 `content`를 사용한다.
- FT 학습은 `messages`를 사용한다.
- 특정 학습 모드에서 필요한 필드가 `null`인 row는 그 모드에서만 제외한다.
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
- 데이터셋 구분은 경로가 아니라 `source` 값으로 한다.
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
`token_count`는 저장 row 자체가 아니라 실제 학습 입력 문자열의 토큰 길이로 계산한다.

공통 규칙:
- `messages`가 있으면 personal 예시대로 `<|system|>`, `<|user|>`, `<|assistant|>`를 붙여 직렬화한다.
- `messages`가 없고 `content`만 있으면 `content` 문자열 자체를 사용한다.
- `<|bos|>`, `<|eos|>`, `<|eot_id|>`는 현재 personal 예시에 없으므로 기본 전처리 직렬화에는 넣지 않는다.
- 학습 로더도 같은 직렬화 규칙을 따라야 하며, 모드별 추가 마스킹은 로더에서 처리한다.

### 1.6 공통 검증 항목
1. 스키마 검증
- feature 5개와 타입 일치
- `split` 허용값 검증
- `content/messages` 최소 1개 존재 규칙 검증

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
- `content` 보유 rows / `messages` 보유 rows

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
- `141.한국어 멀티세션 대화`

### 2.1 입력 경로 패턴
- `009/010/011`: `.../Training/02.라벨링데이터/*.json`, `.../Validation/02.라벨링데이터/*.json`
- `141`: `.../Training/02.라벨링데이터/**/*.json`, `.../Validation/02.라벨링데이터/**/*.json`

### 2.2 split 매핑
- `Training -> train`
- `Validation -> val`

### 2.3 세션 정규화 규칙
- 세션 배열은 `sessionInfo`
- 발화 배열은 `sessionInfo[*].dialog`
- 텍스트 키는 `dialog[*].utterance`
- `utterance`는 좌우 공백을 제거한 뒤 사용한다.
- 빈 문자열 발화가 하나라도 있으면 세션 전체 제외
- 시작 role이 `user`가 아니면 세션 전체 제외
- role 교대가 깨지면 세션 전체 제외
- 마지막이 `user`면 마지막 `user` 턴만 제거하고 세션은 유지
- 마지막 `user` 제거 후 assistant 턴이 1개도 없으면 세션 전체 제외

### 2.4 row 생성 규칙
- 정규화에 성공한 세션마다 row를 1개 생성한다.
- `source`는 데이터셋명으로 채운다.
- `content`는 system을 제외한 user/assistant `content`를 원순서대로 공백 하나로 이어 붙인 문자열이다.
- `messages = [system, user, assistant, ...]`
- `token_count`는 `messages` 직렬화 기준으로 계산한다.

---

## 3. 009.전문분야_기술과학_한국어 멀티세션 데이터
- `source = "009.전문분야_기술과학_한국어 멀티세션 데이터"`
- role 매핑: `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

## 4. 010.전문분야_사회과학_한국어 멀티세션 데이터
- `source = "010.전문분야_사회과학_한국어 멀티세션 데이터"`
- role 매핑: `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 사회과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

## 5. 011.일상대화 한국어 멀티세션 데이터
- `source = "011.일상대화 한국어 멀티세션 데이터"`
- role 매핑: `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."`

## 6. 019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터
- 원본 JSON 한 건이 row 단위다.
- `content`는 아래 필드의 문자열을 지정 순서대로 줄바꿈 `\n` 하나로 이어 붙여 생성한다.
  1. `mentionedItems.rqestObjet`
  2. `disposal.disposalcontent`
  3. `assrs.dedatAssrs`
  4. `facts.bsisFacts`
  5. `dcss.courtDcss`
  6. `close.cnclsns`
  7. `clauseArticle`
  8. `comProvision`
- 각 필드는 배열일 때만 사용하고, 문자열 원소에만 strip을 적용한다.
- 8개 후보 필드 중 하나라도 유효하면 row를 유지한다.
- 8개 후보 필드 중 일부가 비어 있거나 배열이 아니면 그 필드만 제외하고 나머지 필드로 계속 `content`를 만든다.
- row가 생성된 경우 `skip_field`, `skip_item`은 `fixup`으로 기록한다.
- 최종적으로 유효 문자열이 하나도 남지 않았을 때만 해당 레코드를 제외한다.
- `messages = null`
- `info`, `concerned`, `org`, `relateLaword`, `qotatPrcdnt`, `acusrAssrs`는 기본 학습 입력에서 사용하지 않는다.

## 7. 020.주제별 텍스트 일상 대화 데이터
- 원본 JSON의 `info[*]`가 row 단위다.
- `annotations.speaker_type == "1:1"`이면 FT용 `messages`를 만든다. `다자간 대화`와 third speaker 케이스는 PT-only row로 salvage한다.
- turn text는 `norm_text` 우선, 없으면 `text`를 사용하고 `"1 : "` 같은 화자 접두어는 제거한다.
- turn 순서대로 처음 등장한 `speaker.id`를 user, 다음 등장한 다른 `speaker.id`를 assistant로 매핑한다.
- `speaker_type=다자간 대화`이면 FT용 `messages`는 만들지 않고, 유효한 turn text를 이어붙인 `content`와 `messages = null` PT-only row로 salvage한다.
- 세 번째 화자가 나오면 FT용 `messages`는 만들지 않고, 유효한 turn text를 이어붙인 `content`와 `messages = null` PT-only row로 salvage한다.
- 연속 같은 화자 turn은 공백 하나로 병합한다.
- 마지막 turn이 user면 마지막 user turn 하나를 제거한다.
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text 공백 결합이다.
- `system = "당신은 일상 대화 상대입니다. 사용자의 말에 자연스럽고 친근하게 반응하세요."`
- `annotations.text`, `speechAct`, `morpheme`, 화자 성별/나이 등 메타정보는 사용하지 않는다.

## 8. 021.용도별 목적대화 데이터
- 원본 JSON의 `info[*]`가 row 단위다.
- `annotations.lines[*]`를 turn으로 사용한다.
- role 매핑은 `speaker.id == "B"`를 user, `speaker.id == "A"`를 assistant로 둔다.
- 첫 번째 `A` 발화 1개를 제거한 뒤 남은 대화를 FT용 `messages`로 구성한다.
- 제거 후 첫 turn은 반드시 `B(user)`여야 한다.
- 제거 후 `bad_start_role`가 발생하면 FT용 `messages`는 만들지 않고, PT용 `content`만 유지한 `messages = null` row로 저장한다.
- `norm_text` 우선, 없으면 `text`를 사용하고 `"A."`, `"B."`, `"A :"`, `"B : "` 같은 접두어를 제거한다.
- 연속 같은 화자 turn은 공백 하나로 병합한다.
- 마지막 turn이 user면 마지막 user turn 하나를 제거한다.
- `messages = [system, user, assistant, ...]`
- `content`는 system을 제외한 user/assistant text를 원순서대로 공백 하나로 이어 붙인 문자열이다.
- `annotations.text`, `speechAct`, `morpheme`, 화자 성별/나이, `annotations.subject`는 기본 학습 입력에서 사용하지 않는다.
- `system = "당신은 콜센터 상담원입니다. 사용자의 문의에 정확하고 친절하게 답변하세요."`

## 9. 023.국회 회의록 기반 지식검색 데이터
- 원본 JSON 한 건이 row 단위다.
- `content = context`
- `messages = [system, user(question.comment), assistant(answer.comment)]`
- `context`, `question.comment`, `answer.comment`에 strip을 적용한다.
- `context_learn`, `context_summary(summary_q, summary_a)`는 기본 규칙에서는 사용하지 않는다.
- `system = "당신은 국회회의록 기반으로 대답을 하는 위원이야. 질문에 대해서 사실 근거에 기반해서 대답해줘."`

## 10. 030.웹데이터 기반 한국어 말뭉치 데이터
- `named_entity[*]` 아래 `title[*]` 항목 하나를 row 단위로 본다.
- 본문은 `content[*].sentence`를 strip 후 공백 하나로 이어 붙여 만든다.
- 제목은 해당 `title[*].sentence`를 사용한다.
- `messages = [system, user(본문), assistant(제목)]`
- `content = 제목 + "\n" + 본문`
- `named_entity` 누락 또는 비배열이면 파일 스킵
- 본문 문장이 모두 비면 해당 제목 row 제외
- 제목이 비면 해당 제목 row 제외
- `system = "당신은 기사를 읽고 제목을 짓습니다. 내용을 요약하고, 사람들의 눈길을 끄는 제목을 작성합니다."`

## 11. 045.지식검색 대화
- 원본 JSON 한 건이 row 단위다.
- role 매핑: `질문자 -> user`, `전문가 -> assistant`
- assistant는 항상 전문가의 `text`를 사용한다.
- `reference_text[*].value`가 있으면 user 문장은 `질문자 text + "\n근거: " + 근거 concat`으로 만든다.
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text 공백 결합이다.
- `search_URL`, `search_query`, `reference_date`, `info.evaluation`, `votes`는 사용하지 않는다.
- `system = "user의 질문에 대해서 텍스트 근거 기반으로 대답을 하는 전문가야"`

## 12. 046.공감형 대화
- 원본 JSON 한 건이 row 단위다.
- role 매핑: `speaker -> user`, `listener -> assistant`
- 정규화 규칙은 빈 turn 제외, 시작 role 검사, role 교대 검사, 마지막 user 제거 규칙을 따른다.
- system 문자열은 `info.situation + "에 대해서 " + listener_behavior join + "에 맞추어서 대답을 해주는 친구가 되어줘"`로 만든다.
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text 공백 결합이다.
- `info.evaluation`, `listener_empathy`, `speaker_changeEmotion`, `votes`는 사용하지 않는다.

## 13. 141.한국어 멀티세션 대화
- role 매핑: `speaker1 -> user`, `speaker2 -> assistant`
- 세션 정규화는 2장의 멀티세션 공통 규칙을 따른다.
- base system 문장은 `"당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."`
- `personaInfo.clInfo.personaFeatures`가 있으면 원본 순서를 유지해 `\n[clInfo persona]\n` 블록으로 system 뒤에 붙인다.
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text 공백 결합이다.
- `topicInfo`, `summary`, `date/time`은 사용하지 않는다.

## 14. gsm8k
- 입력은 `data/korean_raw/gsm8k/train-00000-of-00001.parquet`, `data/korean_raw/gsm8k/test-00000-of-00001.parquet`다.
- split 매핑은 `train parquet -> train`, `test parquet -> val`로 둔다.
- 원본 row 단위는 parquet row 1개다.
- `question`은 문제, `answer`는 풀이와 정답으로 사용한다.
- `messages = [system, user(question), assistant(answer)]`
- `content = question + "\n" + answer`
- `question`, `answer`에는 strip을 적용한다.
- strip 후 `question` 또는 `answer`가 비면 해당 row는 제외한다.
- `question_en`, `answer_en`는 기본 학습 입력에서 사용하지 않는다.
- `system = "당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."`

## 15. novel24
- 입력은 `data/korean_raw/novel24/*.txt`다.
- 숨김 파일과 `.DS_Store`는 제외한다.
- txt 파일 묶음을 안정적인 순서로 정렬한 뒤 앞 90%를 train, 뒤 10%를 val로 나눈다.
- 한 txt 파일 내부를 다시 90:10으로 나누지 않는다.
- row는 `\n` 단위 누적 방식으로 만든다.
- 후보 청크의 `token_count`가 `1024`를 넘기 직전까지만 현재 row에 넣고, 넘기는 줄부터 다음 row로 넘긴다.
- 문장 중간이나 줄 중간은 자르지 않는다.
- `content`에는 현재 청크 문자열을 넣고 `messages = null`로 둔다.

---

## 16. 구현 순서
1. 공통 스키마, 직렬화, `token_count`, Lance append 절차를 구현한다.
2. `009/010/011/141` 멀티세션 계열 adapter를 구현한다.
3. `019/020/021/023/030/045/046/gsm8k` adapter를 dataset별 규칙에 맞게 구현한다.
4. `novel24` txt 분할 adapter를 구현한다.
5. dataset별 결과를 같은 Lance 저장소에 append한다.

## 17. 수용 기준
1. 모든 dataset 계획이 최신 personal 문서와 논리적으로 충돌하지 않는다.
2. 각 데이터셋 규칙만 보고 입력 경로, row 단위, `content/messages`, `token_count`, 품질 규칙을 바로 구현할 수 있다.
3. 모든 데이터셋 결과가 같은 5-feature 스키마로 append 가능하다.
4. PT는 `content`, FT는 `messages`를 사용한다는 규칙이 plans와 pseudo 전반에서 일관된다.
