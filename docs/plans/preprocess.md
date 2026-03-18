# Korean Raw 전처리 총계획

## 0. 목적
이 문서는 `data/korean_raw` 원천 데이터를 하나의 통합 전처리 파이프라인으로 정리하기 위한 상위 계획이다.
기준 문서는 [docs/personal/preprocess/preprocess.md](/home/hwajoong/projects/ido/docs/personal/preprocess/preprocess.md)이며, 세부 데이터셋 규칙은 `docs/personal/preprocess/*.md`를 따른다.

핵심 원칙:
1. `docs/personal/preprocess/preprocess.md`를 최상위 기준으로 본다.
2. 하위 데이터셋 문서와 충돌하면 상위 `preprocess.md` 기준으로 해석한다.
3. 모든 데이터셋 결과는 동일한 6개 feature 스키마로 맞춘다.
4. `docs/plans/preprocess.md`는 한국어 데이터 전처리 총계획 문서로서, `docs/personal/preprocess`의 핵심 내용을 빠짐없이 반영한다.
5. 이후 `docs/pseudo/preprocess`와 `docs/src/preprocess`는 이 문서의 결정사항을 그대로 따른다.

---

## 1. 공통 데이터 계약

### 1.1 특수 토큰
- `<|bos|>`: begin of sequence
- `<|eos|>`: end of sequence
- `<|system|>`, `<|user|>`, `<|assistant|>`: 텍스트 발화 주체
- `<|eot_id|>`: end of turn
- `<|pad|>`: 배치 패딩 토큰

학습 시 주의:
- `<|system|>`, `<|user|>`, `<|assistant|>` 자체는 예측 대상이 아니다.
- SFT/REASONING 학습에서는 assistant 응답만 역전파 대상으로 사용한다.
- 전처리 단계에서는 row 스키마와 텍스트 구성만 고정하고, 실제 마스킹은 학습 로더에서 처리한다.

### 1.2 Canonical Row Schema
최종 row는 아래 6개 feature만 사용한다.

1. `source: String`
2. `data_usage: String`
3. `split: String`
4. `content: String | null`
5. `messages: List[Struct{role: String, content: String}] | null`
6. `token_count: int32`

강제 규칙:
- `content`와 `messages`는 둘 다 nullable이지만 최소 하나는 반드시 채운다.
- `data_usage`는 `PT`, `SFT`, `REASONING`만 허용한다.
- PT 학습은 `content`를 사용한다.
- SFT 및 REASONING 학습은 `messages`를 사용한다.
- 특정 학습 모드에서 필요한 필드가 `null`인 row는 해당 모드에서만 제외한다.
- 일부 하위 personal 문서에 나오는 `data_type` 개념은 최종 통합 스키마에 포함하지 않는다.

### 1.3 content / messages / token_count 규칙
- `content`는 PT 입력으로 쓰는 문자열이다.
- `messages`는 SFT 또는 REASONING 입력으로 쓰는 대화 배열이다.
- `token_count`는 항상 `content` 필드 기준 토큰 길이로 계산한다.
- `messages`가 존재하더라도 `token_count`는 직렬화된 `messages` 길이가 아니라 `content` 길이로 저장한다.
- 중간 parquet 단계에서는 `token_count`가 비어 있을 수 있으나, 최종 Lance 적재 전에는 모든 row를 채운다.

### 1.4 split 규칙
최종 split은 모든 데이터셋에 대해 `train:val = 99:1`로 재구성한다.

강제 규칙:
- `data/korean_raw` 안에 이미 train/val 또는 train/test가 나뉘어 있어도 최종 split은 다시 99:1로 재구성한다.
- 별도 split 디렉터리가 없는 데이터셋도 같은 기준으로 99:1 재구성한다.
- val 1%는 각 데이터셋 항목에서 랜덤으로 추출한다.
- 데이터셋에 따라 row 단위가 문서, 세션, JSON row, txt 파일, txt 청크 중 무엇인지 먼저 확정한 뒤 그 단위에서 split을 재구성한다.
- 한 원천 항목 내부를 잘라 train/val에 섞지 말아야 한다는 하위 규칙이 있으면 그 규칙은 유지하되, 최종 비율은 99:1로 맞춘다.

### 1.5 통계 및 메타데이터
전처리 중 아래 메타데이터를 반드시 남긴다.

1. 중간 parquet 메타데이터
- 원천 파일 수
- 원천 row 수
- 전처리 후 row 수
- drop 사유별 건수
- source별 row 수
- split별 row 수

2. 토크나이저 메타데이터
- 사용한 학습 데이터 범위
- vocab size
- special token 목록
- dedup 후 학습 데이터 row 수
- exact dedup 제거 row 수
- minhash + LSH 제거 row 수
- minhash + LSH 중복 클러스터 수

3. Lance 메타데이터
- chunk 또는 shard 수
- shard별 byte 크기
- shard별 row 수
- source별 row 수
- source별 총 token 수
- data_usage별 row 수
- split별 row 수
- `content` 보유 row 수
- `messages` 보유 row 수

4. 인덱스 및 통계
- Lance dataset append 이후 indices를 생성한다.
- source별 token 통계를 남긴다.
- `namu`는 `source`가 개별 문서 단위로 달라도 통계는 하나의 `namu` 소스로 묶어서 집계한다.
- dedup 전후 row 수를 source별, data_usage별로 남긴다.
- 대표 row 선별 규칙에 따라 어떤 row가 남았는지 사유를 통계로 남긴다.

---

## 2. 공통 파이프라인

### 2.1 전체 흐름
1. `data/korean_raw`의 각 데이터셋을 읽는다.
2. 데이터셋별 규칙에 따라 통일된 6-feature 스키마의 중간 parquet row로 변환한다.
3. `content` 기준 exact dedup으로 완전히 동일한 row를 제거한다.
4. 전체 데이터 소스를 섞은 뒤, `content` 기준 datatrove의 kiwi 기반 한국어 형태소 n-gram shingle로 minhash + LSH 후보를 만들고 Jaccard similarity 임계값 기반으로 near-duplicate를 제거한다.
5. dedup 결과를 사용해 토크나이저를 학습한다.
6. 학습된 토크나이저로 모든 row의 `token_count`를 `content` 기준으로 채운다.
7. 데이터셋별 Lance shard를 만들고 하나의 Lance 데이터셋에 append한다.
8. append 완료 후 metadata, manifests, indices, token 통계를 생성한다.

### 2.2 중간 parquet 단계
- 모든 데이터셋은 먼저 통일 스키마의 parquet로 정규화한다.
- parquet 단계에서 데이터셋별 규칙 차이는 `source`, `data_usage`, `content`, `messages` 구성 규칙으로만 표현한다.
- 이후 dedup, tokenizer, token_count, Lance 적재는 parquet를 기준으로 진행한다.

### 2.3 dedup 단계
- exact dedup은 `content`가 완전히 동일한 row를 제거한다.
- exact dedup과 minhash + LSH dedup은 데이터 소스를 섞은 전체 row 집합 기준으로 수행한다.
- minhash + LSH dedup은 `content` 문자열을 datatrove의 kiwi 기반 한국어 형태소 5-gram shingle 집합으로 바꾼 뒤 수행한다.
- minhash + LSH는 datatrove 구현 흐름에 따라 near-duplicate 후보를 찾는 단계로만 사용하고, 후보 row들에 대해서는 shingle 집합 기준 Jaccard similarity를 다시 계산한다.
- Jaccard similarity 임계값은 코드 맨 앞에서 수정하기 쉬운 하이퍼파라미터로 둔다.
- Jaccard similarity가 임계값을 넘는 row들은 같은 중복 클러스터로 판단하고 대표 row 하나만 남긴다.
- 대표 row 우선순위는 `REASONING > SFT > PT`로 둔다.
- `data_usage`가 같으면 `content`의 Python 기준 `len`이 더 긴 row를 대표로 남긴다.
- 그마저도 같으면 안정적인 row locator 순서가 앞선 row를 대표로 남긴다.
- dedup 이후 남은 row만 토크나이저 학습과 최종 Lance 적재에 사용한다.

### 2.4 토크나이저 빌드
- dedup 이후 데이터를 사용해 토크나이저를 새로 학습한다.
- 기본 vocab size는 `48k`로 둔다.
- 토크나이저는 이후 `token_count` 계산과 학습 입력 구성의 기준이 된다.

### 2.5 token_count 채우기
- 토크나이저 빌드 후, `content`가 있는 모든 row의 `token_count`를 채운다.
- `token_count`는 항상 `content` 기준 길이이다.
- 최종 Lance 적재 시 `token_count`는 null을 허용하지 않는다.

### 2.6 Lance 적재 및 샤딩
- 각 데이터셋은 Lance shard로 구성한 뒤 append한다.
- 샤드 하나의 목표 크기는 `1GB`다.
- 행 중간 분할은 허용하지 않는다.
- 모든 데이터셋은 같은 스키마로 append 가능해야 한다.

### 2.7 manifests / indices
- 중간 parquet 생성 시점의 manifest를 남긴다.
- dedup 결과 manifest를 남긴다.
- dedup manifest에는 exact 제거 수, minhash 후보 수, 최종 중복 클러스터 수, 대표 row 선별 통계를 포함한다.
- tokenizer 생성 정보를 manifest로 남긴다.
- Lance append 완료 후 최종 manifest와 indices를 생성한다.

---

## 3. 데이터셋별 전처리 규칙

### 3.1 멀티세션 공통 계열
대상:
- `009.전문분야_기술과학_한국어 멀티세션 데이터`
- `010.전문분야_사회과학_한국어 멀티세션 데이터`
- `011.일상대화 한국어 멀티세션 데이터`
- `141.한국어 멀티세션 대화`

공통 규칙:
- `sessionInfo[*]`의 각 session이 row 단위다.
- `dialog[*].utterance`를 좌우 strip하여 사용한다.
- 빈 문자열 utterance가 하나라도 있으면 해당 session은 제외한다.
- 시작 role이 user가 아니면 제외한다.
- role 교대가 깨지면 제외한다.
- 마지막 turn이 user면 마지막 user turn만 제거한다.
- 마지막 user 제거 후 assistant turn이 하나도 없으면 제외한다.
- `content`는 system을 제외한 user/assistant utterance를 원순서대로 공백 하나로 이어 붙인다.
- `messages`는 `[system, user, assistant, ...]` 구조로 둔다.
- `data_usage`는 `SFT`를 사용한다.
- 최종 split은 session row 기준 99:1 랜덤 재구성이다.

#### 009.전문분야_기술과학_한국어 멀티세션 데이터
- `source = "009.전문분야_기술과학_한국어 멀티세션 데이터"`
- `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

#### 010.전문분야_사회과학_한국어 멀티세션 데이터
- `source = "010.전문분야_사회과학_한국어 멀티세션 데이터"`
- `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 사회과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다."`

#### 011.일상대화 한국어 멀티세션 데이터
- `source = "011.일상대화 한국어 멀티세션 데이터"`
- `speaker1 -> user`, `speaker2 -> assistant`
- `system = "당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."`

#### 141.한국어 멀티세션 대화
- `source = "141.한국어 멀티세션 대화"`
- `speaker1 -> user`, `speaker2 -> assistant`
- base system 문장은 `"당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."`
- FT용 `messages[0].content`에는 `personaInfo.clInfo.personaFeatures`를 원본 순서대로 `\n[clInfo persona]\n` 블록에 붙인다.
- `content`는 system 없이 user/assistant utterance만 이어 붙인다.
- `topicInfo`, `summary`, `date/time`은 사용하지 않는다.

### 3.2 019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터
- 원본 JSON 한 건이 row 단위다.
- `data_usage = "PT"`
- `messages = null`
- 문자열은 모두 strip 후 사용한다.
- `content`는 아래 필드의 유효 문자열을 원순서대로 줄바꿈 하나로 이어 붙여 만든다.
  1. `mentionedItems.rqestObjet`
  2. `disposal.disposalcontent`
  3. `assrs.dedatAssrs`
  4. `facts.bsisFacts`
  5. `dcss.courtDcss`
  6. `close.cnclsns`
  7. `clauseArticle`
  8. `comProvision`
- 각 필드는 배열일 때만 사용한다.
- 배열 원소가 문자열일 때만 strip 후 사용한다.
- 최종 유효 문자열이 하나도 없으면 제외한다.
- row가 생성된 경우 `skip_field`, `skip_item`은 fixup으로 기록한다.
- `info`, `concerned`, `org`, `relateLaword`, `qotatPrcdnt`, `acusrAssrs`는 사용하지 않는다.
- 최종 split은 JSON row 기준 99:1 랜덤 재구성이다.

### 3.3 020.주제별 텍스트 일상 대화 데이터
- 원본 JSON의 `info[*]`가 row 단위다.
- `data_usage = "SFT"`
- `annotations.lines[*]`를 turn으로 사용한다.
- 문자열은 모두 strip 후 사용한다.
- `norm_text`가 문자열이면 우선 사용한다.
- `norm_text`가 없으면 `text`를 사용하고 `"1 : "` 같은 화자 접두어를 제거한다.
- turn text가 비면 해당 turn은 제외한다.
- `annotations.speaker_type == "1:1"`인 레코드만 사용한다.
- turn 순서대로 처음 등장한 `speaker.id`를 user, 다음 등장한 다른 `speaker.id`를 assistant로 매핑한다.
- 세 번째 화자가 나오면 해당 레코드는 제외한다.
- 연속 같은 화자 turn은 공백 하나로 병합한다.
- 마지막 turn이 user면 마지막 user turn을 제거한다.
- 최종 assistant turn이 하나도 없으면 제외한다.
- `system = "당신은 일상 대화 상대입니다. 사용자의 말에 자연스럽고 친근하게 반응하세요."`
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text를 공백 하나로 이어 붙인다.
- `annotations.text`, `speechAct`, `morpheme`, 화자 성별/나이 등 메타정보는 사용하지 않는다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.4 021.용도별 목적대화 데이터
- 원본 JSON의 `info[*]`가 row 단위다.
- row에는 `content`와 `messages`를 모두 채운다.
- `data_usage = "SFT"`
- PT용 `content`는 `annotations.text`를 사용한다.
- `annotations.text`가 없거나 비면 해당 레코드는 제외한다.
- FT용 turn은 `annotations.lines[*]`를 사용한다.
- `speaker.id == "B"`를 user, `speaker.id == "A"`를 assistant로 둔다.
- 첫 번째 `A` 발화 하나를 제거한 뒤 남은 대화를 FT에 사용한다.
- 제거 후 첫 turn이 `B(user)`가 아니면 해당 레코드는 제외한다.
- `norm_text` 우선, 없으면 `text`를 사용한다.
- `text`를 사용할 때 `"A."`, `"B."`, `"A :"`, `"B : "` 같은 접두어를 제거한다.
- turn text가 비면 해당 turn은 제외한다.
- 연속 같은 화자 turn은 공백 하나로 병합한다.
- 마지막 turn이 user면 마지막 user turn 하나를 제거한다.
- assistant turn이 하나도 없으면 제외한다.
- `system = "당신은 콜센터 상담원입니다. 사용자의 문의에 정확하고 친절하게 답변하세요."`
- `messages = [system, user, assistant, ...]`
- `speechAct`, `morpheme`, 화자 성별/나이, `annotations.subject`는 사용하지 않는다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.5 023.국회 회의록 기반 지식검색 데이터
- 원본 JSON 한 건이 row 단위다.
- `data_usage = "SFT"`
- `content = context`
- FT용 `messages`는 `[system, user(question.comment), assistant(answer.comment)]` 구조다.
- `context`, `question.comment`, `answer.comment`는 strip 후 사용한다.
- `system = "당신은 국회회의록 기반으로 대답을 하는 위원이야. 질문에 대해서 사실 근거에 기반해서 대답해줘."`
- `context_learn`, `context_summary(summary_q, summary_a)`는 사용하지 않는다.
- 최종 split은 JSON row 기준 99:1 랜덤 재구성이다.

### 3.6 030.웹데이터 기반 한국어 말뭉치 데이터
- `named_entity` 안의 요소 하나가 row 단위다.
- `data_usage = "PT"`
- `messages = null`
- `named_entity.title.sentence`를 제목으로 사용한다.
- `named_entity.content.sentence`들을 strip 후 공백 하나로 이어 본문을 만든다.
- `content = title + "\n" + body`
- 제목이 비면 제외한다.
- 본문 문장이 모두 비면 제외한다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.7 045.지식검색 대화
- 원본 JSON 한 건이 row 단위다.
- `data_usage = "SFT"`
- `utterances`에서 `질문자 -> user`, `전문가 -> assistant`
- 문자열은 모두 strip 후 사용한다.
- assistant는 항상 전문가의 `text`를 사용한다.
- `reference_text[*].value`가 있으면 user 문장은 `질문자 text + "\n근거: " + reference_text concat`으로 만든다.
- 유효한 `reference_text`가 없으면 질문자 `text`만 user로 사용한다.
- `system = "user의 질문에 대해서 텍스트 근거 기반으로 대답을 하는 전문가야"`
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text를 공백 하나로 이어 붙인다.
- `search_URL`, `search_query`, `reference_date`, `info.evaluation`, `votes`는 사용하지 않는다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.8 046.공감형 대화
- 원본 JSON 한 건이 row 단위다.
- `data_usage = "SFT"`
- `utterances`에서 `speaker -> user`, `listener -> assistant`
- 문자열은 모두 strip 후 사용한다.
- `utterances`가 배열이 아니면 제외한다.
- role이 `speaker` 또는 `listener`가 아니면 해당 turn은 제외한다.
- text가 비면 해당 turn은 제외한다.
- 정규화 후 turn이 비어 있으면 제외한다.
- 시작 role이 user가 아니면 제외한다.
- role 교대가 깨지면 제외한다.
- 마지막 turn이 user면 마지막 user turn만 제거한다.
- 마지막 user 제거 후 assistant turn이 하나도 없으면 제외한다.
- `system = info.situation + "에 대해서 " + info.listener_behavior join(", ") + "에 맞추어서 대답을 해주는 친구가 되어줘"`
- `messages = [system, user, assistant, ...]`
- `content`는 system 제외 user/assistant text를 공백 하나로 이어 붙인다.
- `info.evaluation`, `listener_empathy`, `speaker_changeEmotion`, `votes`는 사용하지 않는다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.9 gsm8k
- 원본 row 단위는 parquet row 1개다.
- `data_usage = "SFT"`
- `question`은 user, `answer`는 assistant로 사용한다.
- `content = question + "\n" + answer`
- `messages = [system, user(question), assistant(answer)]`
- `question`, `answer`는 strip 후 사용한다.
- 둘 중 하나라도 비면 제외한다.
- `system = "당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."`
- `question_en`, `answer_en`는 사용하지 않는다.
- 원본 train/test 분할은 유지하지 않고 전체 row를 모아 최종 split을 99:1로 재구성한다.

### 3.10 novel24
- 입력은 txt 파일 묶음이다.
- 숨김 파일과 `.DS_Store`는 제외한다.
- 하나의 txt 파일 내부를 train/val로 나누지 않는다.
- txt 파일 묶음을 기준으로 99:1 랜덤 분할한다.
- row는 `\n` 단위 누적 방식으로 만든다.
- 줄 중간이나 문장 중간은 자르지 않는다.
- `data_usage = "PT"`
- `messages = null`
- `content`에는 현재 청크 문자열을 넣는다.

### 3.11 국립국어원 구어 말뭉치
- 원본 JSON의 `document[*]` 각 요소가 row 단위다.
- `data_usage = "PT"`
- `messages = null`
- `document.utterance[*].form`을 strip 후 원순서대로 수집한다.
- 유효한 발화들을 줄바꿈 하나로 이어 `content`를 만든다.
- `document`가 배열이 아니면 해당 JSON은 제외한다.
- `document.utterance`가 배열이 아니면 해당 row는 제외한다.
- 유효한 `utterance.form`이 하나도 없으면 제외한다.
- `document.metadata.title`, `author`, `publisher`, `date`, `topic`, `speaker`, 파일 상위 `metadata`, `utterance.original_form`, `speaker_id`, `note`는 사용하지 않는다.
- 최종 split은 `document` row 기준 99:1 랜덤 재구성이다.

### 3.12 국립국어원 문어 말뭉치
- 원본 JSON의 `document[*]` 각 요소가 row 단위다.
- `data_usage = "PT"`
- `messages = null`
- 제목은 `document.metadata.title`을 사용한다.
- 본문은 `document.paragraph[*].form`의 유효 문자열을 줄바꿈 하나로 이어 만든다.
- `content = "제목: " + title + "\n내용: " + body`
- 제목이 비면 제외한다.
- `document.paragraph`가 배열이 아니면 제외한다.
- 유효 문단이 하나도 없으면 제외한다.
- `document.metadata.author`, `publisher`, `date`, 파일 상위 `metadata`, `paragraph.id`는 사용하지 않는다.
- 최종 split은 `document` row 기준 99:1 랜덤 재구성이다.

### 3.13 국립국어원 신문 말뭉치 2020
- 원본 JSON의 `document[*]` 각 요소가 row 단위다.
- `data_usage = "PT"`
- `messages = null`
- 유효한 `document.paragraph[*].form` 중 첫 번째를 제목으로 사용한다.
- 나머지 유효 문단을 줄바꿈 하나로 이어 본문을 만든다.
- `content = "제목: " + title + "\n내용: " + body`
- `document.paragraph`가 배열이 아니면 제외한다.
- 유효 문단이 하나도 없으면 제외한다.
- 제목만 있고 본문이 없으면 제외한다.
- `document.metadata.author`, `publisher`, `date`, `topic`, `original_topic`, 파일 상위 `metadata`는 사용하지 않는다.
- 최종 split은 `document` row 기준 99:1 랜덤 재구성이다.

### 3.14 HAERAE-HUB-KOREAN-WEBTEXT
- `text` 필드만 사용한다.
- `data_usage = "PT"`
- `messages = null`
- `content = text`
- 원본의 다른 필드는 사용하지 않는다.
- 원본 데이터에 있는 `token_count` 필드는 최종 전처리 값으로 쓰지 않는다.
- 해당 원본 `token_count`는 이후 별도 토크나이저 성능 비교 참고값으로만 본다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.15 HAERAE-HUB-HR-Instruct-Math-v0.1
- `instruction`을 user, `response`를 assistant로 사용한다.
- `data_usage = "REASONING"`
- `system = "사용자의 질문을 읽고 단계 별로 사고하여 논리적인 답변을 제시합니다."`
- `messages = [system, user(instruction), assistant(response)]`
- `content`는 user/assistant 내용을 이어 붙인 reasoning 학습용 텍스트로 구성한다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.16 nohurry-Opus-4.6-Reasoning-3000x-filtered
- 원본 jsonl의 각 row가 전처리 row 단위다.
- `problem`을 user 질의로 사용한다.
- `thinking + solution`을 assistant 응답으로 사용한다.
- `data_usage = "REASONING"`
- `system = "사용자의 질문을 읽고 단계 별로 사고하여 논리적인 답변을 제시합니다."`
- `messages = [system, user(problem), assistant(thinking + solution)]`
- `content`는 `problem + " " + thinking + " " + solution` 형태의 reasoning 학습용 텍스트로 구성한다.
- 최종 split은 row 기준 99:1 랜덤 재구성이다.

### 3.17 namu
- 기존 JSON row가 이미 목표 스키마와 유사하다.
- `data_usage = "PT"`
- `messages = null`
- `content`는 원본 텍스트를 사용한다.
- 원본 row는 모두 train로 라벨링되어 있어도 최종 split은 99:1로 재구성한다.
- source가 `"namu.문서명.섹션명"`처럼 세분화되어 있어도 통계 산출 시에는 하나의 `namu` 소스로 묶는다.
- 중복이 매우 많을 것으로 예상되므로 dedup 단계의 영향을 별도 통계로 남긴다.

---

## 4. 검증 기준

### 4.1 스키마 검증
- 모든 row가 6개 feature 스키마를 만족해야 한다.
- `content`와 `messages` 중 최소 하나는 반드시 존재해야 한다.
- `data_usage`는 `PT`, `SFT`, `REASONING` 중 하나여야 한다.
- 최종 Lance 적재 전 `token_count`는 null이 아니어야 한다.

### 4.2 데이터 규칙 검증
- 각 데이터셋 규칙이 `docs/personal/preprocess/*.md`와 충돌하지 않아야 한다.
- split은 최종적으로 모든 데이터셋에서 99:1 규칙을 만족해야 한다.
- `020`, `021`, `030`과 같이 과거 계획과 현재 personal 규칙이 달라진 항목은 최신 규칙으로 덮어써야 한다.
- `국립국어원 구어/문어/신문`, `045`, `046`, `141`, `gsm8k`, `novel24`는 기존 plan의 구 split 규칙을 더 이상 사용하지 않아야 한다.

### 4.3 통계 검증
- source별 row 수와 token 수를 남긴다.
- data_usage별 row 수를 남긴다.
- split별 row 수를 남긴다.
- drop 사유별 집계를 남긴다.
- `namu`는 source 통계를 하나로 묶어 집계한다.
- exact dedup 전후 row 수와 minhash + LSH dedup 전후 row 수를 남긴다.
- 대표 row 선별 시 `data_usage` 우선순위로 남은 경우와 `len(content)` 비교로 남은 경우를 구분해 집계한다.

---

## 5. 후속 문서 연결
- `docs/pseudo/preprocess`는 이 문서를 바탕으로 실제 구현 순서와 함수 수준 의사코드를 한국어로 상세화한다.
- `docs/src/preprocess`는 `pseudo`를 바탕으로 실제 코드 설계 및 모듈 단위 작업으로 내린다.
- 이후 문서들은 이 문서에서 이미 결정한 스키마, split, dedup, tokenizer, token_count, metadata 정책을 다시 바꾸지 않는다.
