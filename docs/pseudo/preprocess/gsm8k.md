# gsm8k 전처리 의사코드

## 목적
이 문서는 `gsm8k`를 `docs/personal/preprocess/gsm8k.md`의 예시 형태 그대로 Lance row로 만드는 절차를 자연어로 설명한다.
이 데이터셋에서 Lance에 저장되는 row는 `messages`와 `content`를 함께 가지며, FT와 PT 학습은 같은 row를 서로 다른 필드로 사용한다.

## 고정값
구현할 때는 아래 값을 그대로 사용한다.

1. `source`는 `"gsm8k"`로 둔다.
2. system 문장은 `"당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."`로 둔다.
3. train 입력은 `data/korean_raw/gsm8k/train-00000-of-00001.parquet`로 둔다.
4. val 입력은 `data/korean_raw/gsm8k/test-00000-of-00001.parquet`로 둔다.
5. `split`은 입력 파일 기준으로 `train`, `val`만 사용한다.

## 입력 파일 탐색 절차
구현할 때는 아래 순서를 따른다.

1. train parquet 파일 하나를 입력 목록에 넣고 `split="train"`을 붙인다.
2. test parquet 파일 하나를 입력 목록에 넣고 `split="val"`을 붙인다.
3. 두 입력을 위 순서대로 처리한다.

## 파일 처리 절차
파일 하나를 처리할 때는 아래 순서를 따른다.

1. parquet 파일을 읽는다.
2. parquet를 row 단위로 순회한다.
3. 각 row에서 `question`, `answer`, `question_en`, `answer_en`를 읽는다.
4. 기본 학습 입력은 `question`, `answer`만 사용한다.

## 레코드 정규화 절차
레코드 하나를 row로 바꿀 때는 아래 순서를 따른다.

1. `question`이 문자열인지 확인한다.
2. `answer`가 문자열인지 확인한다.
3. 둘 중 하나라도 문자열이 아니면 이 레코드는 버리고 품질 이벤트를 남긴다.
4. `question`과 `answer`에 좌우 공백을 제거한다.
5. strip 후 `question`이 빈 문자열이면 이 레코드는 버린다.
6. strip 후 `answer`가 빈 문자열이면 이 레코드는 버린다.
7. `question_en`, `answer_en`는 읽더라도 기본 규칙에서는 사용하지 않는다.
8. `answer` 안의 줄바꿈, `<<...>>` 표기, `####` 정답 표기는 그대로 유지한다.

## row 생성 절차
정규화에 성공한 레코드는 아래 형태의 row 하나를 만든다.

1. `source`는 고정값을 넣는다.
2. `split`은 입력 파일에서 가져온 값을 넣는다.
3. `messages`는 길이 3의 배열로 만든다.
4. `messages[0]`에는 system 문장을 `role="system"`으로 넣는다.
5. `messages[1]`에는 정규화된 `question`을 `role="user"`로 넣는다.
6. `messages[2]`에는 정규화된 `answer`를 `role="assistant"`로 넣는다.
7. `content`는 `question + "\n" + answer` 형식으로 만든다.
8. `token_count`는 저장 직전에 FT 직렬화 기준으로 계산한다.

## PT 사용 전략
이 데이터셋의 PT 입력은 별도 PT row를 만들지 않고 같은 row의 `content`를 사용한다.

1. PT 학습 시 row의 `content`를 그대로 사용한다.
2. `messages`는 PT 입력 생성에서 사용하지 않는다.

## FT 직렬화와 token_count
`token_count`를 계산할 때는 아래 순서를 따른다.

1. `messages`를 `<|system|>...<|user|>...<|assistant|>...` 형태로 직렬화한다.
2. 직렬화된 문자열을 공통 토크나이저로 인코딩한다.
3. 나온 토큰 수를 `token_count`에 넣는다.

## 품질 이벤트 기록 방식
품질 이벤트를 남길 때는 최소한 아래 정보를 기록한다.

1. dataset은 `gsm8k`
2. split
3. file path
4. row index
5. reason code
6. reason detail
7. sample text 일부
8. `skip`, `error` 중 성격

## 구현 체크리스트
1. split 매핑이 `train parquet -> train`, `test parquet -> val`로 고정되는가
2. `question`, `answer`만 사용하고 영어 컬럼은 기본 규칙에서 제외하는가
3. `messages`가 `[system, user(question), assistant(answer)]` 순서를 가지는가
4. `content`가 `question + "\n" + answer` 규칙으로 생성되는가
5. `answer` 안의 줄바꿈과 풀이 표기를 보존하는가
6. `token_count`를 FT 직렬화 기준으로 계산하는가
