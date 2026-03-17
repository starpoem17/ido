# gsm8k 전처리 스도코드

## 목적
`gsm8k`를 SFT용 canonical parquet row로 만들고 Lance 업로드 규칙을 적는다.

## 고정값
1. `source = "gsm8k"`
2. `data_usage = "SFT"`
3. `system = "당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."`
4. 입력 경로는 `data/korean_raw/gsm8k/*.parquet`

## parquet 생성 규칙
1. parquet row 하나를 canonical row 하나로 본다.
2. `split_key`는 원본 row id 우선, 없으면 `file_path + row_index`다.
3. `question`과 `answer`를 strip한다.
4. 둘 중 하나라도 비면 row를 제외한다.
5. `content = question + "\n" + answer`
6. `messages = [system, user(question), assistant(answer)]`
7. 원본 train/test 구분은 무시하고 최종 split은 공통 99:1 helper를 사용한다.

## Lance 업로드 규칙
1. minhash dedup 완료 parquet에서 `source == gsm8k`만 읽는다.
2. `content` 기준 `token_count`를 채운다.
3. source별 Lance dataset으로 저장한다.

## 품질 이벤트
- empty question
- empty answer
- parquet read failure
