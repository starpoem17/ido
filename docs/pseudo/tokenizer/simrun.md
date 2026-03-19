# 토크나이저 simrun 스도코드

## 목적
이 문서는 [docs/plans/tokenizer.md](/home/hwajoong/projects/ido/docs/plans/tokenizer.md)의
simrun 흐름을 실제 코드로 옮길 때 따라야 하는 한국어 자연어 의사코드다.
사용자가 터미널에 직접 텍스트를 입력하면, 해당 tokenizer로 토큰화했을 때 어느 정도 압축률을 보이는지 즉시 확인한다.

## 사용자 설정값
구현할 때 자주 바뀌는 값은 터미널 인자가 아니라 파일 맨 앞 상수 블록에 둔다.

1. `TOKENIZER_JSON_PATH = "data/tokenizers/korean_bbpe_vN/tokenizer.json"`
2. `MAX_DISPLAY_TOKENS = 64`
3. `SHOW_TOKEN_BREAKDOWN = True`
4. `EXIT_COMMANDS = ["exit", "quit", ":q"]`

주의:
- 입력 tokenizer는 특정 버전 디렉터리의 산출물을 명시적으로 지정한다.
- 이 도구는 build나 benchmark를 대체하지 않는 대화형 보조 점검 도구다.

## 실행 형식
실행 코드는 `uv run python -m src.tokenizer.simrun` 형태로 작성한다.

## 전체 실행 절차
`simrun_실행()`은 아래 순서로 동작한다.

1. 실행 시작 시 tokenizer 경로와 사용자 설정값을 로그로 남긴다.
2. `TOKENIZER_JSON_PATH`에서 tokenizer를 로드한다.
3. 사용자에게 입력 안내 문구와 종료 명령 목록을 출력한다.
4. 터미널에서 텍스트 한 줄을 입력받는다.
5. 입력이 `EXIT_COMMANDS` 중 하나면 종료 로그를 남기고 정상 종료한다.
6. 입력이 비어 있으면 빈 입력이라는 안내를 출력하고 다시 다음 입력을 기다린다.
7. 입력 텍스트를 tokenizer로 encode한다.
8. 문자 수, UTF-8 바이트 수, 토큰 수를 계산한다.
9. `tokens_per_char`, `chars_per_token`, `tokens_per_byte` 같은 압축률 지표를 계산한다.
10. 계산 결과를 사람이 읽기 쉬운 형식으로 즉시 출력한다.
11. `SHOW_TOKEN_BREAKDOWN = True`면 token breakdown preview도 함께 출력한다.
12. 다시 다음 입력을 기다린다.

## 압축률 계산 절차
`압축률_계산()`은 아래 순서로 동작한다.

1. 문자 수는 Python `len(text)`로 계산한다.
2. 바이트 수는 `len(text.encode("utf-8"))`로 계산한다.
3. 토큰 수는 `len(encoded.ids)`로 계산한다.
4. `tokens_per_char = token_count / char_count`를 계산한다. 문자 수가 0이면 0으로 둔다.
5. `chars_per_token = char_count / token_count`를 계산한다. 토큰 수가 0이면 0으로 둔다.
6. `tokens_per_byte = token_count / byte_count`를 계산한다. 바이트 수가 0이면 0으로 둔다.

## token breakdown 출력 절차
`토큰분해_출력()`은 아래 순서로 동작한다.

1. encode 결과의 token id와 token 문자열을 읽는다.
2. 최대 `MAX_DISPLAY_TOKENS`개까지만 출력한다.
3. 각 token에 대해 아래를 함께 보여준다.
   - 순번
   - `token_id`
   - raw token 문자열
   - 사람이 읽기 쉬운 decoded 또는 escaped 표현
4. token 수가 `MAX_DISPLAY_TOKENS`를 넘으면 뒤쪽은 생략하고 생략 개수를 표시한다.

## 출력 형식
한 번의 입력에 대해 최소한 아래를 출력한다.

1. 원문 텍스트
2. 문자 수
3. UTF-8 바이트 수
4. 토큰 수
5. `tokens_per_char`
6. `chars_per_token`
7. `tokens_per_byte`

## 최종 검증
실행 중 아래가 만족되어야 한다.

1. tokenizer를 정상 로드할 수 있는가
2. 사용자가 입력한 텍스트에 대해 문자 수, 바이트 수, 토큰 수, 압축률 지표가 즉시 출력되는가
3. `SHOW_TOKEN_BREAKDOWN = True`일 때 token breakdown preview가 출력되는가
4. 종료 명령을 입력하면 정상 종료하는가

## fail fast 조건
아래 중 하나라도 만족하면 즉시 중단한다.

1. tokenizer 파일이 없는 경우
2. tokenizer 로드에 실패한 경우
3. tokenizer encode 단계에서 복구 불가능한 예외가 발생한 경우
