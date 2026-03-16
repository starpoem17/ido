특수 토큰
<|bos|> : begin of sequence
<|eos|> : end of sequence
<|system|>, <|user|>, <|assistant|> : 텍스트 발화 주체. 학습 시 마스킹(이 토큰들을 예측할 필요가 없다.)
<|eot_id|> : end of turn. 전체 종료X. 한 화자의 발언 종료.
<|pad|> : 배치 내에서 가장 긴 텍스트의 길이에 맞춰 짧은 텍스트의 빈 공간 패딩. 학습 시에는 짧은 문장의 오른쪽을 패딩한다. 이후 추론 서비스 시에는 왼쪽을 패딩한다(다수 사용자의 쿼리를 동시 처리할 때)

데이터 feature 6개

source : String
    ex)
    "009.전문분야_기술과학_한국어 멀티세션 데이터"
    "gsm8k"
    "novel24"

data_usage : String
    ex)
    "PT",
    "SFT",
    "REASONING"

split : String
    ex)
    "train"
    "val"

content : String | null
    PT 학습 입력으로 사용하는 텍스트
    ex)
    "임진왜란은 1592년 발발했습니다."

messages : List[Dict] | null
    FT 학습 입력으로 사용하는 대화 배열
    ex)
    "messages" : [
        {"role" : "system", "content" : "너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해."},
        {"role" : "user", "content" : "임진왜란이 언제 일어났는지 알려줘."},
        {"role" : "assistant", "content" : "임진왜란은 선조 25년, 기원후 1592년에 발발했습니다."},
        {"role" : "user", "content" : "..."},
        {"role" : "assistant", "content" : "..."},
        ...
    ]

token_count : int32
    저장 row를 학습 입력으로 직렬화했을 때의 토큰 길이
    기본 규칙: messages가 있으면 messages 직렬화 기준, 없으면 content 기준

공통 규칙
1. content와 messages는 둘 다 nullable이지만, 최소 하나는 반드시 채운다.
2. FT 학습에서는 messages의 content를 사용한다.
3. PT 학습에서는 content를 사용한다.
4. 특정 학습 모드에서 필요한 필드가 null인 row는 해당 모드에서만 제외한다.

lance 데이터 예시
ex)
{
    "source" : "045.지식검색 대화",
    "data_usage" : "SFT",
    "split" : "train",
    "content" : "임진왜란이 언제 일어났는지 알려줘. 임진왜란은 선조 25년, 기원후 1592년에 발발했습니다.",
    "messages" : [
        {"role" : "system", "content" : "너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해."},
        {"role" : "user", "content" : "임진왜란이 언제 일어났는지 알려줘."},
        {"role" : "assistant", "content" : "임진왜란은 선조 25년, 기원후 1592년에 발발했습니다."}
    ],
    "token_count" : 456
}

{
    "source" : "novel24",
    "data_usage" : "PT",
    "split" : "train",
    "content" : "...",
    "messages" : null,
    "token_count" : 123
}

보다 구체적인 데이터 종류별 포맷은 docs/personal/preprocess 안의 md 파일들을 참조하여 확인한다

각 데이터 종류별로 전처리를 진행하고 lance에 append하는 방식으로 하나의 lance 데이터셋에 전체 데이터를 정리한다

split의 경우 99:1 비율로 train:val 데이터를 구성한다. data/korean_raw 안의 데이터를 보면 train, val로 이미 나눠진 데이터가 있고 그렇지 않은 데이터가 있는데 이들 모두 99:1 비율로 재구성한다. 1%의 val 데이터는 각 데이터 항목에서 랜덤으로 추출한다.

각 데이터 소스 별로 몇 개의 토큰이 있는지 통계를 남긴다. 이 경우 namu 데이터는 특수하게 처리하는데 구체적인 사항은 docs/personal/preprocess/namu.md 를 참조한다.

### 파이프라인
- data/korean_raw 안의 데이터를 통일된 형식의 parquet 파일로 변환한다. 구체적인 변환 전략은 docs/personal/preprocess 안의 문서들을 참조한다.
- exact dedup으로 완전히 동일한 content 필드를 갖는 행을 제거한다.
- minhash + lsh 로 중복 후보를 제거한다.
- 중복이 제거된 데이터를 활용해 토크나이저를 빌드한다. 토크나이저의 사전 크기는 48k를 디폴트로 한다
- 토크나이저 빌드 이후 비어있는 token_count 필드를 채워넣는다. token_count 필드 안의 값은 content 필드의 텍스트 토큰 길이를 기준으로 한다.
- 이후 각 데이터 항목 별로 lance 데이터셋을 구축(이 경우 샤딩을 진행하는데 샤드 하나의 크기는 1gb로 한다.)한 뒤 append하여 LLM 학습에 사용할 최종 lancedb를 완성한다. 