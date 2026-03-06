특수 토큰
<|bos|> : begin of sequence
<|eos|> : end of sequence
<|system|>, <|user|>, <|assistant|> : 텍스트 발화 주체. 학습 시 마스킹(이 토큰들을 예측할 필요가 없다.)
<|eot_id|> : end of turn. 전체 종료X. 한 화자의 발언 종료.
<|pad|> : 배치 내에서 가장 긴 텍스트의 길이에 맞춰 짧은 텍스트의 빈 공간 패딩. 학습 시에는 짧은 문장의 오른쪽을 패딩한다. 이후 추론 서비스 시에는 왼쪽을 패딩한다(다수 사용자의 쿼리를 동시 처리할 때)

데이터 feature 6개

data_type : String
    ex)
    "PT" : PreTrain 사전 학습용 텍스트
    "FT" : FineTune 사전 학습, 파인튜닝에 모두 적용되는 텍스트

source : String
    ex)
    "009.전문분야_기술과학_한국어 멀티세션 데이터"
    "gsm8k"
    "novel24"

split : String
    ex)
    "train"
    "val"

content : String 
    PT용 텍스트. FT일 때는 null
    ex)
    "임진왜란은 1592년 발발했습니다."

messages : List[Dict]
    FT용 대화 배열
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
    토큰화 했을 때 토큰 길이

lance 데이터 예시
ex)
{
    "data_type" : "PT",
    "source" : "045.지식검색 대화",
    "split" : "train",
    "content" : "...",
    "messages" : null,
    "token_count" : 123
}

{
    "data_type" : "FT",
    "source" : "045.지식검색 대화",
    "split" : "val",
    "content" : null,
    "messages" : [
        {"role" : "system", "content" : "너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해."},
        {"role" : "user", "content" : "임진왜란이 언제 일어났는지 알려줘."},
        {"role" : "assistant", "content" : "임진왜란은 선조 25년, 기원후 1592년에 발발했습니다."}
    ],
    "token_count" : 456
}

보다 구체적인 데이터 종류별 포맷은 docs/personal/preprocess 안의 md 파일들을 참조하여 확인한다

각 데이터 종류별로 전처리를 진행하고 lance에 append하는 방식으로 하나의 lance 데이터셋에 전체 데이터를 정리한다