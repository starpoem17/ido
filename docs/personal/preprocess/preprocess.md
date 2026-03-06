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

특수 토큰을 전처리 시에 넣는 게 아니다. 전처리 시에는 자연어 형태 텍스트만 입력하고 이를 불러올 때에 특수 토큰을 붙인다.

FT 데이터는 사전 학습 시 세 개의 "content" 키의 텍스트들을 concat한 뒤 사용한다.
    ex) "<|bos|><|system|>너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해.<|eot_id|><|user|>임진왜란이 언제 일어났는지 알려줘.<|eot_id|><|assistant|>임진왜란은 선조 25년, 기원후 1592년에 발발했습니다.<|eos|>"
    이렇게 system, user, assistant 토큰을 사이에 붙이면서 concat한 뒤 PT의 content와 똑같이 사용한다

FT 학습 시에는 모델에 "<|bos|><|system|>너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해.<|eot_id|><|user|>임진왜란이 언제 일어났는지 알려줘.<|eot_id|><|assistant|>"를 입력하고 라벨은 "임진왜란은 선조 25년, 기원후 1592년에 발발했습니다."가 된다. 이 경우 teacher forcing을 적용하여 학습시킨다. 즉,
"<|bos|><|system|>너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해.<|eot_id|><|user|>임진왜란이 언제 일어났는지 알려줘.<|eot_id|><|assistant|>" 다음에 "임",
"<|bos|><|system|>너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해.<|eot_id|><|user|>임진왜란이 언제 일어났는지 알려줘.<|eot_id|><|assistant|>임" 다음에 "진", (설사 모델의 앞에서 '임'이 아닌 다른 토큰을 예측했더라도 teacher forcing을 적용하여 입력은 '임'이 들어간다.)
...
"<|bos|><|system|>너는 유능한 비서야. 사용자의 질문에 친절하고 자세하게 대답해.<|eot_id|><|user|>임진왜란이 언제 일어났는지 알려줘.<|eot_id|><|assistant|>임진왜란은 " 다음에 "선"이 오도록 학습시킨다.