import os
import pandas as pd
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.agents import create_pandas_dataframe_agent

import constants

def convert_price_to_int(price_str):
    parts = price_str.split('억')
    billion = int(parts[0]) * 100000000 if parts[0] else 0
    million = int(parts[1].replace(',', '').strip()) * 10000 if len(parts) > 1 and parts[1].strip() else 0
    return billion + million

os.environ["OPENAI_API_KEY"] = constants.APIKEY
#temperature 값을 수정하여 모델의 온도를 변경할 수  있다. default는 0.7, 1에 가까울 수록 다양성 증진
chat3 = OpenAI(temperature=0)
# chat3([SystemMessage(content="한국어로 완성된 문장으로 대답해줘")])

chat4 = ChatOpenAI(model_name='gpt-4', temperature=0.2, openai_api_key=constants.APIKEY)
chat4([SystemMessage(content="한국어로 완성된 문장으로 대답해줘")])

df = pd.read_csv('data/HelioCity.csv')

###데이터 전처리 ###
df = df[~df['Price'].str.contains("취소", na=False, case=False)]
df = df[~df['Price'].str.contains("직", na=False, case=False)]
df = df[~df['Floor'].str.contains("입주권", na=False, case=False)]
df = df[~df['Floor'].str.contains("분양권", na=False, case=False)]

df['Price'] = df['Price'].apply(convert_price_to_int)
df['Floor'] = df['Floor'].str.extract(r'(\d+)층')[0].astype(int)

print(df.head())

# GPT version 3 사용
agent = create_pandas_dataframe_agent(OpenAI(temperature=0.3),df,verbose=True)

# GPT version 4 사용
# agent = create_pandas_dataframe_agent(chat4,df,verbose=True) 
# verbose는 생각의 과정을 설정해준다.

Answer = []

# print(agent.run("몇개의 행이 있어?"))
Answer.append(agent.run("상위 30% 층의 평균 가격을 몇억 몇천 만원인지까지만 알려줘"))
Answer.append(agent.run("하위 30% 층의 평균 가격을 몇억 몇천 만원인지까지만 알려줘"))
Answer.append(agent.run("최근 거래량의 방향성과 판단한 정확한 이유를 알려줘"))
Answer.append(agent.run("최근 가격의 방향성과 판단한 정확한 이유를 알려줘"))
Answer.append(agent.run("평균 가격에 비해 가장 최근 가격이 몇억 몇천 만원 높은지 낮은지 알려줘"))
Answer.append(agent.run("최고가 대비 최근 가격이 몇억 몇천 만원 낮은지 알려줘"))
Answer.append(agent.run("최저가 대비 최근 가격이 몇억 몇천 만원 높은지 알려줘"))

print(Answer)


print("\n***Test Done***\n\n")