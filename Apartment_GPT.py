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

chat1 = ChatOpenAI(model_name='gpt-4', temperature=0.2, openai_api_key=constants.APIKEY)
# chat1 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=constants.APIKEY)
# chat1([SystemMessage(content="한국어로 변역해줘")])

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
agent1 = create_pandas_dataframe_agent(OpenAI(temperature=0.2),df,verbose=True)

# GPT version 4 사용
agent2 = create_pandas_dataframe_agent(chat1,df,verbose=True) 
# verbose는 생각의 과정을 설정해준다.

Answer = []
Final_Answer = []

# Temp1 = agent2.run("상위 30% 층의 평균 가격을 알려줘. 가격에 대해
#  가격을 만원 단위로 바꿔줘")
# print(Temp1)
# Temp2 = Temp1 + "앞의 문장의 가격을 만원 단위로 바꿔줘. 미만의 금액은 버려줘. 나머지 문장은 그대로 둬. 한글이 아니라면 한글로 바꿔줘"
# Temp2 = agent2.run(Temp2)
# print(Temp2)
# Answer.append(Temp2)

Temp1 = agent1.run("Please tell me the average price of the top 30%% floors of apartments. in won")
print(Temp1)
Temp2 = Temp1 +" 의 문장을 한국어로 번역해줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Temp2 = Temp2 +" 에서 가격을 만원 단위로 정리한 문장을 만들어줘. 나머지 금액은 버려줘"
Temp3 = agent2.run(Temp2)
print(Temp3)
Answer.append(Temp3)

# # Answer.append(agent.run("하위 30% 층의 평균 가격을 몇억 몇천 만원인지 알려줘"))
Temp1 = agent1.run("Please tell me the average price of the bottom 30%% floors of apartments. in won")
print(Temp1)
Temp2 = Temp1 +" 의 문장을 한국어로 번역해줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Temp2 = Temp2 +" 에서 가격을 만원 단위로 정리한 문장을 만들어줘. 나머지 금액은 버려줘"
Temp3 = agent2.run(Temp2)
print(Temp3)
Answer.append(Temp3)

# print(Temp3)
# Answer.append(agent.run("하위 30% 층의 평균 가격을 몇억 몇천 만원인지 알려줘"))
# Answer.append(agent.run("최근 거래량의 방향성과 판단한 정확한 이유를 알려줘"))
# Answer.append(agent1.run("최근 가격의 방향성과 판단한 정확한 이유를 알려줘"))
# Answer.append(agent1.run("평균 가격에 비해 가장 최근 가격이 얼마나 높은지 낮은지 알려줘"))
# Temp1 = agent2.run("Tell me the most how much recent price in data compared to the highest price, and what percentage")
Temp1 = agent2.run("평균 가격에 비해 가장 최신의 거래와(1행)의 가격이 얼마나 높은지 낮은지 알려줘. 추가로 비율도 알려줘.")
print(Temp1)
Temp2 = Temp1 + "앞의 문장의 가격의 차이에 대해 단위를 만원 단위로 바꿔줘. 미만의 금액은 버려줘. 나머지 문장은 그대로 둬. 한글이 아니라면 한글로 바꿔줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Answer.append(Temp2)

# Answer.append(agent2.run("Tell me the price and percentage how high the latest trading day price is compared to the lowest price"))

print(Answer)
print(Final_Answer)

print("\n***Test Done***\n\n")