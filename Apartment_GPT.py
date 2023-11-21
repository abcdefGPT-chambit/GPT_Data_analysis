import re
import os
import csv
import pandas as pd
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from datetime import datetime

from langchain.agents import create_pandas_dataframe_agent

import constants

def convert_price_to_int(price_str):
    parts = price_str.split('억')
    billion = int(parts[0]) * 100000000 if parts[0] else 0
    million = int(parts[1].replace(',', '').strip()) * 10000 if len(parts) > 1 and parts[1].strip() else 0
    return billion + million

def format_price_correctly(text):
    # '만원'이 포함된 모든 인스턴스 찾기
    price_matches = re.finditer(r'(\d+)( 만원|만원)', text)
    for match in price_matches:
        price = match.group(1)
        # 숫자를 10000으로 나누어 '억' 단위로 변경 (소수점 아래는 버림)
        billions = int(price) // 10000
        # 나머지 '만' 단위
        millions = int(price) % 10000
        # 변경할 문자열 생성
        formatted_price = f'{billions}억{millions} 만원' if billions > 0 else f'{millions} 만원'
        # 원본 텍스트 내 해당 부분을 새로운 형식으로 교체
        text = text.replace(match.group(0), formatted_price)
    return text

os.environ["OPENAI_API_KEY"] = constants.APIKEY
#temperature 값을 수정하여 모델의 온도를 변경할 수  있다. default는 0.7, 1에 가까울 수록 다양성 증진

Answer = []


###############################
##### select Apt and type #####
###############################
df = pd.read_csv('data/10_ParkRio_52py.csv')
Answer.append('1Xyb5')
Answer.append('52')


chat1 = ChatOpenAI(model_name='gpt-4', temperature=0.1, openai_api_key=constants.APIKEY)

###데이터 전처리 ###
df = df[~df['Price'].str.contains("취소", na=False, case=False)]
df = df[~df['Price'].str.contains("직", na=False, case=False)]
df = df[~df['Floor'].str.contains("입주권", na=False, case=False)]
df = df[~df['Floor'].str.contains("분양권", na=False, case=False)]

df['Price'] = df['Price'].apply(convert_price_to_int)
df['Floor'] = df['Floor'].str.extract(r'(\d+)층')[0].astype(int)

print(df.head())
Transaction = df.copy()
Transaction['Date'] = pd.to_datetime(Transaction['Date'], format='%y.%m.%d')
Transaction = Transaction[Transaction['Date'] > datetime(2022, 10, 1)]
# '년도-월' 기반으로 그룹화하여 항목 수 계산
monthly_counts = Transaction.groupby(Transaction['Date'].dt.to_period('M')).size()

# 새로운 데이터프레임 생성
Transaction_df = monthly_counts.reset_index(name='Count')
Transaction_df.columns = ['Month', 'Count']  # 열 이름 변경
Transaction_df['Month'] = Transaction_df['Month'].dt.to_timestamp()  # Period를 datetime으로 변환
Transaction_df['Month'] = pd.to_datetime(Transaction_df['Month']).dt.to_period('M')
Transaction_df =Transaction_df.rename(columns={'Count':'Volume'})
Transaction_df = Transaction_df[Transaction_df['Month'] != '2023-10']
print(Transaction_df)

# GPT version 3 사용
agent1 = create_pandas_dataframe_agent(OpenAI(temperature=0.1),df,verbose=True)

# GPT version 4 사용
agent2 = create_pandas_dataframe_agent(chat1,df,verbose=True) 
# verbose는 생각의 과정을 설정해준다.
# GPT version 4의 거래량 전처리 데이터 사용
agent3 = create_pandas_dataframe_agent(chat1,Transaction_df,verbose=True) 

#####Question1#####
Temp1 = agent2.run("Tell me the average price of the apartment. in won")
print(Temp1)
Temp2 = Temp1 +" 의 문장을 한국어로 번역해줘"
# Temp2 = agent2.run(Temp2)
Temp2 = chat1([HumanMessage(content=Temp2)])
Temp2 = Temp2.content
print(Temp2)
if '억' in Temp2:
    Answer.append(Temp2)
else:
    Temp2 = Temp2 +" 에서 가격을 만원 단위로 정리한 문장을 만들어줘. 나머지 금액은 버려줘"
    # Temp3 = chat1([HumanMessage(content=Temp2)])
    # Temp3 = Temp3.content
    Temp3 = agent2.run(Temp2)
    print(Temp3)
    Temp3 = Temp3.replace(",","")
    Temp4 = format_price_correctly(Temp3)
    print(Temp4)
    Answer.append(Temp4)

#####Question2#####
Temp1 = agent2.run("Please tell me the average price of the top 30%% floors of apartments. in won. Please do not give me final answer in variable!")
print(Temp1)
Temp2 = Temp1 +" 의 문장을 한국어로 번역해줘"
# Temp2 = agent2.run(Temp2)
Temp2 = chat1([HumanMessage(content=Temp2)])
Temp2 = Temp2.content
print(Temp2)
if '억' in Temp2:
    Answer.append(Temp2)
else:
    Temp2 = Temp2 +" 에서 가격을 만원 단위로 정리한 문장을 만들어줘. 나머지 금액은 버려줘"
    # Temp3 = chat1([HumanMessage(content=Temp2)])
    # Temp3 = Temp3.content
    Temp3 = agent2.run(Temp2)
    print(Temp3)
    Temp3 = Temp3.replace(",","")
    Temp4 = format_price_correctly(Temp3)
    print(Temp4)
    Answer.append(Temp4)

#####Question3#####
Temp1 = agent2.run("Please tell me the average price of the bottom 30%% floors of apartments. in won. Please do not give me final answer in variable!")
print(Temp1)
Temp2 = Temp1 +" 의 문장을 한국어로 번역해줘"
# Temp2 = agent2.run(Temp2)
Temp2 = chat1([HumanMessage(content=Temp2)])
Temp2 = Temp2.content
print(Temp2)
if '억' in Temp2:
    Answer.append(Temp2)
else:
    Temp2 = Temp2 +" 에서 가격을 만원 단위로 정리한 문장을 만들어줘. 나머지 금액은 버려줘"
    # Temp3 = chat1([HumanMessage(content=Temp2)])
    # Temp3 = Temp3.content
    Temp3 = agent2.run(Temp2)
    print(Temp3)
    Temp3 = Temp3.replace(",","")
    Temp4 = format_price_correctly(Temp3)
    print(Temp4)
    Answer.append(Temp4)

#####Question4#####
Temp1 = agent2.run("평균 가격에 비해 가장 최신의 거래와(1행)의 가격이 얼마나 높은지 낮은지 알려줘. 추가로 비율도 알려줘.")
print(Temp1)
Temp2 = Temp1 + "앞의 문장의 가격의 차이에 대해 단위를 만원 단위로 바꿔줘. 미만의 금액은 버려줘. 나머지 문장은 그대로 둬. 한글이 아니라면 한글로 바꿔줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Temp3 = format_price_correctly(Temp2)
print(Temp3)
Answer.append(Temp3)

#####Question5#####
Temp1 = agent2.run("최고 가격을 알려줘. 최고 가격에 비해 가장 최신의 거래와(1행)의 가격이 얼마나 낮은지 알려줘. 추가로 비율도 알려줘. 제발 변수를 이용하지 말아줘!")
print(Temp1)
Temp2 = Temp1 + "앞의 문장의 가격에 대해 단위를 만원 단위로 바꿔줘. 미만의 금액은 버려줘. 나머지 문장은 그대로 둬. 한글이 아니라면 한글로 바꿔줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Temp2 = Temp2.replace(",","")
Temp3 = format_price_correctly(Temp2)
print(Temp3)
Answer.append(Temp3)

#####Question6#####
Temp1 = agent2.run("최저 가격을 알려줘. 최저 가격에 비해 가장 최신의 거래와(1행)의 가격이 얼마나 높은지 알려줘. 추가로 비율도 알려줘. 제발 변수를 이용하지 말아줘!")
print(Temp1)
Temp2 = Temp1 + "앞의 문장의 가격에 대해 단위를 만원 단위로 바꿔줘. 미만의 금액은 버려줘. 나머지 문장은 그대로 둬. 한글이 아니라면 한글로 바꿔줘"
Temp2 = agent2.run(Temp2)
print(Temp2)
Temp2 = Temp2.replace(",","")
Temp3 = format_price_correctly(Temp2)
print(Temp3)
Answer.append(Temp3)

#####Question7#####
Temp1 = agent3.run("전체적인 거래량중 최근 거래량의 방향성을 알려줘. 그렇게 답한 상세한 이유도 알려줘. 한글로 대답해줘")
Temp1 = Temp1.replace(",","")
print(Temp1)
Answer.append(Temp1)

######Question8#####
Temp1 = agent2.run("최근 가격의 방향성을 알려줘. 행이 낮을 수록 최신 데이터야. 판단한 이유도 말해줘")
print(Temp1)
Answer.append(Temp1)

# print(Answer)

# File = open('result/Apt_transaction_result.csv', 'w',newline='')
# data=['apt_code','apt_sq','avg_price','top_avg_price','bottom_avg_price','most_recent_trade','recent_trade_trend','recent_price_trend']
# writer = csv.writer(File)
# writer.writerow(data)

# File = open('result/Apt_transaction_result.csv', 'a',newline='')
# writer = csv.writer(File)

# data=[Answer[0],Answer[1],Answer[2],Answer[3],Answer[4],Answer[5],Answer[6],Answer[7],Answer[8],Answer[9]]
# writer.writerow(data)
# File.close()

print("\n*** Test Done***\n\n")