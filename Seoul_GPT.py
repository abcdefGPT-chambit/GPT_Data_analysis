import os
import pandas as pd
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import PromptTemplate

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_pandas_dataframe_agent

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY
chat = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key=constants.APIKEY)
#temperature 값을 수정하여 모델의 온도를 변경할 수  있다. default는 0.7, 1에 가까울 수록 다양성 증진
chat([SystemMessage(content="한국어로 완성된 문장으로 대답해줘")])


# Load Data
df = pd.read_csv('data/Seoul_Transaction_Sep.csv')
# print(df.head())
Seoul_df = df.copy()
Data_df = df.copy()
Data_df.drop(0, inplace=True)
Data_df.reset_index(drop=True, inplace=True)
Seoul_df = Seoul_df.iloc[:1]

# GPT version 3 사용
# agent1 = create_pandas_dataframe_agent(OpenAI(temperature=0),Seoul_df,verbose=True)
# agent2 = create_pandas_dataframe_agent(OpenAI(temperature=0),Data_df,verbose=True)
# GPT version 4 사용
agent1 = create_pandas_dataframe_agent(chat,Seoul_df,verbose=True)
agent2 = create_pandas_dataframe_agent(chat,Data_df,verbose=True)
# verbose는 생각의 과정을 설정해준다.

Answer = []

#Question 1 : 거래량 증가
Answer.append(agent2.run("8월에서 9월 사이에 거래량의 비율이 가장 많이 상승한 위치와 몇퍼센트인지 알려줘."))
print(Answer[0])

#Question 2 : 거래량 감소
Answer.append(agent2.run("8월에서 9월 사이에 거래량의 비율이 가장 많이 감소한 위치와 몇퍼센트인지 알려줘."))
print(Answer[1])

#Question 3 : 서울시의 방향성
Answer.append(agent1.run("23년 들어서 거래량의 방향성은 어때? 완결된 한국어 문장으로 답변해줘"))
print(Answer[2])

print(Answer)

print("\n***Test Done***\n\n")