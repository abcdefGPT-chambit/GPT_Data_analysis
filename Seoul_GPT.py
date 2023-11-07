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

# Load Data
df = pd.read_csv('data/Seoul_Transaction_Sep.csv')
print(df.head())

# GPT version 3 사용
# agent = create_pandas_dataframe_agent(OpenAI(temperature=0),df,verbose=True)
# GPT version 4 사용
agent = create_pandas_dataframe_agent(chat,df,verbose=True) 
# verbose는 생각의 과정을 설정해준다.

Answer = []

#Question 1 : 거래량 증가
Answer.append(agent.run("8월에서 9월 사이에 가장 거래량이 많이 상승한 위치는 어디야?"))

#Question 2 : 거래량 감소
Answer.append(agent.run("8월에서 9월 사이에 가장 거래량이 많이 감소한 위치는 어디야?"))
print(Answer[0])

print("\n***Test Done***\n\n")