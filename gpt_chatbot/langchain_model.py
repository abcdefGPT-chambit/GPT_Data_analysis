import os
import constants

os.environ['OPENAI_API_KEY'] = constants.APIKEY
import pandas as pd
df = pd.read_csv('Seoul_Transaction_Sep.csv')
print(df.head())

from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType


# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-4-0613'),        # 모델 정의
    df,                                    # 데이터프레임
    verbose=True,                          # 추론과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

print(agent.run('광진구 방향성은 어때?'))