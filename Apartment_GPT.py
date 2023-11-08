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

from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY
chat = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key=constants.APIKEY)
#temperature 값을 수정하여 모델의 온도를 변경할 수  있다. default는 0.7, 1에 가까울 수록 다양성 증진

#############################
##      Input Prompt       ##
#############################

GPT_Input_prompt = PromptTemplate(
    input_variables=["floor"], 
    template="{floor}층의 최신 1년의 평균 가격을 알려줘"
)

GPT_Input_prompt.format(floor="상위 30%") #input prompt example

# print(GPT_Input_prompt.format(floor="상위 30%")) #output prompt example


#############################
##      Data Loader        ##
#############################

# loader = TextLoader('data/data2.txt')
loader = CSVLoader('data/HelioCity.csv')
data = loader.load()

df = pd.read_csv('data/HelioCity.csv')
# df = pd.read_csv('data/Seoul_Transaction_Sep.csv')
print(df.head())

# GPT version 3 사용
# agent = create_pandas_dataframe_agent(OpenAI(temperature=0),df,verbose=True)
# GPT version 4 사용
agent = create_pandas_dataframe_agent(chat,df,verbose=True) 
# verbose는 생각의 과정을 설정해준다.

# print(agent.run("몇개의 행이 있어?"))
print(agent.run("상위 30% 층의 평균 가격을 알려줘"))
# print(agent.run("하위 30% 층의 평균 가격을 알려줘"))
# print(agent.run("최근 거래량의 방향성을 알려줘"))
# print(agent.run("최근 가격의 방향성을 알려줘"))

###Start

# # 스플리터 준비하기
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50) #1000자 단위로 자른다.
# all_splits = text_splitter.split_documents(data)

# # 문서를 텍스트로 분할
# texts = text_splitter.split_documents(data)

# # 임베딩 엔진 준비하기
# embeddings = OpenAIEmbeddings()

# print (f"You have {len(texts)} documents")
# print(texts[1])

# Text = ""
# for temp_text in texts:
#     # print(temp_text.page_content)
#     Text+=temp_text.page_content

# print(Text)

# embedding_list = embeddings.embed_documents([text.page_content for text in texts])

# print (f"You have {len(embedding_list)} embeddings")
# print (f"Here's a sample of one: {embedding_list[0][:3]}...")

# qa_chain = load_qa_chain(chat, chain_type="map_reduce")
# qd_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)


# # print(qd_document_chain.run(
# #     input_document = Text,
# #     question = "2019년 9월 28일 가격을 알려줘"
# # ))


# # sys = SystemMessage(content="당신은 부동산의 거래량을 분석해주는 AI 봇이야")
# # msg = HumanMessage(content='23년 10월 5일에 거래된 가격은 얼마야?')

# # aimsg = chat([sys, msg])
# # print(aimsg.content)

print("\n***Test Done***\n\n")