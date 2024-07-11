
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import ChatMessage

import collections
try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
except:
    pass
import collections 
import sys
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    from collections.abc import MutableSet
    collections.MutableSet = collections.abc.MutableSet
else: 
    from collections import MutableSet
import os 
from dotenv import load_dotenv

load_dotenv()

key = st.secrets["OPENAI_API_KEY"]
   
st.set_page_config(page_title=" AI투자상담 통통선생", page_icon="🍅")
st.title("🍅 AI 투자상담 통통선생")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

#이전 채팅 내용 초기화 안되게

from utils import print_messages
print_messages()

if prompt := st.chat_input("메시지를 입력하세요."):
    #사용자 input
    st.chat_message("user").write(f"{prompt}")
    #st.session_state["messages"].append(("user", prompt))
    st.session_state["messages"].append(ChatMessage(role ="user",content=prompt))
 
    #LLM 구축
    promptt = PromptTemplate.from_template(
    """
    You are an assistant specialized in providing information strictly within the domain of [SPECIFIC DOMAIN].
    NEVER provide any information outside of this domain. Answer must be KOREAN. Defalut 10개.
    Basically, respond in a polite way.
    *** 출처 관련한 질문이 들어올 시 출처는 "증권통" 으로 답변.


    *** 골든크로스종목의 정보는 아래 사이트 
    '한국앤컴퍼니',
    '홈캐스트',
    '대호에이엘',
    '바이오플러스',
    'KC코트렐',
    '메리츠금융지주',
    '에코캡',
    '동아에스티',
    '효성중공업',
    '유진테크놀로지',
    '윌링스',
    '에스와이스틸텍',
    '유진스팩10호',
    '한국앤컴퍼니',
    '삼지전자'
    
 

    *** 오늘 52주 신고가를 갱신한 종목
    0           삼천당제약
    1            한양증권
    2            한국공항
    3            삼아제약
    4        HD한국조선해양
    5          HD현대미포
    6           LG이노텍
    7       한화에어로스페이스
    8            세명전기
    9            우리기술
    10           제룡전기
 

    *** 주가 대비 수익률 높은 종목 상위 10개
    태양금속우
    태양금속
    KBI메탈
    대상홀딩스우
    디티앤씨알오
    코아스
    포커스에이치엔에스
    래몽래인
    비케이홀딩스
    쎌바이오텍


    *** 오늘 서비스 업종 과열 종목
    NameK	StockCode	PER	
	큐알티	405100	84.89	
	와이즈버즈	273060	67.00	
	유투바이오	221800	56.75	
	우진엔텍	457550	47.23	
	하이브	352820	43.58	
	스튜디오미르	408900	42.25	
	스튜디오드래곤	253450	40.96	
	제이오	418550	37.75	
	현대오토에버	307950	35.23	
	드림씨아이에스	223250	26.28	

    ***오늘 제조 업종 과열 종목
    NameK	StockCode	PER	
	레이저옵텍	199550	4960.00	
	폰드그룹	472850	795.71	
	NPX	222160	536.00	
	엔젯	419080	368.00	
	파버나인	177830	295.00	
	마이크로디지탈	305090	276.84	
	지투파워	388050	264.05	
	동일금속	109860	252.22	
	미래생명자원	218150	231.30	
	한솔제지	213500	146.22	
    
    *** 오늘 IT 업종 과열 종목
  	NameK	StockCode	PER	
	티에스이	131290	4809.09	
	이노룰스	296640	1014.29	
	엑스게이트	356680	431.36	
	커넥트웨이브	119860	428.57	
	모아데이타	288980	275.29	

    *** 오늘 금융 업종 과열 종목
    NameK	StockCode	PER
	카카오페이	377300	1434.21	
	LS에코에너지	229640	223.90	
	나우IB	293580	49.81	
	SV인베스트먼트	289080	30.57	
	카카오뱅크	323410	29.13	

    *** 오늘 기계건설 업종 과열 종목
	NameK	StockCode	PER	
	씨에스윈드	112610	100.94	
	대명에너지	389260	39.81	
	SNT에너지	100840	9.96	
	금양그린파워	282720	7.97	
	조선선재	120030	7.45

    *** 오늘 운수 업종 과열 종목
    NameK	StockCode	PER	
	일진하이솔루스	271940	569.23	
	HD현대중공업	329180	546.76	
	STX그린로지스	465770	63.88	
	HL만도	204320	14.52	
	한진칼	180640	11.55	


    *** 오늘 유통 업종 과열 종목
    NameK	StockCode	PER	
	제로투세븐	159580	111.25	
	실리콘투	257720	70.21	
	미래반도체	254490	43.75	
	한국화장품	123690	39.72	
	뉴지랩파마	214870	23.07		
    
    *** 오늘 의료 업종 과열 종목
    NameK	StockCode	PER	
	SK바이오사이언스	302440	184.19	
	한컴라이프케어	372910	161.50	
	이연제약	102460	71.79	
	삼성바이오로직스	207940	65.72	
	경보제약	214390	60.00	


    *** 오늘 전기전자 업종 과열 종목
    NameK	StockCode	PER	
	에코프로머티	450080	1187.06	
	LG에너지솔루션	373220	67.81	
	HD현대일렉트릭	267260	44.58	
	대덕전자	353200	44.41
	드림텍	192650	42.42

    *** 오늘 철강및금속 업종 과열 종목
    NameK	StockCode	PER	
	동국씨엠	460850	19.97	
	풍산	103140	11.68	
	조선내화	462520	9.55	
	KCC글라스	344820	7.80	
	한일시멘트	300720	5.44	

    *** 오늘 화학 업종 과열 종목
    NameK	StockCode	PER	
	한국콜마	161890	270.48	
	토니모리	214420	73.74	
	잇츠한불	226320	71.66	
	효성첨단소재	298050	43.23	
	미원에스씨	268280	37.89	

    #Question: 
    {question} 
    #Answer:"""
        )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain =promptt| llm

    response = chain.invoke({"question": prompt})
    msg = response.content
    #AI 답변
    with st.chat_message("assistant"):
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role ="assistant",content=msg))