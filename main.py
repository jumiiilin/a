import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("서울시 상권 매출 예측 (유동인구 기반)")

@st.cache_data
def load_data():
    df_sales = pd.read_csv("seoul_sales_2024.csv", encoding='cp949')
    df_subway = pd.read_csv("seoul_subway_2024.csv", encoding='cp949')
    return df_sales, df_subway

df_sales, df_subway = load_data()

# 유동인구 데이터 컬럼명 출력 (확인용)
st.write("=== df_subway 컬럼명 ===")
st.write(df_subway.columns.tolist())

# 아래부터 전처리, 모델 등 작업 진행
# 시간대 컬럼명 만들기 등은 df_subway 컬럼명 확인 후 다음 작업 진행해주세요
