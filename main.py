import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

st.title("서울시 시간대별 유동인구 기반 매출 예측 (머신러닝)")

# 파일 존재 여부 확인
st.write("현재 작업 디렉토리:", os.getcwd())
st.write("seoul_sales_2024.csv 존재 여부:", os.path.exists("seoul_sales_2024.csv"))
st.write("seoul_subway_2024.csv 존재 여부:", os.path.exists("seoul_subway_2024.csv"))

@st.cache_data
def load_data():
    df_sales = pd.read_csv("seoul_sales_2024.csv", encoding="cp949")
    df_subway = pd.read_csv("seoul_subway_2024.csv", encoding="cp949")
    return df_sales, df_subway

df_sales, df_subway = load_data()

# 지하철 시간대별 그룹화
def group_subway_timezones(df):
    timezones = {
        "시간대_00~06": ["06시 이전"],
        "시간대_06~11": ["06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시"],
        "시간대_11~14": ["11시-12시", "12시-13시", "13시-14시"],
        "시간대_14~17": ["14시-15시", "15시-16시", "16시-17시"],
        "시간대_17~21": ["17시-18시", "18시-19시", "19시-20시", "20시-21시"],
        "시간대_21~24": ["21시-22시", "22시-23시", "23시-24시"],
    }
    df_grouped = df.copy()
    for new_col, cols in timezones.items():
        df_grouped[new_col] = df_grouped[cols].sum(axis=1)
    return df_grouped[["날짜", "역명", "구분"] + list(timezones.keys())]

df_subway_grouped = group_subway_timezones(df_subway)

# 승하차 데이터를 합산해서 유동인구 계산
def calculate_total_traffic(df):
    board = df[df["구분"] == "승차"]
    alight = df[df["구분"] == "하차"]
    merged = pd.merge(board, alight, on=["날짜", "역명"], suffixes=("_승차", "_하차"))
    timezones = ["시간대_00~06", "시간대_06~11", "시간대_11~14",
                 "시간대_14~17", "시간대_17~21", "시간대_21~24"]
    for tz in timezones:
        merged[tz + "_유동인구"] = merged[tz + "_승차"] + merged[tz + "_하차"]
    return merged[["날짜", "역명"] + [tz + "_유동인구" for tz in timezones]]

df_traffic = calculate_total_traffic(df_subway_grouped)

# 날짜 기준 평균 유동인구 생성
df_traffic_avg = df_traffic.groupby("날짜").mean(numeric_only=True).reset_index()

# 날짜 기준 매출 평균
sales_cols = [col for col in df_sales.columns if "시간대_" in col and "매출_금액" in col]
df_sales_avg = df_sales.groupby("기준_일자")[sales_cols].mean(numeric_only=True).reset_index()

# 날짜 컬럼 정리
df_sales_avg.rename(columns={"기준_일자": "날짜"}, inplace=True)

# 유동인구와 매출 데이터 병합
df_merge = pd.merge(df_traffic_avg, df_sales_avg, on="날짜", how="inner")

st.write("### 병합된 데이터 샘플")
st.write(df_merge.head())

# 시간대 선택
timezones = ["시간대_00~06", "시간대_06~11", "시간대_11~14",
             "시간대_14~17", "시간대_17~21", "시간대_21~24"]
selected = st.selectbox("예측할 시간대 선택", timezones)

X = df_merge[[selected + "_유동인구"]]
y = df_merge[selected + "_매출_금액"]

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 평가 지표
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

st.write(f"### 예측 성능 ({selected})")
st.write(f"R² Score: `{r2:.4f}`")
st.write(f"MSE: `{mse:.2f}`")

# 시각화
fig, ax = plt.subplots()
ax.scatter(X, y, label="실제 매출", alpha=0.6)
ax.plot(X, y_pred, color="red", label="예측 매출 선형회귀")
ax.set_xlabel("유동인구 수")
ax.set_ylabel("매출 금액")
ax.set_title(f"{selected} 유동인구 기반 매출 예측")
ax.legend()
ax.grid(True)

st.pyplot(fig)


