import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# 디버깅: 파일 경로 확인
st.write("현재 작업 디렉토리:", os.getcwd())
st.write("seoul_sales_2024.csv 존재 여부:", os.path.exists("seoul_sales_2024.csv"))
st.write("seoul_subway_2024.csv 존재 여부:", os.path.exists("seoul_subway_2024.csv"))

st.title("서울시 상권 vs 유동인구 분석 + 머신러닝 예측")

@st.cache_data
def load_data():
    df_sales = pd.read_csv("seoul_sales_2024.csv", encoding='cp949')
    df_subway = pd.read_csv("seoul_subway_2024.csv", encoding='cp949')
    return df_sales, df_subway

df_sales, df_subway = load_data()

st.subheader("1. 데이터 샘플")
st.write("📊 df_sales")
st.write(df_sales.head())

st.write("🚇 df_subway")
st.write(df_subway.head())

# 시간대 그룹핑 함수
def group_subway_timezones(df_subway):
    subway_time_columns = {
        "시간대_00~06": ["06시 이전"],
        "시간대_06~11": ["06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시"],
        "시간대_11~14": ["11시-12시", "12시-13시", "13시-14시"],
        "시간대_14~17": ["14시-15시", "15시-16시", "16시-17시"],
        "시간대_17~21": ["17시-18시", "18시-19시", "19시-20시", "20시-21시"],
        "시간대_21~24": ["21시-22시", "22시-23시", "23시-24시"],
    }

    df_grouped = df_subway.copy()
    for new_col, time_range in subway_time_columns.items():
        df_grouped[new_col] = df_grouped[time_range].sum(axis=1)

    grouped_cols = ["날짜", "역명", "구분"] + list(subway_time_columns.keys())
    return df_grouped[grouped_cols]

# 승하차 → 유동인구 합계
def calculate_total_traffic(df_grouped):
    df_board = df_grouped[df_grouped["구분"] == "승차"]
    df_alight = df_grouped[df_grouped["구분"] == "하차"]

    df_merged = pd.merge(
        df_board, df_alight, on=["날짜", "역명"], suffixes=("_승차", "_하차")
    )

    traffic_cols = ["시간대_00~06", "시간대_06~11", "시간대_11~14",
                    "시간대_14~17", "시간대_17~21", "시간대_21~24"]

    for col in traffic_cols:
        df_merged[col + "_유동인구"] = df_merged[col + "_승차"] + df_merged[col + "_하차"]

    result_cols = ["날짜", "역명"] + [col + "_유동인구" for col in traffic_cols]
    return df_merged[result_cols]

# 2. 유동인구 계산
df_grouped = group_subway_timezones(df_subway)
df_traffic = calculate_total_traffic(df_grouped)

st.subheader("2. 동대문 지역 필터링")
dongdaemun_sales = df_sales[df_sales['상권_코드_명'].str.contains("동대문", na=False)]
dongdaemun_traffic = df_traffic[df_traffic["역명"] == "동대문"]

st.write("🏪 매출 데이터")
st.write(dongdaemun_sales.head())

st.write("🚶‍♂️ 유동인구 데이터")
st.write(dongdaemun_traffic.head())

# 3. 평균 계산
dongdaemun_traffic_avg = dongdaemun_traffic.mean(numeric_only=True)
dongdaemun_sales_avg = dongdaemun_sales[
    [col for col in dongdaemun_sales.columns if "시간대_" in col and "매출_금액" in col]
].mean()

st.subheader("3. 평균 계산")
st.write("🟦 유동인구 평균")
st.write(dongdaemun_traffic_avg)

st.write("🟧 매출 평균")
st.write(dongdaemun_sales_avg)

# 4. 상관계수 및 시각화 준비
avg_df = pd.DataFrame({
    '시간대': dongdaemun_sales_avg.index.str.replace("_매출_금액", "", regex=False),
    '평균_매출': dongdaemun_sales_avg.values,
    '평균_유동인구': dongdaemun_traffic_avg.values
})

correlation = avg_df['평균_매출'].corr(avg_df['평균_유동인구'])
st.subheader("4. 시간대별 상관관계")
st.write(f"📈 상관계수: `{correlation:.4f}`")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_df['시간대'], avg_df['평균_매출'], marker='o', label='평균 매출')
ax.plot(avg_df['시간대'], avg_df['평균_유동인구'], marker='s', label='평균 유동인구')
ax.set_title("동대문 시간대별 매출 vs 유동인구")
ax.set_xlabel("시간대")
ax.set_ylabel("값")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 5. 머신러닝 분석
st.subheader("5. 머신러닝: 유동인구로 매출 예측")

X = avg_df[['평균_유동인구']].values
y = avg_df['평균_매출'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.write(f"🔍 선형 회귀 R² 점수: `{r2_score(y, y_pred):.4f}`")
st.write(f"🧮 평균 제곱 오차 (MSE): `{mean_squared_error(y, y_pred):,.0f}`")

fig2, ax2 = plt.subplots()
ax2.scatter(X, y, color='blue', label='실제 매출')
ax2.plot(X, y_pred, color='red', label='예측 선형 회귀선')
ax2.set_xlabel("평균 유동인구")
ax2.set_ylabel("평균 매출")
ax2.set_title("유동인구 → 매출 선형 회귀")
ax2.legend()
st.pyplot(fig2)
