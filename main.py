import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

st.title("서울시 상권 매출 예측 (유동인구 기반)")

@st.cache_data
def load_data():
    df_sales = pd.read_csv("seoul_sales_2024.csv", encoding='cp949')
    df_subway = pd.read_csv("seoul_subway_2024.csv", encoding='cp949')
    return df_sales, df_subway

df_sales, df_subway = load_data()

# --- 유동인구 데이터 전처리 ---
def preprocess_subway(df):
    # 날짜에서 분기 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['분기'] = df['날짜'].dt.to_period('Q')

    # 시간대 통합
    time_groups = {
        "00~06": ["06시 이전"],
        "06~11": ["06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시"],
        "11~14": ["11시-12시", "12시-13시", "13시-14시"],
        "14~17": ["14시-15시", "15시-16시", "16시-17시"],
        "17~21": ["17시-18시", "18시-19시", "19시-20시", "20시-21시"],
        "21~24": ["21시-22시", "22시-23시", "23시-24시"]
    }

    df_total = df[df['구분'].isin(['승차', '하차'])].copy()
    for k, v in time_groups.items():
        df_total[f"시간대_{k}"] = df_total[v].sum(axis=1)

    df_total_grouped = df_total.groupby('분기')[
        [f"시간대_{k}" for k in time_groups.keys()]
    ].sum().reset_index()
    return df_total_grouped

# --- 매출 데이터 전처리 ---
def preprocess_sales(df):
    df_grouped = df.groupby('기준_년분기_코드')[
        [col for col in df.columns if '시간대_' in col and '매출_금액' in col]
    ].mean(numeric_only=True).reset_index()
    return df_grouped

# --- 전처리 실행 ---
df_traffic_quarter = preprocess_subway(df_subway)
df_sales_quarter = preprocess_sales(df_sales)

# 분기 이름 정리
quarter_map = {
    "2024Q1": "2024년 1분기",
    "2024Q2": "2024년 2분기",
    "2024Q3": "2024년 3분기",
    "2024Q4": "2024년 4분기"
}
df_traffic_quarter['분기'] = df_traffic_quarter['분기'].astype(str)
df_sales_quarter['기준_년분기_코드'] = df_sales_quarter['기준_년분기_코드'].astype(str)

# --- 데이터 병합 ---
df_merge = pd.merge(
    df_traffic_quarter,
    df_sales_quarter,
    left_on='분기',
    right_on='기준_년분기_코드'
)

# --- 모델 학습 ---
X = df_merge[[col for col in df_merge.columns if '시간대_' in col and '_매출_금액' not in col]]
y = df_merge[[col for col in df_merge.columns if '시간대_' in col and '_매출_금액' in col]].sum(axis=1)  # 전체 매출합

model = LinearRegression()
# --- 모델 학습 ---
X = df_merge[[col for col in df_merge.columns if '시간대_' in col and '_매출_금액' not in col]]
y = df_merge[[col for col in df_merge.columns if '시간대_' in col and '_매출_금액' in col]].sum(axis=1)

# 결측값 제거
train_df = pd.concat([X, y], axis=1).dropna()
X = train_df[X.columns]
y = train_df[y.name] if hasattr(y, 'name') else train_df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

# --- 예측 ---
preds = model.predict(X)

# --- 시각화 ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_merge['분기'], y, marker='o', label='실제 매출')
ax.plot(df_merge['분기'], preds, marker='x', label='예측 매출')
ax.set_title('분기별 실제 vs 예측 매출')
ax.set_xlabel('분기')
ax.set_ylabel('총 매출')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- 상관계수 출력 ---
correlation = pd.Series(preds).corr(y)
st.write(f"### 예측값과 실제 매출의 상관계수: `{correlation:.4f}`")
