import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("서울시 상권 매출 예측 (유동인구 기반)")

@st.cache_data
def load_data():
    df_sales = pd.read_csv("seoul_sales_2024.csv", encoding='cp949')
    df_subway = pd.read_csv("seoul_subway_2024.csv", encoding='cp949')
    return df_sales, df_subway

df_sales, df_subway = load_data()

# --- 유동인구 데이터 전처리 ---
def preprocess_subway(df):
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['분기'] = df['날짜'].dt.to_period('Q').astype(str)

    # 시간대 그룹핑 (승차 + 하차)
    time_cols = [
        "06시 이전", "06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시",
        "11시-12시", "12시-13시", "13시-14시", "14시-15시", "15시-16시", "16시-17시",
        "17시-18시", "18시-19시", "19시-20시", "20시-21시", "21시-22시", "22시-23시", "23시-24시", "24시 이후"
    ]

    df_selected = df[df['구분'].isin(['승차', '하차'])].copy()

    # 시간대별 승차+하차 합산
    for col in time_cols:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
    df_selected_grouped = df_selected.groupby(['분기'])[time_cols].sum().reset_index()

    # 큰 시간대별 합산 예시 (선택적)
    time_groups = {
        "00~06": ["06시 이전", "24시 이후"],
        "06~11": ["06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시"],
        "11~14": ["11시-12시", "12시-13시", "13시-14시"],
        "14~17": ["14시-15시", "15시-16시", "16시-17시"],
        "17~21": ["17시-18시", "18시-19시", "19시-20시", "20시-21시"],
        "21~24": ["21시-22시", "22시-23시", "23시-24시"]
    }

    for grp, cols in time_groups.items():
        df_selected_grouped[f"시간대_{grp}"] = df_selected_grouped[cols].sum(axis=1)

    return df_selected_grouped[['분기'] + [f"시간대_{k}" for k in time_groups.keys()]]

# --- 매출 데이터 전처리 ---
def preprocess_sales(df):
    # 기준_년분기_코드 컬럼을 str 타입으로 맞춰줌
    df['기준_년분기_코드'] = df['기준_년분기_코드'].astype(str)
    # 매출 관련 컬럼만 평균값 계산 (시간대별 매출_금액 컬럼들)
    sales_cols = [col for col in df.columns if '시간대_' in col and '매출_금액' in col]
    df_grouped = df.groupby('기준_년분기_코드')[sales_cols].mean(numeric_only=True).reset_index()
    return df_grouped

df_traffic_quarter = preprocess_subway(df_subway)
df_sales_quarter = preprocess_sales(df_sales)

# 데이터 병합
df_merged = pd.merge(
    df_traffic_quarter,
    df_sales_quarter,
    left_on='분기',
    right_on='기준_년분기_코드'
)

# 모델 학습을 위한 데이터 준비
X_cols = [col for col in df_merged.columns if col.startswith('시간대_') and '매출_금액' not in col]
y_cols = [col for col in df_merged.columns if col.startswith('시간대_') and '매출_금액' in col]

X = df_merged[X_cols]
y = df_merged[y_cols].sum(axis=1)  # 시간대별 매출 합계로 타겟 설정

# 결측값 제거 및 데이터 타입 확인/변환
train_df = pd.concat([X, y], axis=1).dropna()
X_clean = train_df[X_cols].astype(float)
y_clean = train_df[y.name] if hasattr(y, 'name') else train_df.iloc[:, -1]
y_clean = y_clean.astype(float)

# 모델 학습
model = LinearRegression()
model.fit(X_clean, y_clean)

# 예측
preds = model.predict(X_clean)

# 시각화
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_df.index, y_clean, marker='o', label='실제 매출')
ax.plot(train_df.index, preds, marker='x', label='예측 매출')
ax.set_title('분기별 실제 vs 예측 매출')
ax.set_xlabel('데이터 인덱스')
ax.set_ylabel('총 매출')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 상관계수 출력
correlation = pd.Series(preds).corr(y_clean)
st.write(f"### 예측값과 실제 매출의 상관계수: `{correlation:.4f}`")
