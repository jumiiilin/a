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

# 유동인구 데이터 전처리
df_subway['날짜'] = pd.to_datetime(df_subway['날짜'])
df_subway['분기'] = df_subway['날짜'].dt.to_period('Q').astype(str)

time_groups = {
    "00~06": ["06시 이전"],
    "06~11": ["06시-07시", "07시-08시", "08시-09시", "09시-10시", "10시-11시"],
    "11~14": ["11시-12시", "12시-13시", "13시-14시"],
    "14~17": ["14시-15시", "15시-16시", "16시-17시"],
    "17~21": ["17시-18시", "18시-19시", "19시-20시", "20시-21시"],
    "21~24": ["21시-22시", "22시-23시", "23시-24시", "24시 이후"]
}

df_total = df_subway[df_subway['구분'].isin(['승차', '하차'])].copy()
for k, v in time_groups.items():
    df_total[f"시간대_{k}"] = df_total[v].sum(axis=1)

df_traffic_quarter = df_total.groupby('분기')[[f"시간대_{k}" for k in time_groups.keys()]].sum().reset_index()

# 매출 데이터 전처리
df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str)
sales_cols = [col for col in df_sales.columns if '시간대_' in col and '매출_금액' in col]
df_sales_quarter = df_sales.groupby('기준_년분기_코드')[sales_cols].mean().reset_index()

# 데이터 병합
df_merge = pd.merge(df_traffic_quarter, df_sales_quarter,
                    left_on='분기', right_on='기준_년분기_코드')

X = df_merge[[f"시간대_{k}" for k in time_groups.keys()]]
y = df_merge[sales_cols].sum(axis=1)
y.name = 'total_sales'

# 숫자형 변환 및 결측치/무한대 처리
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

mask = (~X.isna().any(axis=1)) & (~y.isna()) & (~np.isinf(X).any(axis=1)) & (~np.isinf(y))
X_clean = X.loc[mask]
y_clean = y.loc[mask].values.ravel()

# 상태 출력 (디버깅용)
st.write("X_clean shape:", X_clean.shape)
st.write("X_clean dtypes:", X_clean.dtypes)
st.write("X_clean sample:\n", X_clean.head())
st.write("y_clean shape:", y_clean.shape)
st.write("y_clean sample:", y_clean[:5])

# 모델 학습
model = LinearRegression()
model.fit(X_clean, y_clean)

# 예측
preds = model.predict(X_clean)

# 시각화
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_merge.loc[mask, '분기'], y_clean, marker='o', label='실제 매출')
ax.plot(df_merge.loc[mask, '분기'], preds, marker='x', label='예측 매출')
ax.set_title('분기별 실제 vs 예측 매출')
ax.set_xlabel('분기')
ax.set_ylabel('총 매출')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 상관계수 출력
correlation = pd.Series(preds).corr(pd.Series(y_clean))
st.write(f"### 예측값과 실제 매출의 상관계수: `{correlation:.4f}`")
