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

# 유동인구 전처리 간단히
df_subway['날짜'] = pd.to_datetime(df_subway['날짜'])
df_subway['분기'] = df_subway['날짜'].dt.to_period('Q').astype(str)

# 필요한 시간대 합계 계산 (예시)
time_cols = [col for col in df_subway.columns if '시간대_' in col]
if not time_cols:
    st.error("유동인구 데이터에 시간대 컬럼이 없습니다.")
    st.stop()

df_subway_quarter = df_subway.groupby('분기')[time_cols].sum().reset_index()

# 매출 데이터 전처리
df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str)
sales_cols = [col for col in df_sales.columns if '시간대_' in col and '매출_금액' in col]
if not sales_cols:
    st.error("매출 데이터에 시간대 매출 컬럼이 없습니다.")
    st.stop()

df_sales_quarter = df_sales.groupby('기준_년분기_코드')[sales_cols].mean().reset_index()

# 병합
df_merge = pd.merge(df_subway_quarter, df_sales_quarter,
                    left_on='분기', right_on='기준_년분기_코드')

# 설명 변수와 목표 변수 선택
X = df_merge[time_cols]
y = df_merge[sales_cols].sum(axis=1)
y.name = 'total_sales'

# 결측치, 무한대 제거
mask = (~X.isna().any(axis=1)) & (~y.isna()) & (~np.isinf(X).any(axis=1)) & (~np.isinf(y))
X_clean = X.loc[mask]
y_clean = y.loc[mask].values.ravel()

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

# 상관계수
correlation = pd.Series(preds).corr(pd.Series(y_clean))
st.write(f"### 예측값과 실제 매출의 상관계수: `{correlation:.4f}`")
