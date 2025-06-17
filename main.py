import streamlit as st
import numpy as np

st.write("=== X_clean info ===")
st.write(type(X_clean))
st.write(X_clean.dtypes)
st.write(X_clean.head())

st.write("=== y_clean info ===")
st.write(type(y_clean))
try:
    st.write(y_clean.dtype)
except:
    st.write("y_clean has no dtype")
try:
    st.write(y_clean.head())
except:
    st.write("y_clean has no head() method")
st.write(y_clean[:5] if hasattr(y_clean, '__getitem__') else y_clean)

st.write("=== NaN, inf 체크 ===")
st.write("X_clean isna any:", X_clean.isna().any().any())
st.write("y_clean isna any:", np.isnan(y_clean).any() if isinstance(y_clean, (np.ndarray, pd.Series)) else "Check manually")
st.write("X_clean inf any:", np.isinf(X_clean.values).any())
st.write("y_clean inf any:", np.isinf(y_clean).any() if isinstance(y_clean, (np.ndarray, pd.Series)) else "Check manually")

st.write("=== shape info ===")
st.write("X_clean shape:", X_clean.shape)
try:
    st.write("y_clean shape:", y_clean.shape)
except:
    st.write("y_clean has no shape")
