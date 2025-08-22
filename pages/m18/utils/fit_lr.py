import streamlit as st
from sklearn.linear_model import LinearRegression


@st.cache_data
def fit_lr(X, y) -> LinearRegression:
    return LinearRegression().fit(X, y)
