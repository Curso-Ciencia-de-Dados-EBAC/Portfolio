import streamlit as st
from os import sep

pg = st.navigation([
    st.Page(f"pages{sep}home.py", url_path="/", title="Home"),
    st.Page(f"pages{sep}m18.py", url_path="linear_regression",
            title="Regressão Linear - Módulo 18")
])

pg.run()
