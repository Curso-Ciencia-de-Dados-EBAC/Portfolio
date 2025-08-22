import streamlit as st
from os import sep

### ROUTING ###
pg = st.navigation([
    st.Page(f"pages{sep}home.py", url_path="/", title="Home"),
    st.Page(f"pages{sep}m18{sep}main.py", url_path="linear_regression",
            title="Módulo 18 - Regressão Linear")
])

pg.run()
