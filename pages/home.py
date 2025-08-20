import streamlit as st
from os import sep

st.set_page_config(
    page_title="Home",
)

st.title("Curso - Ciência de dados - EBAC")

st.html(
    """
    <p>Boas-vindas ao meu portfólio de projetos do curso de Ciência de Dados da EBAC!</p>
    <div style="display: flex; align-items: center; gap: 10px;">
        <img style="height: 100px" src="app/static/hi-otag.gif"/>
        <p style="margin: 0; text-align: justify"> Aqui você encontrará uma seleção dos trabalhos que desenvolvi ao longo do curso, abordando temas como análise de dados, visualização, estatística aplicada e <i>machine learning</i>.</p>
    </div>
    """
)

st.markdown(
    ":point_left: Utilize o painel lateral à esquerda para visualizar os projetos realizados.")
