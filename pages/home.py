import streamlit as st
from os import sep

st.set_page_config(
    page_title="Home"
)

st.title("Curso - Ciência de dados - EBAC")

st.html(
    """
    <div style="display: flex; align-items: flex-start; gap: 10px;">
        <img style="height: 100px" src="app/static/hi-otag.gif"/>
        <div>
            <p style="margin: 0;"><b>Olá!</b><br>Boas-vindas ao meu portfólio de projetos do curso de <b>Ciência de Dados</b> da EBAC!</b></p>
            <p style="margin: 0; text-align: justify"> Aqui você encontrará uma seleção dos trabalhos que desenvolvi ao longo do curso, abordando temas como <b>análise de dados</b>, <b>visualização</b>, <b>estatística aplicada</b> e <b><i>machine learning</i></b>.</p>
        </div>
    </div>
    """
)

st.markdown(
    ":point_left: Utilize o painel lateral à esquerda para visualizar os projetos realizados.")
