import streamlit as st
from threading import RLock

from pages.m18.utils.fit_lr import fit_lr
from pages.m18.charts import *

_lock = RLock()

st.subheader("Regressão linear simples")


@st.fragment
def plot_simple_lr(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    st.subheader("Regressão linear simples")

    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="left"):
        st.markdown(
            """
            Primeiro, vamos avaliar um modelo de regressão linear simples. Você pode escolher
            a variável independente no campo ao lado.
            """
        )

        opt = st.selectbox("X", X_train.columns, label_visibility="collapsed",
                           placeholder="Variável independente")

    if opt:
        f"> `opt = {opt}`"
        with st.echo():
            _X_train, _X_test = (
                # Transformação dos dados
                X_train[opt].values.reshape(-1, 1), X_test[opt].values.reshape(-1, 1))
            simple_lr = fit_lr(_X_train, y_train)  # Ajuste do modelo

        with st.status("Preparando visualização com os dados de treino...", expanded=True) as status:
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_regression_line(_X_train, y_train, simple_lr, opt)
                fig.suptitle("Dados de treinamento", fontweight="bold")
                st.pyplot(fig, clear_figure=True)
                status.update(
                    label="Preparando visualização com os dados de teste...", state="running")

            with col2:
                fig = plot_regression_line(_X_test, y_test, simple_lr, opt)
                fig.suptitle("Dados de teste", fontweight="bold")
                st.pyplot(fig, clear_figure=True)

            status.update(label="Visualizações prontas!", state="complete")


@st.fragment
def plot_multiple_lr(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    st.subheader("Regressão linear múltipla")
    st.markdown(
        """
        Agora, vamos avaliar um modelo de regressão linear múltipla.\\
        Selecione duas variáveis independentes nos campos abaixo. Para mais de duas variáveis, a visualização do hiperplano resultante se torna inviável.
        """
    )

    rcol1, rcol2 = st.columns(2, border=True)
    x1, x2 = None, None
    with rcol1:
        x1 = st.selectbox(
            label="$X_1$", options=X_train.columns, key="x1", index=None, placeholder="Escolha uma variável")
    with rcol2:
        x2 = st.selectbox(
            label="$X_2$", options=X_train.columns, key="x2", index=None, placeholder="Escolha uma variável")

    if x1 is not None and x2 is not None:
        if x1 == x2:
            st.error("As variáveis preditoras não podem ser iguais!")
            return

        with st.echo():
            _X_train, _X_test = X_train[[x1, x2]], X_test[[x1, x2]]
            mult_lr = fit_lr(X_train[[x1, x2]], y_train)  # Ajuste do modelo

        with st.status("Preparando visualização com os dados de treino...", expanded=True) as status:
            col1, col2 = st.columns(2, border=True)
            with col1:
                fig = plot_regression_plane(
                    _X_train, y_train, mult_lr, [x1, x2])
                st.write(
                    fr"""
                    $R^2 = {mult_lr.score(_X_train, y_train):.3f}$ | $RMSE = {root_mean_squared_error(y_train, mult_lr.predict(_X_train)):.3f}$\
                    $z = {mult_lr.intercept_:.2f} + ({mult_lr.coef_[0]:.2f})x + ({mult_lr.coef_[1]:.2f})y$
                    """
                )
                st.plotly_chart(fig)
                status.update(
                    label="Preparando visualização com os dados de teste...", state="running")

            with col2:
                fig = plot_regression_plane(_X_test, y_test, mult_lr, [x1, x2])
                st.write(
                    fr"""
                    $R^2 = {mult_lr.score(_X_test, y_test):.3f}$ | $RMSE = {root_mean_squared_error(y_test, mult_lr.predict(_X_test)):.3f}$\
                    $z = {mult_lr.intercept_:.2f} + ({mult_lr.coef_[0]:.2f})x + ({mult_lr.coef_[1]:.2f})y$
                    """
                )
                st.plotly_chart(fig)

            status.update(label="Visualizações prontas!", state="complete")
