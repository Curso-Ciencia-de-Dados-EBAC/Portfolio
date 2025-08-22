import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from os import sep

from pages.m18.charts import *
from pages.m18.fragments import plot_simple_lr, plot_multiple_lr, fit_lr, root_mean_squared_error


### CUSTOM STYLES ###
st.html(
    """
    <style>
        .appview-container .stMainBlockContainer {
            max-width: 75%;
        }
    </style>
    """
)
### -------------- ###


@st.cache_resource
def fit_lr(X, y) -> LinearRegression:
    return LinearRegression().fit(X, y)


st.title("Regressão Linear")
"""
Este projeto teve como objetivo aplicar um modelo de regressão linear a um conjunto de dados de imóveis, visando prever o valor do aluguel a partir de diferentes características do imóvel, como metragem, número de quartos, banheiros e valor do condomínio. A análise inclui etapas de pré-processamento, exploração dos dados e avaliação do desempenho do modelo.
"""
"---"
st.subheader("Carregamento dos dados e pré-processamento")
with st.echo():
    df = pd.read_csv(f"data{sep}ALUGUEL_MOD12.csv", delimiter=';')
    # st.write(df.head(5))

"""
Após o carregamento dos dados, iniciei a análise exploratória para preparar o conjunto para o modelo de regressão.
O primeiro passo foi verificar os tipos de dados e identificar possíveis valores ausentes.
"""

with st.echo():
    info = pd.DataFrame({
        "Tipo de dado": df.dtypes.astype(str),
        "Quantidade de nulos": df.isnull().sum()
    })

st.dataframe(info, use_container_width=False)

"A inferência correta dos tipos de dados para cada coluna implica que não há valores errados."
"Para identificar outliers, analisei a distribuição das variáveis independentes com boxplots."


st.pyplot(plot_boxplots(df))

r"""
Os *boxplots* acima revelam que todas as variáveis apresentam distribuições assimétricas à direita, indicando a presença de *outliers* acima da média. Para as variáveis contínuas (*valor do aluguel*, *valor do condomínio* e *metragem*), observa-se uma quantidade significativa de valores superiores a 3 desvios-padrão em relação à média. Já as variáveis discretas possuem uma amplitude de valores mais restrita.

Diante desse cenário, optei por tratar apenas os *outliers* das variáveis contínuas, removendo os valores cujo *z-score* excede 3. Essa abordagem visa preservar possíveis casos extremos relevantes para a análise. Antes da remoção dos *outliers*, apliquei a transformação logarítmica ($\log_{10}$) nas variáveis contínuas, a fim de padronizar suas escalas e reduzir o impacto de valores extremos.
"""


"> Transformação Log"
with st.echo():
    tdf = df.copy()
    tdf[CONTINUOUS_VARIABLES] = (tdf[CONTINUOUS_VARIABLES]
                                 .mask(lambda x: x == 0, pd.NA)
                                 .dropna()
                                 .apply(np.log10)
                                 .mask(lambda x: x == 0, pd.NA)
                                 .dropna())

    tdf.dropna(inplace=True)

"> Remoção de _outliers_"
with st.echo():
    for c in CONTINUOUS_VARIABLES:
        q1, q3 = tdf[c].quantile([0.25, 0.75])
        z_score = (tdf[c] - tdf[c].mean()) / tdf[c].std()
        tdf[c] = tdf[c].mask(z_score > 3, pd.NA)
    tdf = tdf.dropna()

"> Criação de histogramas"
st.pyplot(plot_histograms(tdf))

r"""
Nota-se que, após a transformação logarítmica e a remoção dos *outliers*, as distribuições das variáveis contínuas tornaram-se mais simétricas e a presença de valores extremos foi consideravelmente reduzida, tornando os dados mais adequados para a modelagem.\
Além disso, os histogramas sugerem que as variáveis contínuas seguem uma distribuição semelhante à Gama. Nos gráficos acima, os parâmetros $\alpha$ e $\beta$ foram estimados por $\hat{\alpha} = \frac{\hat{\mu}^2}{\hat{\sigma}^2}$ e $\hat{\beta} = \frac{\hat{\sigma}^2}{\hat{\mu}}$, respectivamente.
"""

"A seguir, foi analisada a matriz de correlação para investigar o relacionamento entre as variáveis do conjunto de dados."

st.pyplot(plot_corr_matrix(tdf), use_container_width=False)

"""
Observa-se que todas as variáveis apresentam correlação positiva entre si, geralmente acima de 0,5, o que é consistente com a intuição. Por exemplo, espera-se que a metragem do imóvel esteja relacionada ao número de quartos e banheiros. No entanto, essa colinearidade pode impactar negativamente a precisão dos coeficientes estimados pelo modelo.\\
Além disso, nota-se que o valor do aluguel possui correlação significativa com as demais variáveis, exceto pelo número de quartos, que apresenta uma relação mais fraca.
"""

"Após o pré-processamento, os dados foram divididos em treino e teste para ajuste e avaliação do modelo."

with st.echo():
    X = tdf.drop('Valor_Aluguel', axis=1)
    y = tdf['Valor_Aluguel']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

f"Quantidade de dados de treino: {X_train.shape[0]}"
f"Quantidade de dados de teste: {X_test.shape[0]}"

plot_simple_lr(X_train, X_test, y_train, y_test)
plot_multiple_lr(X_train, X_test, y_train, y_test)

"As métricas do modelo ajustado com todas as variáveis são apresentadas abaixo."
lr = fit_lr(X_train, y_train)
y_pred = lr.predict(X_test)

st.dataframe(pd.DataFrame({
    "R²": [lr.score(X_train, y_train), lr.score(X_test, y_test)],
    "MSE": [root_mean_squared_error(y_train, lr.predict(X_train)), root_mean_squared_error(y_test, y_pred)]
}, index=["Treino", "Teste"]), use_container_width=False)

r"""
No geral, os resultados obtidos mostram que os modelos apresentaram desempenho semelhante nos conjuntos de treino e teste, sugerindo ausência de _overfitting_. No entanto, os valores de R², que ficaram abaixo de 0,7, indicam que a relação entre as variáveis não é totalmente capturada por um modelo linear. Isso evidencia tanto o potencial quanto as limitações da regressão linear para este conjunto de dados, reforçando a importância de considerar abordagens mais complexas ou a inclusão de novas variáveis para melhorar o ajuste do modelo.
"""
