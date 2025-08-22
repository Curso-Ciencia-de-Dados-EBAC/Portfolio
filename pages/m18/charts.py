import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gamma
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from os import sep

CONTINUOUS_VARIABLES = ["Valor_Aluguel", "Valor_Condominio", "Metragem"]


@st.cache_data(show_spinner="Preparando visualização...")
def plot_boxplots(df: pd.DataFrame) -> Figure:
    with st.echo():
        fig, axes = plt.subplots(
            ncols=2, nrows=4, figsize=(15, 10), layout="tight")
        fig.delaxes(axes[-1][-1])

        for col, ax in zip(df.columns, axes.flatten()):
            sns.boxplot(df[col], orient="h", ax=ax)

            if col in CONTINUOUS_VARIABLES:
                z_threshold = df[col].mean() + 3*(df[col].std())
                ax.axvline(x=z_threshold, ymin=0, ymax=1,
                           linestyle="--", color="black")
                ax.annotate(xy=(z_threshold + 0.025*z_threshold, -0.25),
                            text=r"$\mu + 3\sigma$")

    return fig


@st.cache_data(show_spinner="Preparando visualização...")
def plot_histograms(df: pd.DataFrame) -> Figure:
    with st.echo():
        fig, axes = plt.subplots(
            ncols=2, nrows=4, figsize=(15, 10), layout="tight")
        fig.delaxes(axes[-1][-1])

        for col, ax in zip(df.columns, axes.flatten()):
            sns.histplot(df[col], stat="density", ax=ax, legend=None)

            if col in CONTINUOUS_VARIABLES:
                alpha_hat = (df[col].mean()**2)/df[col].var()
                beta_hat = df[col].var()/df[col].mean()
                X = df[col].sort_values(ascending=True).values
                gamma_dist = [
                    gamma.pdf(x, a=alpha_hat, scale=beta_hat) for x in X]
                pd.Series(gamma_dist, index=X).plot.area(ax=ax, color="orange",
                                                         alpha=0.4, linestyle="--",
                                                         label=rf"$X \sim Gamma({alpha_hat:.2f}, {beta_hat:.2f})$")
                ax.legend(loc="upper right")

    return fig


@st.cache_data(show_spinner="Preparando visualização...")
def plot_corr_matrix(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    corr_matrix = df.corr(method="pearson")

    sns.heatmap(corr_matrix, fmt="s", annot=corr_matrix.map(lambda x: f"{x:.2f}" if abs(x) > 0.5 else ""),
                cmap="coolwarm", square=True, vmax=1, vmin=-1, annot_kws={"fontsize": 8}, ax=ax)

    return fig


@st.cache_data(show_spinner=False)
def plot_regression_line(X: pd.DataFrame, y: pd.Series, _lr: LinearRegression, colname: str) -> Figure:
    fig, ax = plt.subplots(layout="tight", figsize=(6, 6))

    X_1D = X.flatten()
    regression_line = _lr.intercept_ + _lr.coef_[0] * X_1D

    sns.scatterplot(x=X_1D, y=y, size=X_1D, alpha=0.4, legend=None, ax=ax)
    sns.lineplot(x=X_1D, y=regression_line, color="red",
                 ax=ax, label=r"$\hat{y}="f"({_lr.intercept_:.2f})+({_lr.coef_[0]:.2f})x$")

    plt.xlabel(colname)
    plt.ylabel("Valor do Aluguel")
    plt.title(f"{colname} X Valor do Aluguel")
    ax.annotate(f"$R^2={_lr.score(X, y):.3f}$ | $RMSE = {root_mean_squared_error(y, _lr.predict(X)):.3f}$",
                xy=(0.025, 0.875), xycoords="axes fraction", ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#00000000"))

    ax.legend(loc="upper left")

    return fig


@st.cache_data(show_spinner=False)
def plot_regression_plane(X: pd.DataFrame, y: pd.Series, _lr: LinearRegression, colname: str) -> go.Figure:

    def regression_plane(x, y): return _lr.intercept_ + [x, y] @ _lr.coef_
    _x, _y = X.iloc[:, 0], X.iloc[:, 1]

    _xx, _yy = np.array([_x.min(), _x.max()]), np.array([_y.min(), _y.max()])
    z = np.array([[regression_plane(x, y) for y in _yy] for x in _xx])
    fig = go.Figure(data=[
        go.Surface(
            x=_xx,
            y=_yy,
            z=z,
            hoverinfo="x+y+z",
            xhoverformat=".3f",
            yhoverformat=".3f",
            zhoverformat=".3f"
        )
    ])

    fig.update_traces(showscale=False, colorscale=[
                      [0, 'rgb(200,0,0)'], [1, 'rgb(200,0,0)']])

    fig.add_scatter3d(mode="markers", x=_x, y=_y, z=y, marker=dict(
        size=2, opacity=0.85, color=y), hoverinfo="x+y+z",
        xhoverformat=".3f", yhoverformat=".3f", zhoverformat=".3f")

    fig.update_layout(
        autosize=False,
        width=500, height=500,
        margin=dict(l=10, r=10, b=10, t=25)
    )

    fig.update_scenes(
        xaxis_title=colname[0],
        yaxis_title=colname[1],
        zaxis_title="Valor_Aluguel"
    )

    return fig
