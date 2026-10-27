"""
Dashboard 1 — Bloons TD 6: quantidade de bloons por round
Layout do rascunho: Sidebar (filtros) | Correlação (heatmap) | Distribuição (histograma)
Gráficos em Altair; dados e cálculos com @st.cache_data.

Rodar:  streamlit run app.py
Deps:   pip install streamlit altair pandas numpy kagglehub
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from typing import Literal, cast

st.set_page_config(page_title="BTD6 — Bloons por Round", layout="wide")

COR_EASY = "#000A6F"   
COR_HARD = "#990F02"     # coral
COR_TODOS = "#7a8fa6"    # azul-acinzentado (sem separar por dificuldade)

GRUPOS = ["Easy", "Hard", "Todos"]
ESCALA_BARRAS = alt.Scale(domain=GRUPOS, range=[COR_EASY, COR_HARD, COR_TODOS])
ESCALA_LINHAS = alt.Scale(domain=GRUPOS, range=["#b9847c", COR_HARD, "gray"])
ESCALA_STATS = alt.Scale(domain=["Média", "Mediana"], range=["#d62728", "black"])
DASH_STATS = alt.Scale(domain=["Média", "Mediana"], range=[[6, 4], [1, 0]])

# Variáveis pré-selecionadas no heatmap (só entram se existirem no dataset)
DEFAULT_CORR = ["RBE", "Red", "Blue", "Green", "Pink", "Black", "White",
                "Zebra", "MOAB", "BFB", "ZOMG", "DDT"]


# ----------------------------------------------------------------------------
# Dados e cálculos (todos com cache)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dataset...")
def carregar_dados() -> pd.DataFrame:
    try:
        # opção 1: CSV baixado na mesma pasta do app.py
        df = pd.read_csv("BTD6_Rounds.csv")
    except FileNotFoundError:
        # opção 2: mesmo carregamento usado na EDA
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "brandonconrady/bloons-tower-defense-6-rounds-1140",
            "BTD6_Rounds.csv",
        )
    df["round_difficulty"] = np.where(df["Round"] > 40, "Hard", "Easy")
    return df


def filtrar_intervalo(intervalo: tuple) -> pd.DataFrame:
    """df_filtrado: ponto de partida de todos os gráficos."""
    df = carregar_dados()
    return df[df["Round"].between(*intervalo)]


@st.cache_data
def matriz_correlacao(intervalo: tuple, variaveis: tuple, metodo: str):
    """Correlação no intervalo, em formato longo (triângulo inferior) p/ o Altair."""
    sub = filtrar_intervalo(intervalo)[list(variaveis)]
    constantes = sub.columns[sub.nunique() <= 1].tolist()
    sub = sub.drop(columns=constantes)
    if sub.shape[1] < 2:
        return None, [], constantes

    corr = sub.corr(method=cast(Literal["pearson", "kendall", "spearman"], metodo))
    ordem = corr.columns.tolist()
    linhas = [(ordem[j], ordem[i], corr.iloc[i, j])
              for i in range(len(ordem)) for j in range(i)]
    corr_long = pd.DataFrame(linhas, columns=["x", "y", "corr"]).dropna()
    return corr_long, ordem, constantes


@st.cache_data
def preparar_histograma(intervalo: tuple, variavel: str, usar_log: bool,
                        por_dificuldade: bool) -> pd.DataFrame:
    df_f = filtrar_intervalo(intervalo)
    dados = pd.DataFrame({
        "valor": df_f[variavel],
        "grupo": df_f["round_difficulty"] if por_dificuldade else "Todos",
    })
    if usar_log:
        dados["valor"] = np.log1p(dados["valor"])
    return dados


# ----------------------------------------------------------------------------
# Gráficos (Altair)
# ----------------------------------------------------------------------------
def grafico_heatmap(
    corr_long: pd.DataFrame, ordem: list, metodo: str
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    n = len(ordem)
    lado = max(18, min(40, 420 // n))  # tamanho da célula, para caber qualquer nº de variáveis

    base = alt.Chart(corr_long).encode(
        x=alt.X("x:N", sort=ordem[:-1], title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("y:N", sort=ordem[1:], title=None),
    )
    celulas = base.mark_rect().encode(
        color=alt.Color(
            "corr:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            legend=alt.Legend(title=f"Correlação ({metodo})"),
        ),
        tooltip=[
            alt.Tooltip("x:N", title="Variável 1"),
            alt.Tooltip("y:N", title="Variável 2"),
            alt.Tooltip("corr:Q", title="Correlação", format=".2f"),
        ],
    )
    camadas = celulas
    if n <= 12:  # com muitas variáveis os números não cabem
        numeros = base.mark_text(fontSize=11).encode(
            text=alt.Text("corr:Q", format=".2f"),
            color=alt.condition("abs(datum.corr) > 0.6", alt.value("white"), alt.value("black")),
        )
        camadas = celulas + numeros
    return camadas.properties(width=alt.Step(lado), height=alt.Step(lado))


def grafico_histograma(dados: pd.DataFrame, xlabel: str, kde: bool) -> alt.LayerChart:
    passo = (dados["valor"].max() - dados["valor"].min()) / 30
    media, mediana = dados["valor"].mean(), dados["valor"].median()

    barras = alt.Chart(dados).mark_bar(opacity=0.75).encode(
        x=alt.X("valor:Q", bin=alt.Bin(step=passo), title=xlabel),
        y=alt.Y("count()", stack=None, title="Nº de rounds"),
        color=alt.Color("grupo:N", scale=ESCALA_BARRAS, legend=alt.Legend(title="Dificuldade")),
        tooltip=[
            alt.Tooltip("grupo:N", title="Grupo"),
            alt.Tooltip("valor:Q", bin=alt.Bin(step=passo), title=xlabel),
            alt.Tooltip("count()", title="Rounds"),
        ],
    )

    stats = pd.DataFrame({"estatistica": ["Média", "Mediana"], "valor": [media, mediana]})
    regras = alt.Chart(stats).mark_rule(strokeWidth=2).encode(
        x="valor:Q",
        color=alt.Color("estatistica:N", scale=ESCALA_STATS, legend=alt.Legend(title="Estatística")),
        strokeDash=alt.StrokeDash("estatistica:N", scale=DASH_STATS, legend=None),
        tooltip=[alt.Tooltip("estatistica:N", title=""),
                 alt.Tooltip("valor:Q", title="Valor", format=",.1f")],
    )

    camadas = [barras, regras]

    if kde:
        # só grupos com pelo menos 2 valores distintos (senão a densidade não existe)
        dados_kde = dados[dados.groupby("grupo")["valor"].transform("nunique") >= 2]
        if not dados_kde.empty:
            linha = (
                alt.Chart(dados_kde)
                .transform_density("valor", groupby=["grupo"], counts=True,
                                   as_=["valor", "densidade"])
                # densidade x largura da barra = contagem esperada por barra
                .transform_calculate(contagem=f"datum.densidade * {passo}")
                .mark_line(strokeWidth=2)
                .encode(
                    x="valor:Q",
                    y=alt.Y("contagem:Q", title="Nº de rounds"),
                    color=alt.Color("grupo:N", scale=ESCALA_LINHAS, legend=None),
                )
            )
            camadas.append(linha)

    return cast(
        alt.LayerChart,
        alt.layer(*camadas)
        .resolve_scale(color="independent")  # cores de grupo e de estatística separadas
        .properties(height=420),
    )


# ----------------------------------------------------------------------------
# SIDEBAR — filtros: variáveis + intervalo
# ----------------------------------------------------------------------------
df = carregar_dados()
VARS_QUANT = df.select_dtypes(include=np.number).columns.tolist()

st.sidebar.header("Filtros")

r_min, r_max = int(df["Round"].min()), int(df["Round"].max())
intervalo = st.sidebar.slider("Intervalo de rounds", r_min, r_max, (r_min, r_max))

st.sidebar.subheader("Correlação")
vars_corr = st.sidebar.multiselect(
    "Variáveis do heatmap",
    options=VARS_QUANT,
    default=[v for v in DEFAULT_CORR if v in VARS_QUANT] or VARS_QUANT[:8],
)
metodo = st.sidebar.selectbox(
    "Método", ["spearman", "pearson", "kendall"], index=0,
    help="Na EDA nenhuma variável passou no Shapiro-Wilk, então Spearman é o padrão.",
)

st.sidebar.subheader("Distribuição")
var_hist = st.sidebar.selectbox(
    "Variável do histograma", VARS_QUANT,
    index=VARS_QUANT.index("RBE") if "RBE" in VARS_QUANT else 0,
)
por_dificuldade = st.sidebar.checkbox("Separar por dificuldade (Easy x Hard)", value=True)
usar_log = st.sidebar.checkbox("Escala log(1 + x)", value=False,
                               help="Reduz a assimetria forte, como feito com o RBE na EDA.")
mostrar_kde = st.sidebar.checkbox("Mostrar curva KDE", value=True)

# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
n_filtrados = len(filtrar_intervalo(intervalo))
st.title("Bloons TD 6 — quantidade de bloons por round")
st.caption(f"{n_filtrados} rounds selecionados (round {intervalo[0]} a {intervalo[1]}) "
           f"de {len(df)} no dataset.")

if n_filtrados == 0:
    st.warning("Nenhum round no intervalo selecionado.")
    st.stop()

# ----------------------------------------------------------------------------
# Painéis: Correlação (heatmap) | Distribuição (histograma)
# ----------------------------------------------------------------------------
col_corr, col_dist = st.columns([1, 1.4])

# --- Correlação -------------------------------------------------------------
with col_corr:
    st.subheader("Correlação")

    if len(vars_corr) < 2:
        st.info("Selecione pelo menos 2 variáveis na sidebar.")
    else:
        corr_long, ordem, constantes = matriz_correlacao(intervalo, tuple(vars_corr), metodo)
        if constantes:
            st.caption("Sem variação no intervalo (removidas): " + ", ".join(constantes))
        if corr_long is None:
            st.info("Variáveis insuficientes com variação neste intervalo.")
        else:
            # heatmap com células quadradas: tamanho fixo ("content"), não esticado
            st.altair_chart(grafico_heatmap(corr_long, ordem, metodo), width="content")

# --- Distribuição -----------------------------------------------------------
with col_dist:
    st.subheader(f"Distribuição — {var_hist}")

    dados = preparar_histograma(intervalo, var_hist, usar_log, por_dificuldade)
    xlabel = f"log(1 + {var_hist})" if usar_log else var_hist

    if dados["valor"].nunique() < 2:
        st.info("A variável não varia neste intervalo de rounds.")
    else:
        st.altair_chart(grafico_histograma(dados, xlabel, mostrar_kde), width="stretch")

        assim = dados["valor"].skew()
        if pd.notna(assim):
            st.caption(f"Assimetria (skewness): {assim:.2f} — valores bem acima de 1 "
                       "indicam cauda longa à direita.")