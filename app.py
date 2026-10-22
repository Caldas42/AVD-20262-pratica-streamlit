"""
Dashboard — Bloons TD 6: quantidade de bloons por round
Rascunho seguindo o wireframe: Sidebar (filtros) | Correlação (heatmap) | Distribuição (histograma)

Rodar:  streamlit run app.py
Deps:   pip install streamlit pandas numpy seaborn matplotlib kagglehub
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Config e paleta (mesma convenção de cores da EDA)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BTD6 — Bloons por Round", layout="wide")
sns.set_theme(style="whitegrid")

COR_NEUTRA = "lightgray"
COR_CONTEXTO = "#e9d4d0"   # Easy
COR_DESTAQUE = "#2d223a"   # Hard
PALETA = {"Easy": COR_CONTEXTO, "Hard": COR_DESTAQUE}

# Variáveis pré-selecionadas no heatmap (só entram se existirem no dataset)
DEFAULT_CORR = ["RBE", "Red", "Blue", "Green", "Pink", "Black", "White",
                "Zebra", "MOAB", "BFB", "ZOMG", "DDT"]


# ----------------------------------------------------------------------------
# Dados
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


df = carregar_dados()
VARS_QUANT = df.select_dtypes(include=np.number).columns.tolist()

# ----------------------------------------------------------------------------
# SIDEBAR — filtros: variáveis + intervalo
# ----------------------------------------------------------------------------
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

# aplica o filtro de intervalo
df_f = df[df["Round"].between(*intervalo)]

# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
st.title("Bloons TD 6 — quantidade de bloons por round")
st.caption(f"{len(df_f)} rounds selecionados (round {intervalo[0]} a {intervalo[1]}) "
           f"de {len(df)} no dataset.")

if df_f.empty:
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
        sub = df_f[vars_corr]
        constantes = sub.columns[sub.nunique() <= 1].tolist()
        if constantes:
            st.caption("Sem variação no intervalo (removidas): " + ", ".join(constantes))
            sub = sub.drop(columns=constantes)

        if sub.shape[1] < 2:
            st.info("Variáveis insuficientes com variação neste intervalo.")
        else:
            corr = sub.corr(method=metodo)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            n = corr.shape[0]
            lado = max(4.5, 0.55 * n)

            fig, ax = plt.subplots(figsize=(lado, lado))
            sns.heatmap(
                corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, square=True,
                annot=n <= 12, fmt=".2f", annot_kws={"size": 8},
                cbar_kws={"shrink": 0.7, "label": f"Correlação ({metodo})"}, ax=ax,
            )
            ax.grid(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# --- Distribuição -----------------------------------------------------------
with col_dist:
    st.subheader(f"Distribuição — {var_hist}")

    dados = df_f[["round_difficulty", var_hist]].copy()
    xlabel = var_hist
    if usar_log:
        dados[var_hist] = np.log1p(dados[var_hist])
        xlabel = f"log(1 + {var_hist})"

    fig, ax = plt.subplots(figsize=(8, 5))
    if por_dificuldade:
        sns.histplot(
            data=dados, x=var_hist, hue="round_difficulty", palette=PALETA,
            hue_order=["Easy", "Hard"], kde=mostrar_kde, alpha=0.75, ax=ax,
        )
    else:
        sns.histplot(data=dados, x=var_hist, color=COR_NEUTRA, kde=mostrar_kde, ax=ax)

    media, mediana = dados[var_hist].mean(), dados[var_hist].median()
    ax.axvline(media, color="tab:red", linestyle="--", linewidth=1.5, label=f"Média = {media:,.1f}")
    ax.axvline(mediana, color="black", linestyle="-", linewidth=1.5, label=f"Mediana = {mediana:,.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Nº de rounds")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    assim = dados[var_hist].skew()
    if pd.notna(assim):
        st.caption(f"Assimetria (skewness): {assim:.2f} — valores bem acima de 1 indicam cauda longa à direita.")