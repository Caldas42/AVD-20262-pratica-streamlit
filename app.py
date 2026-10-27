"""
BTD6 — Quantidade de bloons por round
Dois dashboards em abas (st.tabs), gráficos em Altair, dados e cálculos com @st.cache_data.

  Dashboard 1: Correlação (heatmap) e Distribuição (histograma)
  Dashboard 2: KPIs (st.metric) + gráficos com seleção cruzada
               (arrastar em A filtra B, C e os KPIs; clicar em B destaca em A e C)

Rodar:  streamlit run app.py
Deps:   pip install streamlit altair pandas numpy kagglehub
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Paleta: teal + coral, com dourado e roxo de apoio (legível nos temas light e dark)
# ----------------------------------------------------------------------------
COR_EASY = "#2a9d8f"     # teal
COR_HARD = "#e76f51"     # coral
COR_TODOS = "#7a8fa6"    # azul-acinzentado (sem separar por dificuldade)
COR_BASE = "#8fa3ad"     # cinza-azulado: contexto / "não selecionado" (Dashboard 2)

GRUPOS = ["Easy", "Hard", "Todos"]
ESCALA_BARRAS = alt.Scale(domain=GRUPOS, range=[COR_EASY, COR_HARD, COR_TODOS])
ESCALA_LINHAS = alt.Scale(domain=GRUPOS, range=["#1b6f65", "#b5472e", "#4a5f73"])
ESCALA_STATS = alt.Scale(domain=["Média", "Mediana"], range=["#7b61b0", "#d98e04"])
DASH_STATS = alt.Scale(domain=["Média", "Mediana"], range=[[6, 4], [1, 0]])
ESCALA_CORR = alt.Scale(domain=[-1, 0, 1], range=["#e76f51", "#f7f4ef", "#2a9d8f"])

# Variáveis pré-selecionadas no heatmap (só entram se existirem no dataset)
DEFAULT_CORR = ["RBE", "Red", "Blue", "Green", "Pink", "Black", "White",
                "Zebra", "MOAB", "BFB", "ZOMG", "DDT"]

# Dashboard 2: famílias de bloons (cada coluna cai em UMA família, por prioridade)
ORDEM_FAMILIAS = ["Comum", "Regen", "Camo", "Fortified", "Classe MOAB"]
COLS_MOAB = ["MOAB", "Fortified_MOAB", "BFB", "Fortified_BFB", "ZOMG", "Fortified_ZOMG",
             "BAD", "Fortified_BAD", "DDT", "Fortified_DDT"]


# ----------------------------------------------------------------------------
# Dados (cache)
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


# ============================================================================
# DASHBOARD 1 — Correlação e distribuição
# ============================================================================
def filtrar_intervalo(intervalo: tuple) -> pd.DataFrame:
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

    corr = sub.corr(method=metodo)
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


def grafico_heatmap(corr_long: pd.DataFrame, ordem: list, metodo: str) -> alt.Chart:
    n = len(ordem)
    lado = max(18, min(40, 420 // n))  # tamanho da célula, para caber qualquer nº de variáveis

    base = alt.Chart(corr_long).encode(
        x=alt.X("x:N", sort=ordem[:-1], title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("y:N", sort=ordem[1:], title=None),
    )
    celulas = base.mark_rect().encode(
        color=alt.Color(
            "corr:Q",
            scale=ESCALA_CORR,
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
            color=alt.condition("abs(datum.corr) > 0.6", alt.value("white"), alt.value("#1f3a44")),
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

    return (
        alt.layer(*camadas)
        .resolve_scale(color="independent")  # cores de grupo e de estatística separadas
        .properties(height=420)
    )


def renderizar_dashboard1(intervalo, vars_corr, metodo, var_hist,
                          por_dificuldade, usar_log, mostrar_kde):
    n_filtrados = len(filtrar_intervalo(intervalo))
    st.subheader("Correlação e distribuição")
    st.caption(f"{n_filtrados} rounds selecionados (round {intervalo[0]} a {intervalo[1]}) "
               f"de {len(carregar_dados())} no dataset.")

    if n_filtrados == 0:
        st.warning("Nenhum round no intervalo selecionado.")
        return

    col_corr, col_dist = st.columns([1, 1.4])

    # --- Correlação ---------------------------------------------------------
    with col_corr:
        st.markdown("**Correlação**")
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

    # --- Distribuição -------------------------------------------------------
    with col_dist:
        st.markdown(f"**Distribuição — {var_hist}**")
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


# ============================================================================
# DASHBOARD 2 — KPIs + seleção cruzada
# ============================================================================
def familia_da_coluna(coluna: str) -> str:
    """Cada coluna de bloon cai em uma única família (prioridade: MOAB > Camo > Regen > Fortified)."""
    if coluna in COLS_MOAB:
        return "Classe MOAB"
    if coluna.startswith("Camo"):
        return "Camo"
    if coluna.startswith("Regen"):
        return "Regen"
    if "Fortified" in coluna:
        return "Fortified"
    return "Comum"


@st.cache_data
def preparar_base_dashboard2():
    """Duas tabelas: uma linha por round (KPIs, linha A, barras C) e outra em formato longo
    (round x coluna de bloon com quantidade > 0), que liga cada round às suas famílias."""
    df = carregar_dados().sort_values("Round").reset_index(drop=True)
    cols_bloons = [c for c in df.columns if c not in ("Round", "RBE", "round_difficulty")]
    cols_moab = [c for c in COLS_MOAB if c in df.columns]
    cols_camo = [c for c in cols_bloons if c.startswith("Camo")]

    rounds = df[["Round", "RBE", "round_difficulty"]].copy()
    rounds["total_bloons"] = df[cols_bloons].sum(axis=1)
    rounds["n_moab"] = df[cols_moab].sum(axis=1)
    rounds["n_camo"] = df[cols_camo].sum(axis=1)
    rounds["rbe_var_pct"] = rounds["RBE"].pct_change() * 100  # vs round anterior

    longo = df.melt(id_vars=["Round"], value_vars=cols_bloons,
                    var_name="coluna", value_name="quantidade")
    longo = longo[longo["quantidade"] > 0].copy()
    longo["familia"] = longo["coluna"].map(familia_da_coluna)
    longo = longo.merge(rounds[["Round", "RBE", "rbe_var_pct", "round_difficulty"]], on="Round")
    return rounds, longo[["Round", "familia", "coluna", "quantidade",
                          "RBE", "rbe_var_pct", "round_difficulty"]]


@st.cache_data
def filtrar_dashboard2(intervalo: tuple, familias: tuple):
    """Aplica os widgets: slider de intervalo (rounds) e multiselect de famílias (gráficos)."""
    rounds, longo = preparar_base_dashboard2()
    rounds_f = rounds[rounds["Round"].between(*intervalo)]
    longo_f = longo[longo["Round"].between(*intervalo) & longo["familia"].isin(familias)]
    return rounds_f, longo_f


def calcular_kpis(rounds_sel: pd.DataFrame) -> dict:
    """As 5 métricas planejadas, calculadas sobre os rounds selecionados."""
    pico = rounds_sel.loc[rounds_sel["RBE"].idxmax()]
    com_variacao = rounds_sel.dropna(subset=["rbe_var_pct"])
    salto = com_variacao.loc[com_variacao["rbe_var_pct"].idxmax()] if not com_variacao.empty else None

    return {
        "pico_rbe": int(pico["RBE"]),
        "pico_round": int(pico["Round"]),
        "rbe_mediano": float(rounds_sel["RBE"].median()),
        "pct_rounds_moab": float((rounds_sel["n_moab"] > 0).mean() * 100),
        "rounds_com_moab": int((rounds_sel["n_moab"] > 0).sum()),
        "pct_camo": float(rounds_sel["n_camo"].sum() / rounds_sel["total_bloons"].sum() * 100),
        "salto_pct": float(salto["rbe_var_pct"]) if salto is not None else None,
        "salto_round": int(salto["Round"]) if salto is not None else None,
    }


def fmt_int(x) -> str:
    return f"{x:,.0f}".replace(",", ".")


def fmt_pct(x, casas=1) -> str:
    return f"{x:.{casas}f}%".replace(".", ",")


def montar_graficos_dashboard2(rounds_f, longo_f, escala_log, intervalo, familias):
    """A (linha, arrastar) + B (barras, clicar) + C (barras por round) num único objeto,
    para a seleção se propagar entre eles (ver ALTAIR-CROSSFILTER.md)."""
    brush = alt.selection_interval(name="faixa", encodings=["x"])
    # empty=False: sem clique, nada fica "destacado"
    clique = alt.selection_point(name="familia", fields=["familia"], empty=False)

    cor_brush, cor_familia = COR_EASY, COR_HARD
    x_round = alt.X("Round:Q", title="Round",
                    scale=alt.Scale(domain=[intervalo[0] - 1, intervalo[1] + 1], nice=False))
    n_rounds = len(rounds_f)

    # ---- A: RBE por round (arrastar = selecionar intervalo) -------------------
    y_rbe = alt.Y("RBE:Q", title="RBE (escala log)" if escala_log else "RBE",
                  scale=alt.Scale(type="log" if escala_log else "linear"))
    tip_a = [alt.Tooltip("Round:Q", title="Round"),
             alt.Tooltip("RBE:Q", title="RBE", format=","),
             alt.Tooltip("round_difficulty:N", title="Dificuldade")]

    linha_base = (
        alt.Chart(rounds_f)
        .mark_line(color=COR_BASE, strokeWidth=2, point=alt.OverlayMarkDef(size=14, color=COR_BASE))
        .encode(x=x_round, y=y_rbe, tooltip=tip_a)
        .add_params(brush)
    )
    linha_sel = (
        alt.Chart(rounds_f).transform_filter(brush)
        .mark_line(color=cor_brush, strokeWidth=3)
        .encode(x=x_round, y=y_rbe)
    )
    pontos_familia = (  # rounds onde a família clicada em B aparece
        alt.Chart(longo_f).transform_filter(clique)
        .mark_circle(size=70, color=cor_familia, opacity=0.9)
        .encode(x=x_round, y=y_rbe,
                tooltip=[alt.Tooltip("Round:Q", title="Round"),
                         alt.Tooltip("familia:N", title="Família"),
                         alt.Tooltip("RBE:Q", title="RBE", format=",")])
    )
    grafico_a = alt.layer(linha_base, linha_sel, pontos_familia).properties(
        width=820, height=240,
        title=alt.TitleParams("A · RBE por round — arraste para selecionar rounds",
                              anchor="start", fontSize=14))

    # ---- B: em quantos rounds cada família aparece (clicar = destacar) ---------
    fam_visiveis = [f for f in ORDEM_FAMILIAS if f in familias]
    grafico_b = (
        alt.Chart(longo_f).transform_filter(brush)
        .mark_bar()
        .encode(
            y=alt.Y("familia:N", sort=fam_visiveis, title=None),
            x=alt.X("distinct(Round):Q", title="Rounds em que a família aparece",
                    scale=alt.Scale(domain=[0, max(n_rounds, 1)])),
            color=alt.condition(clique, alt.value(cor_familia), alt.value(COR_BASE)),
            tooltip=[alt.Tooltip("familia:N", title="Família"),
                     alt.Tooltip("distinct(Round):Q", title="Rounds")],
        )
        .add_params(clique)
        .properties(width=330, height=200,
                    title=alt.TitleParams("B · Composição — clique para destacar",
                                          anchor="start", fontSize=14))
    )

    # ---- C: variação do RBE round a round -----------------------------------------
    c_base = (
        alt.Chart(rounds_f.dropna(subset=["rbe_var_pct"]))
        .mark_bar(size=3)
        .encode(
            x=x_round,
            y=alt.Y("rbe_var_pct:Q", title="Variação do RBE (%)"),
            color=alt.condition(brush, alt.value(cor_brush), alt.value(COR_BASE)),
            tooltip=[alt.Tooltip("Round:Q", title="Round"),
                     alt.Tooltip("rbe_var_pct:Q", title="Variação (%)", format=".1f")],
        )
    )
    c_familia = (  # rounds onde a família clicada em B aparece
        alt.Chart(longo_f).transform_filter(clique)
        .mark_bar(size=3, color=cor_familia)
        .encode(x=x_round, y=alt.Y("mean(rbe_var_pct):Q"))
    )
    grafico_c = alt.layer(c_base, c_familia).properties(
        width=380, height=200,
        title=alt.TitleParams("C · Variação do RBE round a round", anchor="start", fontSize=14))

    return (alt.vconcat(grafico_a, alt.hconcat(grafico_b, grafico_c))
            .configure_view(stroke=None))


def renderizar_dashboard2(intervalo, escala_log, familias):
    st.subheader("Pressão por round e composição dos bloons")

    if not familias:
        st.info("Selecione pelo menos uma família de bloons na sidebar.")
        return
    rounds_f, longo_f = filtrar_dashboard2(intervalo, tuple(familias))
    if rounds_f.empty:
        st.warning("Nenhum round no intervalo selecionado.")
        return

    area_kpis = st.container()  # os KPIs ficam no topo, mas só dá pra preencher depois do gráfico
    st.caption("Arraste em A para selecionar rounds (filtra B, C e os KPIs). "
               "Clique numa barra de B para destacar os rounds dessa família em A e C.")

    chart = montar_graficos_dashboard2(rounds_f, longo_f, escala_log, intervalo, familias)
    # a chave muda com os widgets, assim uma seleção antiga não sobrevive a um novo intervalo
    chave = f"d2_chart_{intervalo[0]}_{intervalo[1]}_{int(escala_log)}_{'-'.join(familias)}"
    evento = st.altair_chart(chart, key=chave, on_select="rerun", width="content")

    # Altair -> Python: faixa de rounds selecionada por arrasto em A (exemplo 3 do guia)
    faixa = evento.selection.get("faixa", {})
    rounds_sel, legenda = rounds_f, f"rounds {intervalo[0]} a {intervalo[1]} (intervalo do slider)"
    if "Round" in faixa:
        minimo, maximo = faixa["Round"]
        candidatos = rounds_f[rounds_f["Round"].between(minimo, maximo)]
        if not candidatos.empty:
            rounds_sel = candidatos
            legenda = (f"rounds {int(candidatos['Round'].min())} a "
                       f"{int(candidatos['Round'].max())} (seleção em A)")

    k = calcular_kpis(rounds_sel)
    with area_kpis:
        st.caption(f"KPIs calculados para os {legenda} — {len(rounds_sel)} rounds.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pico de RBE", fmt_int(k["pico_rbe"]),
                  help="Maior RBE (Red Bloon Equivalent) de um único round no recorte.")
        c1.caption(f"no round {k['pico_round']}")
        c2.metric("RBE mediano por round", fmt_int(k["rbe_mediano"]),
                  help="Mediana do RBE: o round típico. Usamos a mediana porque o RBE é muito "
                       "assimétrico (a média seria puxada pelos rounds finais).")
        c2.caption("mediana dos rounds")
        c3.metric("Rounds com classe MOAB", fmt_pct(k["pct_rounds_moab"]),
                  help="% de rounds com pelo menos um MOAB, BFB, ZOMG, BAD ou DDT "
                       "(incluindo as versões Fortified).")
        c3.caption(f"{k['rounds_com_moab']} de {len(rounds_sel)} rounds")
        c4.metric("Bloons camuflados", fmt_pct(k["pct_camo"]),
                  help="Parcela dos bloons do recorte que são camo (colunas Camo_*).")
        c4.caption("do total de bloons")
        if k["salto_pct"] is None:
            c5.metric("Maior salto de RBE", "—", help="Precisa de pelo menos 2 rounds.")
        else:
            c5.metric("Maior salto de RBE", "+" + fmt_pct(k["salto_pct"], 0),
                      help="Maior aumento percentual do RBE em relação ao round anterior.")
            c5.caption(f"no round {k['salto_round']} vs anterior")


# ============================================================================
# App: sidebar (widgets) + abas
# ============================================================================
def main():
    st.set_page_config(page_title="BTD6 — Bloons por Round", layout="wide")

    df = carregar_dados()
    vars_quant = df.select_dtypes(include=np.number).columns.tolist()
    r_min, r_max = int(df["Round"].min()), int(df["Round"].max())

    # ---- Sidebar: cada dashboard tem seus próprios widgets ----------------------
    with st.sidebar:
        st.header("Filtros")
        with st.expander("Dashboard 1 — correlação e distribuição", expanded=True):
            d1_intervalo = st.slider("Intervalo de rounds", r_min, r_max, (r_min, r_max),
                                     key="d1_intervalo")
            d1_vars = st.multiselect(
                "Variáveis do heatmap", options=vars_quant,
                default=[v for v in DEFAULT_CORR if v in vars_quant] or vars_quant[:8],
                key="d1_vars")
            d1_metodo = st.selectbox(
                "Método de correlação", ["spearman", "pearson", "kendall"], index=0, key="d1_metodo",
                help="Na EDA nenhuma variável passou no Shapiro-Wilk, então Spearman é o padrão.")
            d1_var_hist = st.selectbox(
                "Variável do histograma", vars_quant, key="d1_var_hist",
                index=vars_quant.index("RBE") if "RBE" in vars_quant else 0)
            d1_dif = st.checkbox("Separar por dificuldade (Easy x Hard)", value=True, key="d1_dif")
            d1_log = st.checkbox("Escala log(1 + x)", value=False, key="d1_log",
                                 help="Reduz a assimetria forte, como feito com o RBE na EDA.")
            d1_kde = st.checkbox("Mostrar curva KDE", value=True, key="d1_kde")

        with st.expander("Dashboard 2 — pressão por round", expanded=True):
            d2_intervalo = st.slider("Intervalo de rounds", r_min, r_max, (r_min, r_max),
                                     key="d2_intervalo")
            d2_escala = st.radio("Escala do RBE", ["Linear", "Log"], horizontal=True, key="d2_escala",
                                 help="O RBE cresce de forma muito assimétrica; a escala log ajuda a "
                                      "ver os rounds iniciais.")
            d2_familias = st.multiselect(
                "Famílias de bloons", ORDEM_FAMILIAS, default=ORDEM_FAMILIAS, key="d2_familias",
                help="Controla o que aparece nos gráficos A, B e C. Os KPIs não mudam com este filtro.")

    # ---- Abas ------------------------------------------------------------------
    st.title("Bloons TD 6 — quantidade de bloons por round")
    aba1, aba2 = st.tabs(["Dashboard 1 · Correlação e distribuição",
                          "Dashboard 2 · Pressão por round"])
    with aba1:
        renderizar_dashboard1(d1_intervalo, d1_vars, d1_metodo, d1_var_hist, d1_dif, d1_log, d1_kde)
    with aba2:
        renderizar_dashboard2(d2_intervalo, d2_escala == "Log", d2_familias)


if __name__ == "__main__":  # o Streamlit executa o script como __main__
    main()