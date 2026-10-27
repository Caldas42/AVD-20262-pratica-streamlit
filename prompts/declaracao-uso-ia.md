# Declaração de Uso de IA

**Nome:** Pedro da Costa Carvalho Caldas
**Data:** 22/10/2026

## Ferramenta

Claude

## Prompts usados

Preciso de ajuda na minha disciplina de análise de dados, vou fazer um dashboard com streamlit. Te mandei minha EDA, faça um rascunho de como vai ser o dashboard tomando como exemplo esse rascunho que te mandei.

Vamos me ajudar agora para a entrega 2 do projeto:
1. Adicione os gráficos de verdade ao Dashboard 1, usando `@st.cache_data`. Use **ALTAIR**: Veja o guia [ALTAIR-CROSSFILTER.md](ALTAIR-CROSSFILTER.md).

Escolha a identidade visual do app (cor de fundo, cor de destaque, fonte) em usando esse exemplo do config.toml da professora tomando como base as cores do jogo Bloons TD 6.

Vamos criar um dashboard2 agora, faça o rascunho e mande como `rascunho-dashboard2.jpg` (ou `.png`). As minhas métricas serão as seguintes: Pico de RBE (contagem, máximo), RBE mediano por round (média, na forma de mediana), % de rounds com bloons da classe MOAB (proporção), % de bloons camuflados (proporção) e Maior salto de RBE (variação, % em relação ao round anterior). Defina os widgets que vão controlar o dashboard.

5. Crie abas (`st.tabs`) separando Dashboard 1 e Dashboard 2 no mesmo app.
6. Implemente o Dashboard 2 a partir do rascunho: widgets → métricas (`st.metric`) → visuais com Altair. Para ligar a seleção de um gráfico às métricas e aos outros gráficos, use o guia [ALTAIR-CROSSFILTER.md](ALTAIR-CROSSFILTER.md).

## O que foi aceito, modificado ou rejeitado e por quê

- **Aceito:** O primeiro rascunho do dashboard 1, a adição de cache e altair, rascunho do dashboard 2, implementação do dashboard 2 no app.py.
- **Modificado:** O config.toml eu praticamente fiz sozinho, não gostei do que ele fez.
- **Rejeitado:** Nada
