# Justificativa das métricas — Dashboard 2

## Métrica 1: Pico de RBE
- **Variável:** RBE.
- **O que mede:** o round mais pesado do intervalo e em que round ele acontece.
- **Por que foi escolhida:** o RBE chega a mais de 1 milhão, e a defesa precisa aguentar o pior caso.
- **Público-alvo:** jogadores avançados ou que se interessam por criar estratégias em BTD6.
- **Qual decisão pode ser apoiada por essa métrica?** quanto de dano total a defesa precisa ter.
- **O que comunica ao público-alvo:** o tamanho do pior round.

## Métrica 2: RBE mediano por Round
- **Variável:** RBE.
- **O que mede:** o round “típico” do intervalo.
- **Por que foi escolhida:** a EDA mostrou assimetria forte (média de 164.866 contra mediana de 9.780), então a mediana representa melhor.
- **Público-alvo:** jogadores avançados ou que se interessam por criar estratégias em BTD6.
- **Qual decisão pode ser apoiada por essa métrica?** dimensionar a defesa base e a economia.
- **O que comunica ao público-alvo:** quanto pressão um round comum exerce.

## Métrica 3: % de rounds com bloons da classe MOAB
- **Variável:** as colunas da classe MOAB.
- **O que mede:** em quantos rounds do intervalo aparece pelo menos um bloon dessa classe.
- **Por que foi escolhida:** a EDA mostrou que esses bloons quase não aparecem nos rounds Easy.
- **Público-alvo:** jogadores avançados ou que se interessam por criar estratégias em BTD6.
- **Qual decisão pode ser apoiada por essa métrica?** quando investir em torres que causam dano alto contra MOAB.
- **O que comunica ao público-alvo:** a frequência da ameaça pesada.

## Métrica 4: % de bloons camuflados
- **Variável:** as colunas Camo_* sobre o total de bloons.
- **O que mede:** a parcela dos bloons que torres comuns não enxergam.
- **Por que foi escolhida:** camuflado exige detecção, e sem ela o bloon passa.
- **Público-alvo:** jogadores avançados ou que se interessam por criar estratégias em BTD6.
- **Qual decisão pode ser apoiada por essa métrica?** se vale ter torres com detecção de camuflado no intervalo.
- **O que comunica ao público-alvo:** quantos bloons que vão precisar de alguma medida de detecção de camuflados.

## Métrica 5: Maior salto de RBE
- **Variável:** RBE.
- **O que mede:** o maior aumento de um round para o seguinte, e em qual round.
- **Por que foi escolhida:** dificuldade em jogo não cresce de forma suave.
- **Público-alvo:** jogadores avançados ou que se interessam por criar estratégias em BTD6.
- **Qual decisão pode ser apoiada por essa métrica?** em qual round melhorar os macacos antes do salto.
- **O que comunica ao público-alvo:** onde o jogo “dá um pulo”.