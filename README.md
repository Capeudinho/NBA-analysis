# Análise exploratória NBA 2025-26 — New York Knicks × Phoenix Suns

Projeto 1 da disciplina **Laboratório de Ciência de Dados** (UFAPE, 2026.2). O relatório é um **sistema web em
Python + Flask + Jinja** que apresenta, para cada item do enunciado, os resultados (tabelas e gráficos interativos) e a
interpretação textual. Não há relatório em PDF separado: todo o conteúdo analítico está dentro do sistema.

Times do grupo: **New York Knicks (NYK)** e **Phoenix Suns (PHX)**.

## Execução local (comando único)

Pré-requisito: Python 3.10 ou superior.

```bash
pip install -r requirements.txt && python app.py
```

Depois, abra <http://127.0.0.1:5000> no navegador. A aplicação funciona sem internet (a biblioteca de gráficos é servida
pela própria aplicação; apenas a fonte tipográfica vem do Google Fonts, com alternativa local automática).

Recomendado usar um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && python app.py
```

Testes automatizados (opcionais):

```bash
pip install pytest && python -m pytest
```

## Hospedagem (link público)

O repositório já inclui `render.yaml` e `Procfile`. No [Render](https://render.com): *New → Blueprint* (ou *Web Service*)
apontando para este repositório; o serviço usa `pip install -r requirements.txt` no build e
`gunicorn app:app` para iniciar. Qualquer plataforma compatível com `Procfile` (Railway, Heroku, Fly.io) também funciona.
A porta é lida da variável de ambiente `PORT`.

## Seções do sistema

| Seção | Rota | Conteúdo |
|---|---|---|
| 0 · Visão geral | `/` | Números da base, resumo e roteiro |
| 1 · Estrutura e tratamento | `/base` | Linhas, colunas, tipos de dados, tamanho, tratamento da data, renomeação, conversão de categóricas (WL, MATCHUP), colunas excluídas para ML |
| 2 · Dicionário de dados | `/dicionario` | Significado, tipo, escala e unidade de cada coluna original e derivada |
| 3.1 · Valores especiais | `/qualidade/especiais` | Zeros, negativos, duplicados, valores fora de domínio e 22 regras de consistência |
| 3.2 · Valores ausentes | `/qualidade/ausentes` | Contagem por coluna, MCAR/MAR/MNAR, testes de Little, Jamshidian-Jalal, Listing-Schlittgen e regressão logística de Fairclough, com laboratório de simulação interativo |
| 3.3 · Outliers | `/qualidade/outliers` | Z-score, Z-score robusto, IQR/Boxplot, Grubbs, ECOD (por coluna) e ECOD, K-NN, LOF, DBSCAN (multivariados), com teoria |
| 4.1 · Medidas e gráficos | `/descritiva` | Média, mediana, moda, mín., máx., amplitude, desvio padrão, variância, Q1, Q3, outliers; boxplot completo; barras comparativas (valores brutos e z-score) |
| 4.2 · Assimetria | `/assimetria` | G1 (Fisher-Pearson ajustado), g1, Bowley, classificação, histograma + KDE + boxplot |
| 4.3 · Coeficiente de variação | `/cv` | CV amostral em %, validade, CV robusto, comparação com Poisson, quadro-resumo |
| 4.4 · IQR | `/iqr` | Q1, Q3, IQR, comparação com amplitude e desvio padrão, regra 1,5 × IQR e 3 × IQR, quadro-resumo |
| 5 · Síntese | `/sintese` | Conclusões, perfil dos times, decisões metodológicas e limitações |

Todas as páginas de análise têm seletores de **time**, **variável** e/ou **escopo** que recalculam tabelas, gráficos e
textos.

## Principais decisões metodológicas

- Desvio padrão e variância **amostrais** (n − 1); quartis por **interpolação linear** (tipo 7, padrão numpy/pandas) em
  todo o relatório, inclusive no boxplot.
- Outliers pela regra de **Tukey (k = 1,5)**, com k = 3 para extremos; outliers **mantidos** (nenhum viola as regras de
  consistência).
- Assimetria pelo coeficiente **G1 (Fisher-Pearson ajustado)**, complementado por Bowley; intensidade segundo Bulmer
  (|G1| < 0,5; 0,5–1; ≥ 1).
- CV = s / média × 100 %; versões robustas (MAD/mediana e IQR/mediana) para comparação.
- z-score no gráfico de barras: média do time relativa às médias dos 30 times da liga.
- K-NN, LOF e DBSCAN aplicados no espaço das 7 variáveis padronizadas (em uma única coluna de contagens com muitos
  empates esses métodos degeneram).
- Tabelas com 2 casas decimais, vírgula decimal e unidades "por jogo".

## Estrutura do código

```
app.py                     # ponto de entrada (Flask)
data/                      # CSV fornecido
nba_analysis/
  config.py                # times, variáveis, rótulos e parâmetros
  dictionary.py            # dicionário de dados e decisões para ML
  data.py                  # leitura, renomeação, conversão de tipos e datas
  quality.py               # valores especiais e regras de consistência
  missing.py               # testes de MCAR (Little, Jamshidian-Jalal, LS, Fairclough) e simulação
  outliers.py              # Z-score, IQR, Grubbs, ECOD, K-NN, LOF, DBSCAN
  stats.py                 # medidas descritivas, assimetria, CV e IQR
  charts.py                # gráficos Plotly
  narrative.py             # textos de interpretação gerados a partir dos resultados
  formatting.py            # formatação numérica no padrão brasileiro
  routes.py                # rotas/páginas
  templates/, static/      # Jinja, CSS e JS
tests/                     # testes automatizados
```
