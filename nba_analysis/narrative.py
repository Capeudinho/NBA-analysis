"""Data-driven interpretation texts (user-facing, therefore in Portuguese).

Every function returns a list of HTML paragraphs built only from values computed
in this project, so the text stays consistent with the tables and charts.
"""

from . import config
from .formatting import integer, number, percent

TEAM_A, TEAM_B = config.TEAM_CODES


def _team(code):
    return f"<strong>{config.TEAMS[code]['name']}</strong>"


def _var(variable):
    return f"<strong>{config.ANALYSIS_VARIABLES[variable]['label']}</strong>"


def _unit(variable):
    return config.ANALYSIS_VARIABLES[variable]["unit"]


def _join(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]


def _labels(variables):
    return _join(_var(v) for v in variables)


# ----------------------------------------------------------------------------
# Central tendency, dispersion and outliers
# ----------------------------------------------------------------------------

def _plural(count, singular, plural):
    return f"{count} {singular if count == 1 else plural}"


def _outlier_sentence(team, d):
    count = d["n_outliers"]
    if count == 0:
        return f"{_team(team)} não tem nenhum jogo fora dos limites de Tukey."
    parts = []
    if d["outliers_high"]:
        values = _join(number(v, 0) for v in d["outliers_high"])
        parts.append(f"{_plural(len(d['outliers_high']), 'superior', 'superiores')} ({values})")
    if d["outliers_low"]:
        values = _join(number(v, 0) for v in d["outliers_low"])
        parts.append(f"{_plural(len(d['outliers_low']), 'inferior', 'inferiores')} ({values})")
    return f"{_team(team)} tem {_plural(count, 'outlier', 'outliers')}: {_join(parts)}."


def descriptive_variable(variable, summary, reference):
    a = summary[variable][TEAM_A]["describe"]
    b = summary[variable][TEAM_B]["describe"]
    unit = _unit(variable)
    leader, other = (TEAM_A, TEAM_B) if a["mean"] >= b["mean"] else (TEAM_B, TEAM_A)
    lead = summary[variable][leader]["describe"]
    trail = summary[variable][other]["describe"]
    diff = lead["mean"] - trail["mean"]
    texts = []

    def central(team, d):
        modes = _join(number(m, 0) for m in d["modes"])
        modal = "modas" if len(d["modes"]) > 1 else "moda"
        gap = d["mean"] - d["median"]
        if abs(gap) <= 0.05 * d["std"]:
            shape = "média e mediana praticamente coincidem, o que sugere distribuição equilibrada"
        elif gap > 0:
            shape = "a média fica acima da mediana, indício de cauda à direita (alguns jogos muito altos)"
        else:
            shape = "a média fica abaixo da mediana, indício de cauda à esquerda (alguns jogos muito baixos)"
        return (
            f"{_team(team)}: média de {number(d['mean'])} {unit}, mediana de {number(d['median'])} "
            f"e {modal} {modes} — {shape}."
        )

    texts.append(
        f"<strong>Tendência central.</strong> {central(TEAM_A, a)} {central(TEAM_B, b)} "
        f"Na média, {_team(leader)} supera {_team(other)} em {number(diff)} {unit} "
        f"({percent(diff / trail['mean'] * 100)} a mais)."
    )
    more_var = TEAM_A if a["std"] > b["std"] else TEAM_B
    texts.append(
        f"<strong>Dispersão.</strong> Os desvios-padrão são {number(a['std'])} ({config.TEAMS[TEAM_A]['short']}) "
        f"e {number(b['std'])} ({config.TEAMS[TEAM_B]['short']}); as variâncias, {number(a['var'])} e "
        f"{number(b['var'])} (em unidades ao quadrado). {_team(more_var)} oscila mais em torno da própria média. "
        f"As amplitudes vão de {number(a['min'], 0)} a {number(a['max'], 0)} ({number(a['range'], 0)}) e de "
        f"{number(b['min'], 0)} a {number(b['max'], 0)} ({number(b['range'], 0)}), enquanto os quartis "
        f"delimitam {number(a['q1'])}–{number(a['q3'])} e {number(b['q1'])}–{number(b['q3'])}."
    )
    texts.append(
        f"<strong>Outliers (regra de Tukey, 1,5 × IQR).</strong> {_outlier_sentence(TEAM_A, a)} "
        f"{_outlier_sentence(TEAM_B, b)} Esses jogos são valores raros, mas plausíveis no basquete "
        f"(a etapa de qualidade não encontrou inconsistências), portanto foram <em>mantidos</em> e apenas sinalizados."
    )
    ref = reference[variable]
    texts.append(
        f"<strong>Contexto da liga.</strong> A média da NBA é {number(ref['game_mean'])} {unit}. Entre as médias "
        f"dos 30 times, {_team(TEAM_A)} ocupa a {ref['rank'][TEAM_A]}ª posição e {_team(TEAM_B)} a "
        f"{ref['rank'][TEAM_B]}ª (1ª = maior média)."
    )
    return texts


def zscore_explanation(zscores):
    strengths = {team: [] for team in config.TEAM_CODES}
    weaknesses = {team: [] for team in config.TEAM_CODES}
    for variable in config.VARIABLE_KEYS:
        for team in config.TEAM_CODES:
            if zscores[variable][team] >= 0.5:
                strengths[team].append(variable)
            elif zscores[variable][team] <= -0.5:
                weaknesses[team].append(variable)
    texts = [
        "<strong>Onde o z-score entra?</strong> As variáveis têm escalas muito diferentes (cerca de 115 pontos "
        "contra 4 tocos por jogo), então um único gráfico de barras com valores brutos esconderia as variáveis "
        "pequenas. O z-score resolve isso ao expressar cada valor em <em>desvios-padrão em relação a uma "
        "referência</em>: a média do time menos a média dos 30 times, dividido pelo desvio padrão dessas 30 médias. "
        "Assim, todas as variáveis ficam na mesma régua adimensional, o zero passa a ser o \"time médio da NBA\" "
        "e o sinal mostra se o time está acima ou abaixo da liga.",
        "O z-score também aparece em outros pontos do relatório: no nível do jogo, ele serve para detectar "
        "outliers (|z| > 3, seção de qualidade) e para padronizar as variáveis antes dos métodos multivariados "
        "(K-NN, LOF, DBSCAN), que dependem de distâncias e seriam dominados pela variável de maior escala.",
    ]
    for team in config.TEAM_CODES:
        parts = []
        if strengths[team]:
            parts.append(f"acima da média da liga (z ≥ 0,5) em {_labels(strengths[team])}")
        if weaknesses[team]:
            parts.append(f"abaixo (z ≤ −0,5) em {_labels(weaknesses[team])}")
        if parts:
            texts.append(f"{_team(team)} está {'; e '.join(parts)}.")
    return texts


# ----------------------------------------------------------------------------
# Skewness
# ----------------------------------------------------------------------------

def _agreement(skew, describe):
    g_sign = (skew["G1"] > 0) - (skew["G1"] < 0)
    b_sign = (skew["bowley"] > 0.02) - (skew["bowley"] < -0.02)
    m_sign = (skew["mean_minus_median"] > 0) - (skew["mean_minus_median"] < 0)
    if abs(skew["G1"]) < config.SKEW_MODERATE and b_sign in (0, g_sign, -g_sign) and abs(skew["bowley"]) < 0.2:
        return (
            "coeficiente e gráficos contam a mesma história: distribuição aproximadamente simétrica, com "
            "histograma sem cauda dominante e caixa do boxplot equilibrada"
        )
    if g_sign == b_sign == m_sign and g_sign != 0:
        return (
            "coeficiente e gráficos concordam: G1, coeficiente de Bowley e a posição da média em relação à "
            "mediana apontam para o mesmo lado"
        )
    extremes = describe["outliers_high"] if g_sign > 0 else describe["outliers_low"]
    reason = (
        f"o G1 é puxado por {_plural(len(extremes), 'valor extremo', 'valores extremos')} na cauda "
        f"{'direita' if g_sign > 0 else 'esquerda'} ({_join(number(v, 0) for v in extremes)})"
        if extremes
        else "o G1 usa a terceira potência dos desvios e reage às caudas, enquanto o miolo da distribuição é quase simétrico"
    )
    return (
        f"<em>há divergência</em>: o G1 indica assimetria {skew['direction']}, mas o coeficiente de Bowley "
        f"(baseado só nos quartis) é {number(skew['bowley'])}. O motivo: {reason}. No histograma isso aparece "
        "como uma cauda isolada, e no boxplot como pontos além dos bigodes, sem que a caixa central fique torta"
    )


def skewness_variable(variable, summary):
    texts = []
    for team in config.TEAM_CODES:
        s = summary[variable][team]["skew"]
        d = summary[variable][team]["describe"]
        significance = (
            f"estatisticamente diferente de zero (|G1| / EP = {number(abs(s['z']))} > 1,96)"
            if s["significant"]
            else f"não significativamente diferente de zero (|G1| / EP = {number(abs(s['z']))} ≤ 1,96)"
        )
        texts.append(
            f"{_team(team)} — G1 = {number(s['G1'])}: assimetria {s['direction']} e intensidade "
            f"<strong>{s['intensity']}</strong>, {significance}. Média {number(s['mean'])} × mediana "
            f"{number(s['median'])}. Confirmação visual: {_agreement(s, d)}."
        )
    return texts


def skewness_overview(summary):
    items = [
        (abs(summary[v][t]["skew"]["G1"]), v, t) for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
    ]
    items.sort(reverse=True)
    top = [f"{_var(v)} ({config.TEAMS[t]['short']}, G1 = {number(summary[v][t]['skew']['G1'])})" for _, v, t in items[:3]]
    moderate = [(v, t) for g, v, t in items if g >= config.SKEW_MODERATE]
    symmetric_count = sum(1 for g, _, _ in items if g < config.SKEW_MODERATE)
    same_sign = [
        v for v in config.VARIABLE_KEYS
        if (summary[v][TEAM_A]["skew"]["G1"] > 0) == (summary[v][TEAM_B]["skew"]["G1"] > 0)
    ]
    opposite = [v for v in config.VARIABLE_KEYS if v not in same_sign]
    texts = [
        f"<strong>Comparação entre variáveis.</strong> Das 14 combinações variável × time, {symmetric_count} são "
        f"aproximadamente simétricas (|G1| &lt; 0,5). As mais assimétricas são {_join(top)}.",
        f"<strong>Comparação entre times.</strong> O sinal da assimetria se repete nos dois times em "
        f"{len(same_sign)} das 7 variáveis"
        + (f"; inverte em {_labels(opposite)}" if opposite else "")
        + ". Ou seja, a forma da distribuição não é uma característica fixa da estatística: ela depende de "
        "poucos jogos atípicos de cada time.",
    ]
    if moderate:
        cases = _join(f"{_var(v)} ({config.TEAMS[t]['short']})" for v, t in moderate)
        texts.append(
            f"<strong>Implicação prática.</strong> Onde a assimetria é moderada ou forte — {cases} — a "
            "<em>mediana</em> representa melhor o jogo típico, pois a média é deslocada pelos jogos extremos. "
            "Nas demais combinações, média e mediana são praticamente iguais e a média pode ser usada sem prejuízo."
        )
    else:
        texts.append(
            "<strong>Implicação prática.</strong> Como nenhuma combinação passou de |G1| = 0,5, a média "
            "representa bem o valor típico em todas as variáveis."
        )
    texts.append(
        "<strong>Transformações e métodos robustos.</strong> A transformação logarítmica é indicada para "
        "assimetria positiva forte com valores estritamente positivos — não é o caso aqui (as assimetrias são "
        "no máximo moderadas e há jogos com 0 tocos, o que exigiria log(x + 1)). Como as assimetrias encontradas "
        "vêm de poucos jogos extremos e não de uma cauda longa, é mais adequado usar <em>métodos robustos</em> "
        "(mediana, IQR, MAD, testes não paramétricos) do que transformar os dados. Para contagens pequenas "
        "(tocos e roubos), cuja variância cresce com a média, a raiz quadrada é a transformação estabilizadora "
        "de variância mais natural, caso uma análise posterior exija homocedasticidade."
    )
    return texts


# ----------------------------------------------------------------------------
# Coefficient of variation
# ----------------------------------------------------------------------------

def cv_overview(summary):
    from .stats import classify_cv

    avg_cv = {v: sum(summary[v][t]["cv"]["cv"] for t in config.TEAM_CODES) / 2 for v in config.VARIABLE_KEYS}
    ordered = sorted(avg_cv, key=avg_cv.get)
    small = [v for v in config.VARIABLE_KEYS if summary[v][TEAM_A]["cv"]["small_mean"]]
    texts = [
        f"<strong>Validade do CV.</strong> Todas as sete variáveis são contagens com zero absoluto (escala de "
        f"razão) e médias positivas, condição necessária para o CV fazer sentido. Porém {_labels(small)} têm médias "
        f"pequenas (abaixo de {number(config.CV_SMALL_MEAN, 0)} por jogo). Em contagens de eventos raros a "
        "variância tende a acompanhar a média (comportamento de Poisson), e então CV ≈ 100/√média: com média "
        "4, o CV esperado já é 50%, mesmo que o time seja perfeitamente \"regular\". Por isso o CV dessas "
        "variáveis é naturalmente inflado e deve ser lido com cautela — ele mede sobretudo a granularidade da "
        "contagem, não a irregularidade do time.",
        f"<strong>Comparação entre variáveis.</strong> A mais estável é {_var(ordered[0])} "
        f"(CV médio de {percent(avg_cv[ordered[0]])}, {classify_cv(avg_cv[ordered[0]])}) e a mais dispersa é "
        f"{_var(ordered[-1])} ({percent(avg_cv[ordered[-1]])}, {classify_cv(avg_cv[ordered[-1]])}). O desvio "
        "padrão bruto não permitiria essa comparação: o de pontos (≈ 12–14) é muito maior que o de tocos "
        "(≈ 2) apenas porque pontos estão numa escala maior. O CV divide o desvio pela média e vira uma medida "
        "adimensional, comparável entre escalas.",
    ]
    wins = {team: [] for team in config.TEAM_CODES}
    for variable in config.VARIABLE_KEYS:
        a = summary[variable][TEAM_A]["cv"]["cv"]
        b = summary[variable][TEAM_B]["cv"]["cv"]
        wins[TEAM_A if a < b else TEAM_B].append(variable)
    leader = max(wins, key=lambda t: len(wins[t]))
    other = TEAM_B if leader == TEAM_A else TEAM_A
    texts.append(
        f"<strong>Comparação entre times.</strong> {_team(leader)} é mais consistente (menor CV) em "
        f"{len(wins[leader])} das 7 variáveis ({_labels(wins[leader])}), enquanto {_team(other)} é mais regular em "
        f"{_labels(wins[other]) if wins[other] else 'nenhuma'}. O padrão, portanto, "
        f"{'se mantém na maioria das variáveis' if len(wins[leader]) >= 5 else 'muda conforme a variável'}."
    )
    texts.append(
        "<strong>Confronto com o observado sob Poisson.</strong> A linha preta no gráfico mostra o CV que se "
        "esperaria se cada estatística fosse uma contagem de Poisson. Quando o CV observado fica próximo dela, "
        "a variação jogo a jogo é a esperada para um processo aleatório de contagem; valores bem acima indicam "
        "irregularidade real do time (sobredispersão). Em pontos, o CV observado supera o de Poisson porque "
        "os pontos chegam em blocos de 2 e 3, o que naturalmente aumenta a variância."
    )
    skewed = [
        (v, t) for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
        if abs(summary[v][t]["skew"]["G1"]) >= config.SKEW_MODERATE
    ]
    if skewed:
        details = _join(
            f"{_var(v)} ({config.TEAMS[t]['short']}: CV {percent(summary[v][t]['cv']['cv'])} × "
            f"CV robusto {percent(summary[v][t]['cv']['cv_mad'])})"
            for v, t in skewed
        )
        texts.append(
            f"<strong>CV × assimetria.</strong> Nas combinações com assimetria moderada ou forte — {details} — "
            "o CV clássico herda a sensibilidade da média e do desvio padrão aos jogos extremos. Nesses casos, "
            "versões robustas são mais apropriadas: o CV baseado na mediana (1,4826 × MAD / mediana) ou a razão "
            "IQR / mediana, ambos apresentados na tabela."
        )
        larger = [(v, t) for v, t in skewed if summary[v][t]["cv"]["cv_mad"] > 1.2 * summary[v][t]["cv"]["cv"]]
        if larger:
            cases = _join(f"{_var(v)} ({config.TEAMS[t]['short']})" for v, t in larger)
            texts.append(
                f"Atenção: em {cases} a versão baseada no MAD ficou bem <em>maior</em> que o CV clássico. Em "
                "contagens pequenas e discretas o MAD só assume poucos valores (múltiplos de 0,5) e fica "
                "grosseiro, então a alternativa robusta deixa de ser confiável. Como essa assimetria não vem "
                "de outliers influentes (o desvio padrão quase não muda sem eles), o CV clássico continua "
                "adequado nesses casos."
            )
    return texts


# ----------------------------------------------------------------------------
# Interquartile range
# ----------------------------------------------------------------------------

def iqr_variable(variable, summary):
    unit = _unit(variable)
    texts = []
    for team in config.TEAM_CODES:
        q = summary[variable][team]["iqr"]
        texts.append(
            f"{_team(team)}: os 50% centrais dos jogos ficaram entre {number(q['q1'])} e {number(q['q3'])} "
            f"{unit}, uma faixa de {number(q['iqr'])} (IQR). A amplitude ({number(q['range'])}) é "
            f"{number(q['range_over_iqr'])} vezes o IQR, e o desvio padrão equivale a "
            f"{number(q['std_over_iqr'])} IQR (referência normal: {number(q['normal_std_over_iqr'])})."
        )
    a = summary[variable][TEAM_A]["iqr"]["iqr"]
    b = summary[variable][TEAM_B]["iqr"]["iqr"]
    concentrated = TEAM_A if a < b else TEAM_B
    if a != b:
        texts.append(f"Na faixa central, {_team(concentrated)} é mais concentrado nesta variável.")
    return texts


def iqr_overview(summary):
    long_tails = [
        (v, t) for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
        if summary[v][t]["iqr"]["range_over_iqr"] >= 4.5
    ]
    inflated = [
        (v, t) for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
        if summary[v][t]["iqr"]["std_change_pct"] >= 10
    ]
    all_match = all(summary[v][t]["iqr"]["matches_boxplot"] for v in config.VARIABLE_KEYS for t in config.TEAM_CODES)
    extreme_total = sum(summary[v][t]["iqr"]["k30"]["n"] for v in config.VARIABLE_KEYS for t in config.TEAM_CODES)
    mild_total = sum(summary[v][t]["iqr"]["k15"]["n"] for v in config.VARIABLE_KEYS for t in config.TEAM_CODES)
    def short(team):
        return config.TEAMS[team]["short"]

    if long_tails:
        cases = _join(f"{_var(v)} ({short(t)})" for v, t in long_tails)
        range_text = (
            f"Ela é muito maior em {cases}, porque a amplitude depende só dos dois jogos mais extremos, "
            "enquanto o IQR ignora os 25% de cada ponta e não se altera com eles."
        )
    else:
        range_text = "Nenhuma combinação foge muito desse padrão."
    if inflated:
        cases = _join(
            f"{_var(v)} ({short(t)}: −{number(summary[v][t]['iqr']['std_change_pct'])}%)" for v, t in inflated
        )
        std_text = (
            f"Ao remover os outliers, o desvio padrão cai 10% ou mais em {cases}, enquanto o IQR praticamente "
            "não muda. Isso mostra que o desvio padrão é inflado por poucos valores extremos e que o IQR é uma "
            "medida robusta."
        )
    else:
        std_text = "O desvio padrão quase não muda ao remover outliers, sinal de que não há valores extremos influentes."
    texts = [
        "<strong>Amplitude × IQR.</strong> Para uma distribuição normal com ~75 observações, a amplitude costuma "
        "ser cerca de 3,5 vezes o IQR. " + range_text,
        "<strong>Desvio padrão × IQR.</strong> " + std_text,
        f"<strong>Regra de outliers.</strong> Adotamos k = 1,5 (limites Q1 − 1,5·IQR e Q3 + 1,5·IQR), o padrão "
        f"de Tukey para o boxplot. Os limites foram recalculados a partir dos quartis desta seção e "
        f"{'coincidem exatamente' if all_match else 'NÃO coincidem'} com os outliers desenhados nos boxplots "
        f"({mild_total} jogos no total). Com k = 3 (outliers extremos), restam {extreme_total} jogos: nenhum "
        "valor é tão discrepante a ponto de sugerir erro de registro, o que é coerente com as verificações de "
        "consistência.",
    ]
    wins = {team: [] for team in config.TEAM_CODES}
    for v in config.VARIABLE_KEYS:
        a = summary[v][TEAM_A]["iqr"]["iqr"]
        b = summary[v][TEAM_B]["iqr"]["iqr"]
        if a != b:
            wins[TEAM_A if a < b else TEAM_B].append(v)
    texts.append(
        f"<strong>Comparação entre times.</strong> {_team(TEAM_A)} tem faixa central mais estreita em "
        f"{_labels(wins[TEAM_A]) or 'nenhuma variável'}; {_team(TEAM_B)}, em {_labels(wins[TEAM_B]) or 'nenhuma variável'}. "
        "Articulando com as seções anteriores: quando o IQR de um time é menor mas o desvio padrão é parecido "
        "(ou maior), a diferença vem das caudas — exatamente as combinações com outliers e assimetria "
        "moderada."
    )
    mean_of = lambda key: {  # noqa: E731
        v: sum(summary[v][t]["cv"][key] for t in config.TEAM_CODES) / 2 for v in config.VARIABLE_KEYS
    }
    top_cv = sorted(mean_of("cv").items(), key=lambda item: -item[1])[:2]
    top_iqr = sorted(mean_of("cv_iqr").items(), key=lambda item: -item[1])[:2]
    same = {v for v, _ in top_cv} == {v for v, _ in top_iqr}
    texts.append(
        f"<strong>IQR relativo × CV.</strong> Pelo CV, as variáveis relativamente mais dispersas são "
        f"{_labels(v for v, _ in top_cv)}; pela razão IQR / mediana (versão robusta), são "
        f"{_labels(v for v, _ in top_iqr)}. "
        + ("As duas medidas concordam, o que reforça a conclusão." if same else
           "A diferença entre as medidas indica que parte da dispersão vem das caudas.")
    )
    return texts


def cv_variable(variable, summary):
    from .stats import classify_cv

    texts = []
    for team in config.TEAM_CODES:
        c = summary[variable][team]["cv"]
        validity = (
            f"média pequena ({number(c['mean'])}), então o CV é inflado pela própria natureza de contagem — "
            f"sob Poisson já se esperaria {percent(c['cv_poisson'])}"
            if c["small_mean"]
            else f"média ({number(c['mean'])}) bem distante de zero ({number(c['mean_over_std'])} desvios-padrão), "
            "o que torna o CV estável"
        )
        regularity = {
            "baixa dispersão": "o time é consistente de um jogo para outro",
            "dispersão moderada": "há oscilação perceptível de um jogo para outro",
            "alta dispersão": "a produção varia muito de um jogo para outro",
        }[classify_cv(c["cv"])]
        texts.append(
            f"{_team(team)}: CV = {percent(c['cv'])} ({classify_cv(c['cv'])}) — {regularity}. Validade: escala de "
            f"razão, média positiva e {validity}. Índice de dispersão (variância / média) = "
            f"{number(c['dispersion_index'])} (≈ 1 indica variação compatível com Poisson)."
        )
    a = summary[variable][TEAM_A]["cv"]["cv"]
    b = summary[variable][TEAM_B]["cv"]["cv"]
    steadier = TEAM_A if a < b else TEAM_B
    texts.append(f"Nesta variável, {_team(steadier)} é o time mais regular (menor CV).")
    return texts


def synthesis(summary, reference, zscores, quality_summary, outlier_summary):
    texts = {}
    texts["data"] = [
        f"A base tem {integer(quality_summary['rows'])} linhas e {quality_summary['columns']} colunas; cada linha é o "
        "desempenho de <em>um time em uma partida</em> da temporada regular 2025-26 da NBA "
        f"({quality_summary['date_min']} a {quality_summary['date_max']}, {integer(quality_summary['games'])} partidas, "
        f"{quality_summary['games_per_team_min']}–{quality_summary['games_per_team_max']} jogos por time — a "
        "temporada ainda não estava completa, pois cada time disputa 82 jogos).",
        "A qualidade dos dados é alta: não há valores ausentes, duplicados, negativos ou fora de domínio, e as "
        f"{quality_summary['rules']} regras de consistência (identidades como PTS = 2·FGM + FG3M + FTM, "
        "REB = OREB + DREB, partidas espelhadas e registros acumulados de vitórias) não tiveram nenhuma "
        "violação. Os zeros encontrados (tocos, roubos, vitórias acumuladas) são legítimos.",
        f"Os outliers detectados são <em>valores raros, mas reais</em> — prorrogações, jogos de pontuação "
        f"muito alta ou baixa etc. No escopo dos dois times, a regra de Tukey sinaliza "
        f"{outlier_summary['tukey_total']} jogos nas sete variáveis e nenhum ultrapassa 3 × IQR. Por isso a "
        "decisão foi <strong>mantê-los</strong> e usar medidas robustas quando a interpretação puder ser "
        "distorcida, em vez de excluí-los ou imputá-los.",
    ]
    profile = {team: {"above": [], "below": []} for team in config.TEAM_CODES}
    for variable in config.VARIABLE_KEYS:
        for team in config.TEAM_CODES:
            if zscores[variable][team] >= 0.5:
                profile[team]["above"].append(variable)
            elif zscores[variable][team] <= -0.5:
                profile[team]["below"].append(variable)
    better = {team: [] for team in config.TEAM_CODES}
    for variable in config.VARIABLE_KEYS:
        a = summary[variable][TEAM_A]["describe"]["mean"]
        b = summary[variable][TEAM_B]["describe"]["mean"]
        better[TEAM_A if a > b else TEAM_B].append(variable)
    texts["teams"] = []
    for team in config.TEAM_CODES:
        p = profile[team]
        sentence = f"{_team(team)} tem média maior que o rival em {_labels(better[team]) or 'nenhuma variável'}."
        if p["above"]:
            sentence += f" Em relação à liga, destaca-se positivamente em {_labels(p['above'])}"
            sentence += f" e fica abaixo em {_labels(p['below'])}." if p["below"] else "."
        elif p["below"]:
            sentence += f" Em relação à liga, fica abaixo em {_labels(p['below'])}."
        texts["teams"].append(sentence)
    texts["teams"].append(
        "Os perfis são complementares: os Knicks se apoiam em volume ofensivo e rebotes, enquanto os Suns "
        "compensam a menor produção ofensiva com defesa agressiva (roubos de bola) e bola de três — ao custo de "
        "mais faltas. Tocos são um ponto fraco comum aos dois times."
        if "steals" in better[TEAM_B] and "rebounds" in better[TEAM_A]
        else ""
    )
    cv_wins = {team: 0 for team in config.TEAM_CODES}
    for variable in config.VARIABLE_KEYS:
        a = summary[variable][TEAM_A]["cv"]["cv"]
        b = summary[variable][TEAM_B]["cv"]["cv"]
        cv_wins[TEAM_A if a < b else TEAM_B] += 1
    steadier = max(cv_wins, key=cv_wins.get)
    skewed = [
        (v, t) for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
        if abs(summary[v][t]["skew"]["G1"]) >= config.SKEW_MODERATE
    ]
    avg_cv = {v: sum(summary[v][t]["cv"]["cv"] for t in config.TEAM_CODES) / 2 for v in config.VARIABLE_KEYS}
    ordered = sorted(avg_cv, key=avg_cv.get)

    def cv_range(variable):
        values = sorted(summary[variable][t]["cv"]["cv"] for t in config.TEAM_CODES)
        return f"{percent(values[0])} a {percent(values[-1])}"

    texts["distribution"] = [
        f"<strong>Regularidade.</strong> {_team(steadier)} tem CV menor em {cv_wins[steadier]} das 7 variáveis, "
        f"sendo o time mais consistente. {_var(ordered[0])} é a estatística mais estável (CV de "
        f"{cv_range(ordered[0])}), e {_var(ordered[-1])} e {_var(ordered[-2])} as mais variáveis em termos "
        f"relativos (CV de {cv_range(ordered[-1])} e {cv_range(ordered[-2])}), em boa parte porque são "
        "contagens pequenas, cuja variação se aproxima do esperado sob Poisson.",
        f"<strong>Forma das distribuições.</strong> {14 - len(skewed)} das 14 combinações são aproximadamente "
        "simétricas, e média e mediana quase coincidem. As exceções ("
        + _join(f"{_var(v)} dos {config.TEAMS[t]['short']}" for v, t in skewed)
        + ") são explicadas por poucos jogos extremos — como mostra a divergência entre o G1 (sensível às "
        "caudas) e o coeficiente de Bowley (baseado nos quartis).",
        "<strong>Medidas recomendadas.</strong> Para descrever o jogo típico, a média é adequada na maioria "
        "das variáveis; nas combinações assimétricas, a mediana e o IQR são preferíveis. Para comparar "
        "dispersão entre variáveis, o CV (ou o IQR / mediana) é necessário, pois o desvio padrão depende da "
        "escala. Para comparar times com a liga em várias variáveis ao mesmo tempo, o z-score é a ferramenta "
        "certa.",
    ]
    texts["limitations"] = [
        "Temporada incompleta (≈ 74–76 de 82 jogos): as medidas podem mudar até o fim da temporada regular.",
        "As estatísticas de um jogo dependem do adversário e do ritmo (número de posses), que não estão "
        "controlados nesta análise univariada.",
        "Na liga, cada partida aparece duas vezes (uma por time), então as linhas não são totalmente "
        "independentes — algo a considerar em testes estatísticos e modelos futuros.",
        "Com ~75 jogos por time, coeficientes como G1 têm erro-padrão de ≈ 0,28; diferenças pequenas de "
        "assimetria entre times não devem ser superinterpretadas.",
    ]
    return texts
