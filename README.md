# Análise Exploratória e Classificação de Vinhos

> **Aluno(a):** Arthur Queiroz Uchôa
> **Disciplina:** Machine Learning 
> **Dataset:** [Wine Quality — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Visão geral

Este repositório contém uma análise exploratória dos vinhos tintos e brancos do conjunto **Wine Quality** da UCI e a comparação dos classificadores **KNN** e **Random Forest** para prever o tipo do vinho usando apenas atributos físico-químicos.

Esta página é o relatório completo: gráficos, hipóteses, resultados e código estão visíveis diretamente no GitHub. O arquivo [analise_wine_quality_tipo_vinho.ipynb](analise_wine_quality_tipo_vinho.ipynb) é mantido como versão executável e reproduzível da análise.

| Item | Decisão adotada |
|---|---|
| Variável-alvo | `tipo_vinho`: criada a partir do arquivo de origem (tinto ou branco) |
| Registros após limpeza | 5.320 |
| Features dos modelos | 11 medidas físico-químicas |
| Variável excluída | `quality`, por ser nota sensorial e não medida química |
| Separação dos dados | 80% treino / 20% teste, com estratificação |
| Melhor modelo | Random Forest |
| Melhor acurácia no teste | **99,53%** |

---

## 1. Perguntas e hipóteses

1. **H1 — Acidez volátil:** vinhos tintos apresentam maior acidez volátil média que vinhos brancos?
2. **H2 — Açúcar e SO₂:** vinhos brancos possuem maior açúcar residual e maior dióxido de enxofre total?
3. **H3 — Qualidade:** a proporção de vinhos com nota `quality ≥ 6` difere entre os dois tipos?
4. **H4 — Predição:** as 11 medidas químicas distinguem vinho tinto de branco com acurácia acima de 85% em dados nunca vistos?

---

## 2. Obtenção, criação do alvo e limpeza

O dataset é composto por dois CSVs oficiais da UCI: um para vinho tinto e outro para vinho branco. Portanto, a variável-alvo é criada de modo transparente a partir da procedência de cada registro. Como há linhas inteiramente repetidas, elas são removidas antes da divisão entre treino e teste para impedir que observações idênticas inflem artificialmente a avaliação.

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use("seaborn-v0_8-whitegrid")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
red_path = DATA_DIR / "winequality-red.csv"
white_path = DATA_DIR / "winequality-white.csv"
source_url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

if not (red_path.exists() and white_path.exists()):
    zip_path = DATA_DIR / "wine_quality_uci.zip"
    urlretrieve(source_url, zip_path)
    with ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)

red = pd.read_csv(red_path, sep=";")
white = pd.read_csv(white_path, sep=";")
red["tipo_vinho"] = "tinto"
white["tipo_vinho"] = "branco"
df_original = pd.concat([red, white], ignore_index=True)

print(df_original.shape)       # (6497, 13)
print(df_original.duplicated().sum())  # 1177
df = df_original.drop_duplicates().copy()
```

### Resultado da limpeza

| Tipo | Observações após remoção de duplicatas | Participação |
|---|---:|---:|
| Tinto | 1.359 | 25,55% |
| Branco | 3.961 | 74,45% |
| **Total** | **5.320** | **100%** |

Não existem valores ausentes. Os outliers não foram removidos automaticamente porque podem representar características reais de vinhos e sua remoção exigiria conhecimento de domínio.

```python
auditoria = pd.DataFrame({
    "tipo": df_original.dtypes.astype(str),
    "ausentes": df_original.isna().sum(),
    "unicos": df_original.nunique(),
})
display(auditoria)

resumo = df.groupby("tipo_vinho").agg(["mean", "median", "std", "min", "max"])
display(resumo.T.round(3))
```

---

## 3. Análise exploratória

### H1 — Acidez volátil

<img src="assets/01_acidez_volatil.svg" width="100%" alt="Gráfico de acidez volátil média por tipo de vinho">

**Resposta:** hipótese sustentada. A média de acidez volátil nos tintos é **0,529**, contra **0,281** nos brancos; uma diferença de **+0,248**.

```python
h1 = df.groupby("tipo_vinho")["volatile acidity"].agg(
    ["count", "mean", "median", "std"]
)
display(h1.round(3))

fig, ax = plt.subplots(figsize=(8, 4.5))
ordem = ["tinto", "branco"]
ax.boxplot(
    [df.loc[df.tipo_vinho.eq(t), "volatile acidity"] for t in ordem],
    tick_labels=ordem,
    patch_artist=True,
    boxprops={"facecolor": "#B03A2E"},
    medianprops={"color": "black", "linewidth": 2},
)
ax.set_title("H1 — Acidez volátil por tipo de vinho")
ax.set_xlabel("Tipo de vinho")
ax.set_ylabel("Acidez volátil")
plt.show()
```

### H2 — Açúcar residual e dióxido de enxofre total

<img src="assets/02_acucar_so2.svg" width="100%" alt="Gráficos de açúcar residual e dióxido de enxofre total">

**Resposta:** hipótese sustentada. As medianas dos brancos são maiores em ambos os indicadores:

| Variável | Tinto | Branco | Diferença branco − tinto |
|---|---:|---:|---:|
| Açúcar residual | 2,2 | 4,7 | +2,5 |
| Dióxido de enxofre total | 38 | 133 | +95 |

```python
colunas_h2 = ["residual sugar", "total sulfur dioxide"]
h2 = df.groupby("tipo_vinho")[colunas_h2].agg(["mean", "median"])
display(h2.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cores = ["#B03A2E", "#F4D03F"]
for ax, coluna in zip(axes, colunas_h2):
    medianas = (
        df.groupby("tipo_vinho")[coluna]
          .median()
          .reindex(["tinto", "branco"])
    )
    barras = ax.bar(medianas.index, medianas.values, color=cores)
    ax.bar_label(barras, labels=[f"{v:.1f}" for v in medianas.values], padding=3)
    ax.set_title(f"H2 — Mediana de {coluna}")
    ax.set_ylabel(coluna)
plt.tight_layout()
plt.show()
```

### H3 — Distribuição de qualidade

<img src="assets/03_quality.svg" width="100%" alt="Percentual de vinhos com qualidade maior ou igual a seis">

**Resposta:** a proporção de `quality ≥ 6` é maior nos brancos (**65,97%**) que nos tintos (**52,91%**). Esta conclusão é apenas descritiva. A coluna `quality` foi propositalmente retirada das entradas para que o modelo se baseie exclusivamente em química.

```python
tabela_quality = pd.crosstab(
    df["quality"], df["tipo_vinho"], normalize="columns"
).mul(100)
display(tabela_quality.round(2))

fig, ax = plt.subplots(figsize=(9, 5))
tabela_quality.plot(kind="bar", ax=ax, color=["#F4D03F", "#B03A2E"])
ax.set_title("H3 — Distribuição percentual da nota quality por tipo")
ax.set_xlabel("Nota quality")
ax.set_ylabel("Percentual dentro do tipo (%)")
ax.legend(title="Tipo")
plt.xticks(rotation=0)
plt.show()

prop_boa = (
    df.assign(qualidade_ge_6=df["quality"].ge(6))
      .groupby("tipo_vinho")["qualidade_ge_6"]
      .mean()
      .mul(100)
)
print(prop_boa)
```

### Correlação entre variáveis

O mapa de correlação é útil para identificar relações lineares. Ele não é usado para concluir causalidade: duas variáveis podem variar juntas por outros fatores do processo de produção.

```python
variaveis_numericas = df.select_dtypes(include="number").columns.tolist()
corr = df[variaveis_numericas].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
imagem = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
ax.set_title("Correlação de Pearson entre variáveis numéricas")
fig.colorbar(imagem, ax=ax, label="Correlação")
plt.tight_layout()
plt.show()

pares = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=np.abs, ascending=False)
        .head(8)
)
display(pares.rename("correlação").to_frame().round(3))
```

---

## 4. Preparação dos dados

As 11 variáveis químicas são usadas como atributos preditores. A acurácia é avaliada em um conjunto de teste que não participou nem do treinamento nem da escolha de hiperparâmetros.

Usei **80% dos dados para treino e 20% para teste**, com `stratify=y`. Esta escolha oferece um volume robusto para treinamento e ainda reserva uma amostra independente de 1.064 vinhos para medir generalização. A estratificação é importante porque a base contém mais vinhos brancos que tintos.

```python
feature_cols = [
    coluna for coluna in df.columns
    if coluna not in ["tipo_vinho", "quality"]
]
X = df[feature_cols].copy()

encoder = LabelEncoder()
y = encoder.fit_transform(df["tipo_vinho"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Treino: {X_train.shape[0]} observações")  # 4.256
print(f"Teste: {X_test.shape[0]} observações")    # 1.064
print(feature_cols)
```

---

## 5. Modelos de Machine Learning

### KNN

O KNN decide pelo voto dos vizinhos mais próximos. Como esse método calcula distâncias, o `StandardScaler` padroniza as escalas dentro de um `Pipeline`, evitando vazamento de informação do teste. Os valores de `k` são selecionados somente com validação cruzada sobre o treino.

```python
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grade_knn = {
    "knn__n_neighbors": [5, 9, 15, 21],
    "knn__weights": ["distance"],
    "knn__p": [2],  # distância Euclidiana
}

busca_knn = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=grade_knn,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    return_train_score=False,
)
busca_knn.fit(X_train, y_train)

print(busca_knn.best_params_)
# {'knn__n_neighbors': 5, 'knn__p': 2, 'knn__weights': 'distance'}
print(busca_knn.best_score_)  # 0.9927
```

### Random Forest

O Random Forest combina diversas árvores de decisão, sendo adequado para relações não lineares e interações entre medidas químicas. Ao contrário do KNN, árvores tomam decisões por limiares e não dependem da escala das variáveis.

```python
random_forest = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)
random_forest.fit(X_train, y_train)
```

---

## 6. Comparação de resultados

<img src="assets/04_metricas_modelos.svg" width="100%" alt="Comparação das métricas KNN e Random Forest">

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN otimizado por validação cruzada | 99,25% | 99,13% | 98,89% | 99,01% | 99,19% |
| **Random Forest** | **99,53%** | **99,56%** | **99,20%** | **99,38%** | **99,95%** |

Os dois modelos excedem com ampla margem a exigência de acurácia superior a 85%. O Random Forest é o vencedor por apresentar acurácia e F1 macro maiores.

```python
modelos = {
    "KNN (otimizado por CV)": busca_knn.best_estimator_,
    "Random Forest": random_forest,
}
resultados, predicoes = [], {}

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    predicoes[nome] = y_pred
    resultados.append({
        "modelo": nome,
        "acurácia": accuracy_score(y_test, y_pred),
        "precisão_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

tabela_resultados = (
    pd.DataFrame(resultados)
      .sort_values(["acurácia", "f1_macro"], ascending=False)
      .reset_index(drop=True)
)
display(tabela_resultados)
assert (tabela_resultados["acurácia"] > 0.85).all()
```

### Matriz de confusão do melhor modelo

<img src="assets/05_matriz_confusao_rf.svg" width="100%" alt="Matriz de confusão do Random Forest">

O melhor modelo acertou **1.059 das 1.064** observações de teste. A matriz deixa claro que o desempenho é alto em ambas as classes, e não apenas na classe majoritária (vinhos brancos).

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (nome, y_pred) in zip(axes, predicoes.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=encoder.classes_,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(nome)
plt.tight_layout()
plt.show()

print(classification_report(
    y_test,
    predicoes["Random Forest"],
    target_names=encoder.classes_,
))
```

### Importância das variáveis

```python
importancia = (
    pd.Series(random_forest.feature_importances_, index=feature_cols)
      .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(8, 5))
importancia.tail(10).plot(kind="barh", ax=ax, color="#2874A6")
ax.set_title("10 maiores importâncias — Random Forest")
ax.set_xlabel("Importância por redução de impureza")
ax.set_ylabel("Variável físico-química")
plt.tight_layout()
plt.show()
```

---

## 7. Salvamento do melhor modelo

O último passo escolhe o modelo pela maior acurácia — usando F1 macro como desempate — e armazena o estimador, a lista de colunas, as classes e a configuração de reprodutibilidade em um arquivo `.joblib`.

```python
ranking = tabela_resultados.sort_values(
    ["acurácia", "f1_macro"], ascending=False
)
melhor_nome = ranking.iloc[0]["modelo"]
melhor_modelo = modelos[melhor_nome]

caminho_modelo = Path("melhor_modelo_tipo_vinho.joblib")
pacote_modelo = {
    "model": melhor_modelo,
    "feature_names": feature_cols,
    "target_classes": encoder.classes_.tolist(),
    "test_accuracy": float(ranking.iloc[0]["acurácia"]),
    "dataset_source": source_url,
    "random_state": RANDOM_STATE,
}
joblib.dump(pacote_modelo, caminho_modelo)

print(f"Melhor modelo: {melhor_nome}")
print(f"Acurácia de teste: {ranking.iloc[0]['acurácia']:.2%}")
```

---

## 8. Discussão e conclusão

- As diferenças em **acidez volátil**, **açúcar residual** e **dióxido de enxofre total** sustentam as hipóteses exploratórias e explicam por que os tipos de vinho são separáveis.
- KNN teve desempenho excelente após a padronização, o que sugere que as classes ocupam regiões distintas no espaço das medidas químicas.
- Random Forest obteve o melhor resultado. A vantagem indica que interações e limites não lineares entre variáveis também ajudam a distinguir os vinhos.
- Acurácia não foi a única métrica analisada: precisão, recall, F1 macro, ROC-AUC e a matriz de confusão verificam desempenho para tintos e brancos.
- Importância de variável indica associação útil para a predição, **não causalidade**.

### Limitações

O conjunto representa vinhos verdes portugueses; os resultados não devem ser generalizados automaticamente para vinhos de todos os países, uvas e métodos de produção. Além disso, as medições são laboratoriais e não substituem uma avaliação completa do processo produtivo.

