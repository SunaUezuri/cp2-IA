# 📊 Análise Preditiva: Redes Neurais vs. Modelos Clássicos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

**Um projeto comparativo de Machine Learning para avaliar o desempenho de Redes Neurais (Keras) contra o algoritmo Random Forest (Scikit-learn) em datasets tabulares.**

</div>

---

## 👥 Membros da Equipe

<div align="center">

| Nome | RM |
| :--- | :--- |
| **Wesley Sena dos Santos** | 558043 |
| **Rafael de Souza Pinto** | 555130 |
| **Samara Victoria Ferraz dos Santos** | 558719 |

</div>

---

## 📋 Índice

- [🎯 Visão Geral](#-visão-geral)
- [📁 Datasets Utilizados](#-datasets-utilizados)
- [🔧 Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [📊 Metodologia](#-metodologia)
- [🚀 Instalação e Uso](#-instalação-e-uso)
- [📈 Principais Resultados](#-principais-resultados)
- [🤖 Modelos de Machine Learning](#-modelos-de-machine-learning)
- [💡 Insights e Conclusões](#-insights-e-conclusões)
- [🔮 Próximos Passos](#-próximos-passos)
- [📚 Referências](#-referências)

---

## 🎯 Visão Geral

Este projeto realiza uma análise comparativa entre Redes Neurais, implementadas com a biblioteca Keras (TensorFlow), e o algoritmo Random Forest, uma técnica de ensemble consolidada do Scikit-learn. O objetivo é determinar qual abordagem oferece melhor desempenho em dois problemas distintos e clássicos de machine learning: uma tarefa de **classificação multiclasse** e uma de **regressão**.

### 🎯 Objetivos Principais

- **Implementar Redes Neurais**: Construir, compilar e treinar modelos de Redes Neurais para tarefas de classificação e regressão.
- **Treinar Modelos Clássicos**: Utilizar o `RandomForestClassifier` e `RandomForestRegressor` como benchmarks de performance.
- **Pré-processar Dados**: Aplicar técnicas de padronização (`StandardScaler`) e codificação (`OneHotEncoder`) para preparar os dados para os modelos.
- **Avaliar Performance**: Medir e comparar a acurácia dos modelos de classificação e o erro (MAE e RMSE) dos modelos de regressão.
- **Extrair Conclusões**: Discutir os resultados e analisar por que um modelo pode ter superado o outro em cada cenário.

---

## 📁 Datasets Utilizados

### 📊 Dataset 1: Wine Dataset (Classificação)

**Fonte**: Repositório UCI (`ucimlrepo`)

#### 📋 Descrição
Este é um dataset clássico para tarefas de classificação, onde o objetivo é prever a qual de três diferentes produtores um vinho pertence, com base em 13 atributos químicos.

#### 📈 Características dos Dados
- **Total de Registros**: 178 observações
- **Dimensões**: 178 × 13 variáveis de entrada (features)
- **Classes**: 3 classes de vinhos

#### 🔧 Variáveis do Dataset

| Variável | Tipo | Descrição |
|:---|:---|:---|
| `Alcohol` | `float64` | Teor alcoólico |
| `Malicacid` | `float64` | Ácido málico |
| `Ash` | `float64` | Cinzas |
| `Alcalinity_of_ash` | `float64` | Alcalinidade das cinzas |
| `Magnesium` | `int64` | Magnésio |
| `Total_phenols` | `float64` | Fenóis totais |
| `Flavanoids` | `float64` | Flavonoides |
| `Nonflavanoid_phenols` | `float64` | Fenóis não flavonoides |
| `Proanthocyanins` | `float64` | Proantocianinas |
| `Color_intensity` | `float64` | Intensidade da cor |
| `Hue` | `float64` | Tonalidade |
| `0D280_0D315_of_diluted_wines` | `float64` | Medida de diluição |
| `Proline` | `int64` | Prolina |

---

### 📊 Dataset 2: California Housing (Regressão)

**Fonte**: `sklearn.datasets`

#### 📋 Descrição
Este dataset contém dados do censo de 1990 da Califórnia. A tarefa é prever o valor mediano das casas em um distrito com base em 8 variáveis demográficas e geográficas.

#### 📈 Características dos Dados
- **Total de Registros**: 20.640 observações
- **Dimensões**: 20.640 × 8 variáveis de entrada (features)
- **Alvo**: Valor mediano das casas (variável contínua)

#### 🔧 Variáveis do Dataset

| Categoria | Variáveis | Descrição |
|:---|:---|:---|
| **Renda** | `MedInc` | Renda mediana no distrito |
| **Idade** | `HouseAge` | Idade mediana das casas |
| **Estrutura** | `AveRooms` | Média de cômodos por domicílio |
| | `AveBedrms` | Média de quartos por domicílio |
| **Demografia** | `Population` | População do distrito |
| | `AveOccup` | Média de ocupantes por domicílio |
| **Geografia** | `Latitude` | Latitude do distrito |
| | `Longitude` | Longitude do distrito |

---

## 🔧 Tecnologias Utilizadas

### 🐍 Linguagens e Frameworks
- **Python 3.8+**: Linguagem principal
- **Jupyter Notebook**: Ambiente de desenvolvimento
- **Google Colab**: Plataforma de execução em nuvem

### 📊 Bibliotecas de Análise e ML
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **ucimlrepo**: Para carregar o dataset de vinhos
- **TensorFlow / Keras**: Para construção e treinamento das Redes Neurais (`Sequential`, `Dense`)
- **Scikit-learn**: Para pré-processamento e modelos de Machine Learning clássicos
  - `StandardScaler`, `OneHotEncoder`, `train_test_split`
  - `RandomForestClassifier`, `RandomForestRegressor`
  - `accuracy_score`, `mean_absolute_error`, `mean_squared_error`

---

## 🤖 Modelos de Machine Learning

### 📊 1. Experimento de Classificação (Wine Dataset)

#### 🎯 Objetivo
Classificar vinhos em uma de três categorias com base em suas características químicas.

#### 🧠 Modelo de Rede Neural
- **Arquitetura**: `(13) -> Dense(32, relu) -> Dense(32, relu) -> Dense(3, softmax)`
- **Otimizador**: `Adam`
- **Função de Perda**: `categorical_crossentropy`
- **Resultado (Acurácia)**: `97.22%`

#### 🌳 Modelo de Comparação (Random Forest Classifier)
- **Hiperparâmetros**: `n_estimators=100`, `random_state=42`
- **Resultado (Acurácia)**: `100.00%`

#### 💡 Interpretação
No problema de classificação de vinhos, o Random Forest atingiu a acurácia perfeita de 100%, superando a rede neural. Isso sugere que, para este dataset pequeno e bem-estruturado, a capacidade do Random Forest de criar múltiplas árvores de decisão e combinar seus resultados foi extremamente eficaz.

### 📊 2. Experimento de Regressão (California Housing Dataset)

#### 🎯 Objetivo
Prever o valor mediano das casas na Califórnia.

#### 🧠 Modelo de Rede Neural
- **Arquitetura**: `(8) -> Dense(64, relu) -> Dense(32, relu) -> Dense(16, relu) -> Dense(1, linear)`
- **Otimizador**: `Adam`
- **Função de Perda**: `Mean Squared Error (mse)`
- **Resultados**:
  - **MAE**: `0.3556`
  - **RMSE**: `0.5299`

#### 🌳 Modelo de Comparação (Random Forest Regressor)
- **Hiperparâmetros**: `n_estimators=100`, `random_state=42`
- **Resultados**:
  - **MAE**: `0.3274`
  - **RMSE**: `0.5051`

#### 💡 Interpretação
Assim como no caso anterior, o Random Forest Regressor apresentou um desempenho superior, com um erro (RMSE) menor em comparação com a rede neural. O resultado indica que para dados tabulares como os deste problema, algoritmos de ensemble baseados em árvores continuam sendo uma escolha muito forte e, muitas vezes, mais eficaz do que redes neurais mais complexas sem um ajuste fino extensivo.

---

## 💡 Insights e Conclusões

### 🎯 Principais Descobertas

- **Superioridade do Random Forest em Dados Tabulares**: Para os dois datasets analisados, que são exemplos clássicos de dados tabulares, o Random Forest (tanto para classificação quanto para regressão) superou as Redes Neurais implementadas.
- **Simplicidade e Eficácia**: O Random Forest é um modelo mais simples de treinar, exigindo menos pré-processamento (não precisa de codificação *one-hot* para o alvo na classificação) e menos ajuste de hiperparâmetros para alcançar um resultado de alta performance.
- **Potencial das Redes Neurais**: Embora superadas neste contexto, as Redes Neurais têm um grande potencial. Seu desempenho poderia ser melhorado com mais dados, uma arquitetura mais complexa ou um ajuste fino de hiperparâmetros (como taxa de aprendizado, número de épocas, e estrutura das camadas).

### 📊 Limitações
- **Datasets Pequenos**: Os datasets utilizados, especialmente o de vinhos, são relativamente pequenos. Redes Neurais geralmente brilham com volumes de dados muito maiores.
- **Ajuste de Hiperparâmetros**: Os modelos de Redes Neurais foram construídos com arquiteturas padrão e não passaram por um processo de otimização de hiperparâmetros, o que poderia melhorar significativamente seus resultados.
- **Ausência de Visualizações**: O projeto foca puramente na comparação de métricas de desempenho, sem incluir análises exploratórias visuais que poderiam fornecer mais insights sobre os dados.

---

## 🔮 Próximos Passos

- **Otimização de Hiperparâmetros**: Utilizar técnicas como Grid Search ou Random Search para encontrar a melhor combinação de hiperparâmetros para as Redes Neurais.
- **Análise Visual dos Dados (EDA)**: Criar gráficos e visualizações para entender melhor a distribuição e as correlações presentes nos datasets.
- **Experimentar Outras Arquiteturas**: Testar diferentes arquiteturas de Redes Neurais, como adicionar mais camadas, usar diferentes funções de ativação ou incluir camadas de regularização (Dropout).
- **Comparar com Outros Modelos**: Incluir outros algoritmos de machine learning na comparação, como Gradient Boosting (XGBoost, LightGBM) e Support Vector Machines (SVM).

---

## 📚 Referências

### 🔗 Datasets
- **Wine Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)
- **California Housing Dataset**: Incluído na biblioteca `sklearn.datasets`.

### 📚 Bibliotecas e Ferramentas
- **Pandas**: [Documentação Oficial](https://pandas.pydata.org/docs/)
- **Scikit-learn**: [Documentação Oficial](https://scikit-learn.org/stable/)
- **TensorFlow/Keras**: [Documentação Oficial](https://www.tensorflow.org/guide)
- **UCIMLRepo**: [PyPI Project](https://pypi.org/project/ucimlrepo/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
