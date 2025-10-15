# 🤖 Projetos de Inteligência Artificial - CP2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

**Coleção de projetos de IA explorando diferentes abordagens: Detecção de Objetos com YOLO, Reconhecimento de Mãos com MediaPipe e Análise Preditiva com Redes Neurais.**

</div>

---

## 👥 Membros da Equipe

<div align="center">

| Nome                       | RM     |
| :------------------------- | :----- |
| **Wesley Sena dos Santos** | 558043 |
| **Rafael de Souza Pinto**  | 555130 |

</div>

---

## 📋 Índice

- [🎯 Visão Geral](#-visão-geral)
- [🤖 Projetos](#-projetos)
  - [🎮 PokeAi - Detecção de Pokémon](#-pokeai---detecção-de-pokémon)
  - [👋 Reconhecimento de Mãos](#-reconhecimento-de-mãos)
  - [📊 Análise Preditiva](#-análise-preditiva)
- [⚖️ Comparativo entre Abordagens](#️-comparativo-entre-abordagens)
- [🔧 Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [🚀 Instalação e Uso](#-instalação-e-uso)
- [💡 Insights e Conclusões](#-insights-e-conclusões)
- [📚 Referências](#-referências)

---

## 🎯 Visão Geral

Esta coleção de projetos explora diferentes abordagens de Inteligência Artificial, demonstrando a versatilidade e aplicações práticas de técnicas modernas de Machine Learning e Computer Vision. Os projetos abrangem desde **detecção de objetos** com YOLO até **reconhecimento de gestos** com MediaPipe, culminando em uma **análise comparativa** de modelos preditivos.

### 🎯 Objetivos Principais

- **Explorar Computer Vision**: Implementar detecção de objetos (Pokémon) e reconhecimento de mãos em tempo real
- **Comparar Abordagens**: Avaliar diferentes frameworks (YOLO vs MediaPipe) para tarefas de visão computacional
- **Análise Preditiva**: Comparar Redes Neurais com modelos clássicos em datasets tabulares
- **Aplicações Práticas**: Desenvolver soluções funcionais com interface interativa
- **Documentação Completa**: Fornecer guias detalhados de instalação e uso para cada projeto

---

## 🤖 Projetos

### 🎮 PokeAi - Detecção de Pokémon

**Objetivo**: Desenvolver um sistema de detecção e reconhecimento de Pokémon usando YOLO (You Only Look Once) para classificação de imagens em tempo real.

#### 🎯 Características

- ✅ Detecção de múltiplos Pokémon simultaneamente
- ✅ Modelo YOLO treinado com dataset personalizado do Roboflow
- ✅ Interface interativa para teste de imagens
- ✅ Processamento em tempo real com alta precisão
- ✅ Exportação de resultados com bounding boxes

#### 🔧 Ferramentas Utilizadas

- **YOLO (Ultralytics)**: Framework de detecção de objetos em tempo real
- **Roboflow**: Plataforma para geração e anotação de datasets
- **OpenCV**: Processamento de imagens e visualização
- **Google Colab**: Ambiente de desenvolvimento e treinamento

#### 📊 Metodologia

1. **Coleta de Dados**: Dataset de imagens de Pokémon anotadas
2. **Treinamento**: Modelo YOLO personalizado para reconhecimento
3. **Validação**: Testes com imagens não vistas durante o treinamento
4. **Deploy**: Modelo otimizado para inferência em tempo real

---

### 👋 Reconhecimento de Mãos

**Objetivo**: Implementar um sistema completo de reconhecimento e rastreamento de mãos em tempo real para controle por gestos e interação humano-computador.

#### 🎯 Características

- ✅ Detecção de até 2 mãos simultaneamente
- ✅ Rastreamento de 21 landmarks por mão
- ✅ Contagem de dedos levantados
- ✅ Identificação de mão direita/esquerda
- ✅ Visualização em tempo real via webcam
- ✅ Cálculo de FPS em tempo real
- ✅ Interface interativa com controles de teclado

#### 🔧 Ferramentas Utilizadas

- **MediaPipe**: Framework de ML do Google para detecção de mãos
- **OpenCV**: Processamento de imagens e captura de vídeo
- **NumPy**: Manipulação de arrays numéricos para cálculos

#### 📊 Metodologia

1. **Detecção**: Algoritmo MediaPipe para localização de mãos
2. **Rastreamento**: Seguimento de landmarks em tempo real
3. **Análise de Gestos**: Algoritmo customizado para contagem de dedos
4. **Interface**: Aplicação desktop com visualização interativa

---

### 📊 Análise Preditiva

**Objetivo**: Realizar uma análise comparativa entre Redes Neurais (Keras/TensorFlow) e modelos clássicos (Random Forest) em datasets tabulares para classificação e regressão.

#### 🎯 Características

- ✅ Comparação direta entre Redes Neurais e Random Forest
- ✅ Análise em dois tipos de problema: classificação e regressão
- ✅ Pré-processamento completo dos dados
- ✅ Métricas de avaliação padronizadas
- ✅ Análise de performance detalhada

#### 🔧 Ferramentas Utilizadas

- **TensorFlow/Keras**: Redes Neurais para classificação e regressão
- **Scikit-learn**: Random Forest e pré-processamento de dados
- **Pandas/NumPy**: Manipulação e análise de dados
- **Jupyter Notebook**: Ambiente de desenvolvimento e análise

#### 📁 Datasets Utilizados

### 📊 Dataset 1: Wine Dataset (Classificação)

**Fonte**: Repositório UCI (`ucimlrepo`)

#### 📋 Descrição

Este é um dataset clássico para tarefas de classificação, onde o objetivo é prever a qual de três diferentes produtores um vinho pertence, com base em 13 atributos químicos.

#### 📈 Características dos Dados

- **Total de Registros**: 178 observações
- **Dimensões**: 178 × 13 variáveis de entrada (features)
- **Classes**: 3 classes de vinhos

#### 🔧 Variáveis do Dataset

| Variável                       | Tipo      | Descrição               |
| :----------------------------- | :-------- | :---------------------- |
| `Alcohol`                      | `float64` | Teor alcoólico          |
| `Malicacid`                    | `float64` | Ácido málico            |
| `Ash`                          | `float64` | Cinzas                  |
| `Alcalinity_of_ash`            | `float64` | Alcalinidade das cinzas |
| `Magnesium`                    | `int64`   | Magnésio                |
| `Total_phenols`                | `float64` | Fenóis totais           |
| `Flavanoids`                   | `float64` | Flavonoides             |
| `Nonflavanoid_phenols`         | `float64` | Fenóis não flavonoides  |
| `Proanthocyanins`              | `float64` | Proantocianinas         |
| `Color_intensity`              | `float64` | Intensidade da cor      |
| `Hue`                          | `float64` | Tonalidade              |
| `0D280_0D315_of_diluted_wines` | `float64` | Medida de diluição      |
| `Proline`                      | `int64`   | Prolina                 |

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

| Categoria      | Variáveis    | Descrição                        |
| :------------- | :----------- | :------------------------------- |
| **Renda**      | `MedInc`     | Renda mediana no distrito        |
| **Idade**      | `HouseAge`   | Idade mediana das casas          |
| **Estrutura**  | `AveRooms`   | Média de cômodos por domicílio   |
|                | `AveBedrms`  | Média de quartos por domicílio   |
| **Demografia** | `Population` | População do distrito            |
|                | `AveOccup`   | Média de ocupantes por domicílio |
| **Geografia**  | `Latitude`   | Latitude do distrito             |
|                | `Longitude`  | Longitude do distrito            |

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

## ⚖️ Comparativo entre Abordagens

### 🎯 YOLO vs MediaPipe: Visão Computacional

| Aspecto           | YOLO (PokeAi)                     | MediaPipe (Reconhecimento de Mãos)  |
| :---------------- | :-------------------------------- | :---------------------------------- |
| **Objetivo**      | Detecção de objetos específicos   | Rastreamento de landmarks corporais |
| **Precisão**      | Alta para objetos treinados       | Muito alta para estruturas humanas  |
| **Velocidade**    | Rápida (tempo real)               | Muito rápida (otimizada)            |
| **Treinamento**   | Requer dataset anotado            | Pré-treinado pelo Google            |
| **Flexibilidade** | Customizável para qualquer objeto | Específico para estruturas humanas  |
| **Recursos**      | GPU recomendada                   | CPU suficiente                      |
| **Aplicações**    | Detecção geral de objetos         | Interação humano-computador         |

### 📊 Redes Neurais vs Random Forest: Análise Preditiva

| Aspecto                  | Redes Neurais                    | Random Forest                 |
| :----------------------- | :------------------------------- | :---------------------------- |
| **Complexidade**         | Alta (muitos hiperparâmetros)    | Baixa (poucos parâmetros)     |
| **Interpretabilidade**   | Baixa (caixa preta)              | Média (feature importance)    |
| **Dados Necessários**    | Grandes volumes                  | Pequenos a médios volumes     |
| **Tempo de Treinamento** | Longo                            | Rápido                        |
| **Overfitting**          | Propenso                         | Menos propenso                |
| **Performance**          | Variável (depende do tuning)     | Consistente                   |
| **Uso Recomendado**      | Dados complexos, grandes volumes | Dados tabulares, prototipagem |

### 💡 Insights das Comparações

#### 🎮 **Computer Vision**

- **YOLO**: Excelente para detecção de objetos personalizados, mas requer preparação de dados e treinamento
- **MediaPipe**: Ideal para aplicações de interação humano-computador, com setup mínimo e alta precisão
- **Escolha**: YOLO para objetos específicos, MediaPipe para estruturas corporais

#### 📊 **Machine Learning Tradicional**

- **Redes Neurais**: Poderosas mas complexas, ideais para dados não-lineares complexos
- **Random Forest**: Simples e eficazes, excelentes para dados tabulares e prototipagem rápida
- **Escolha**: Random Forest para dados tabulares simples, Redes Neurais para problemas complexos

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
- **Simplicidade e Eficácia**: O Random Forest é um modelo mais simples de treinar, exigindo menos pré-processamento (não precisa de codificação _one-hot_ para o alvo na classificação) e menos ajuste de hiperparâmetros para alcançar um resultado de alta performance.
- **Potencial das Redes Neurais**: Embora superadas neste contexto, as Redes Neurais têm um grande potencial. Seu desempenho poderia ser melhorado com mais dados, uma arquitetura mais complexa ou um ajuste fino de hiperparâmetros (como taxa de aprendizado, número de épocas, e estrutura das camadas).

### 📊 Limitações

- **Datasets Pequenos**: Os datasets utilizados, especialmente o de vinhos, são relativamente pequenos. Redes Neurais geralmente brilham com volumes de dados muito maiores.
- **Ajuste de Hiperparâmetros**: Os modelos de Redes Neurais foram construídos com arquiteturas padrão e não passaram por um processo de otimização de hiperparâmetros, o que poderia melhorar significativamente seus resultados.
- **Ausência de Visualizações**: O projeto foca puramente na comparação de métricas de desempenho, sem incluir análises exploratórias visuais que poderiam fornecer mais insights sobre os dados.

---

## 🚀 Instalação e Uso

### 📋 Pré-requisitos

- Python 3.8 ou superior
- Webcam (para reconhecimento de mãos)
- Jupyter Notebook ou Google Colab
- Conexão com internet (para download de dependências)

### 🔧 Instalação por Projeto

#### 🎮 PokeAi - Detecção de Pokémon

```bash
# Navegar para o diretório
cd PokeAi

# Instalar dependências
pip install roboflow ultralytics opencv-python

# Executar no Google Colab
# Abrir PokeAi.ipynb no Google Colab
```

#### 👋 Reconhecimento de Mãos

```bash
# Navegar para o diretório
cd reconhecimento_de_mao

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows)
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt

# Executar o programa
python hand_recognition.py
```

#### 📊 Análise Preditiva

```bash
# Navegar para o diretório
cd RedeNeural

# Instalar dependências
pip install pandas numpy scikit-learn tensorflow ucimlrepo jupyter

# Executar o notebook
jupyter notebook cp2Ia.ipynb
```

### 🎮 Como Usar Cada Projeto

#### 🎮 PokeAi

1. **Treinamento**: Execute todas as células no Google Colab
2. **Teste**: Carregue uma imagem de Pokémon para detecção
3. **Resultado**: Visualize bounding boxes com confiança

#### 👋 Reconhecimento de Mãos

1. **Execução**: Execute o script Python
2. **Controles**:
   - `Q`: Sair do programa
   - `H`: Alternar visualização dos landmarks
3. **Funcionalidades**: Contagem de dedos, identificação de mão

#### 📊 Análise Preditiva

1. **Classificação**: Execute seção de vinhos
2. **Regressão**: Execute seção de casas da Califórnia
3. **Comparação**: Analise métricas de ambos os modelos

### 📈 Interpretando os Resultados

#### 🎮 PokeAi

- **Confiança**: Probabilidade de detecção correta
- **Bounding Box**: Localização do Pokémon na imagem
- **Classe**: Tipo de Pokémon detectado

#### 👋 Reconhecimento de Mãos

- **Landmarks**: 21 pontos de referência por mão
- **Dedos**: Status de cada dedo (0=fechado, 1=levantado)
- **FPS**: Taxa de processamento em tempo real

#### 📊 Análise Preditiva

- **Acurácia**: Percentual de predições corretas (classificação)
- **MAE**: Erro absoluto médio (regressão)
- **RMSE**: Raiz do erro quadrático médio (regressão)

---

## 🔮 Próximos Passos

### 🎮 PokeAi

- **Expansão do Dataset**: Adicionar mais espécies de Pokémon para melhorar a diversidade
- **Otimização do Modelo**: Implementar técnicas de data augmentation e fine-tuning
- **Interface Web**: Desenvolver uma aplicação web para upload e detecção de imagens
- **Real-time Detection**: Implementar detecção em tempo real via webcam

### 👋 Reconhecimento de Mãos

- **Reconhecimento de Gestos**: Implementar reconhecimento de gestos específicos (como "OK", "Peace")
- **Controle de Aplicações**: Integrar com aplicações para controle por gestos
- **Multi-mãos**: Melhorar detecção para mais de 2 mãos simultaneamente
- **Interface Gráfica**: Criar interface gráfica mais amigável

### 📊 Análise Preditiva

- **Otimização de Hiperparâmetros**: Utilizar técnicas como Grid Search ou Random Search
- **Análise Visual dos Dados (EDA)**: Criar gráficos e visualizações exploratórias
- **Experimentar Outras Arquiteturas**: Testar diferentes arquiteturas de Redes Neurais
- **Comparar com Outros Modelos**: Incluir XGBoost, LightGBM e SVM na comparação

### 🚀 Integração entre Projetos

- **Sistema Unificado**: Criar uma interface que combine todos os projetos
- **API REST**: Desenvolver APIs para integração com outras aplicações
- **Dashboard**: Interface web para monitoramento e análise de todos os modelos
- **Mobile App**: Desenvolver aplicativo mobile para reconhecimento de mãos

---

## 📚 Referências

### 🔗 Datasets

- **Wine Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)
- **California Housing Dataset**: Incluído na biblioteca `sklearn.datasets`
- **Pokémon Dataset**: Dataset personalizado via Roboflow

### 📚 Bibliotecas e Ferramentas

#### 🎮 Computer Vision

- **YOLO/Ultralytics**: [Documentação Oficial](https://docs.ultralytics.com/)
- **Roboflow**: [Plataforma de Datasets](https://roboflow.com/)
- **MediaPipe**: [Documentação Oficial](https://mediapipe.dev/)
- **OpenCV**: [Documentação Oficial](https://docs.opencv.org/)

#### 📊 Machine Learning

- **Pandas**: [Documentação Oficial](https://pandas.pydata.org/docs/)
- **Scikit-learn**: [Documentação Oficial](https://scikit-learn.org/stable/)
- **TensorFlow/Keras**: [Documentação Oficial](https://www.tensorflow.org/guide)
- **UCIMLRepo**: [PyPI Project](https://pypi.org/project/ucimlrepo/)

#### 🛠️ Desenvolvimento

- **Jupyter Notebook**: [Documentação Oficial](https://jupyter.org/documentation)
- **Google Colab**: [Plataforma de Execução](https://colab.research.google.com/)
- **NumPy**: [Documentação Oficial](https://numpy.org/doc/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
