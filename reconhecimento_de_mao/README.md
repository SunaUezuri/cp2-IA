# 👋 Reconhecimento de Mãos com MediaPipe

Projeto completo de reconhecimento e rastreamento de mãos em tempo real usando MediaPipe e OpenCV.

## 📋 Características

- ✅ Detecção de até 2 mãos simultaneamente
- ✅ Rastreamento de 21 landmarks por mão
- ✅ Contagem de dedos levantados
- ✅ Identificação de mão direita/esquerda
- ✅ Visualização em tempo real via webcam
- ✅ Cálculo de FPS
- ✅ Interface interativa

## 🔧 Requisitos

- Python 3.8 ou superior
- Webcam conectada ao computador

## 📦 Instalação

1. Clone ou baixe este projeto

2. Crie um ambiente virtual:

```powershell
python -m venv venv
```

3. Ative o ambiente virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Instale as dependências:

```bash
pip install -r requirements.txt
```

> **⚠️ Importante:** Sempre ative o ambiente virtual antes de executar o programa!

## 🚀 Como Usar

### 1️⃣ Ativar o ambiente virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2️⃣ Executar o programa:

```powershell
python hand_recognition.py
```

### 🔄 Ou em uma linha só:

```powershell
.\venv\Scripts\Activate.ps1 ; python hand_recognition.py
```

### Controles

- **Q**: Sair do programa
- **H**: Alternar visualização dos landmarks (pontos das mãos)

## 📊 Informações Exibidas

- **Número de mãos detectadas**: Mostra quantas mãos estão na tela
- **Tipo de mão**: Identifica se é mão direita ou esquerda
- **Dedos levantados**: Conta quantos dedos estão estendidos
- **Estado dos dedos**: Lista [polegar, indicador, médio, anelar, mindinho]
  - 1 = dedo levantado
  - 0 = dedo fechado
- **FPS**: Taxa de quadros por segundo

## 🎯 Landmarks das Mãos

MediaPipe detecta 21 pontos em cada mão:

```
        8  12  16  20
        |   |   |   |
    4   |   |   |   |
    |   7  11  15  19
    |   |   |   |   |
    3   6  10  14  18
    |   |   |   |   |
    2   5   9  13  17
    |   |   |   |   |
    1___|___|___|___|
    |               |
    0               |
  (pulso)           |
```

- **0**: Pulso
- **1-4**: Polegar
- **5-8**: Indicador
- **9-12**: Médio
- **13-16**: Anelar
- **17-20**: Mindinho

## 🛠️ Estrutura do Código

### Classe `HandDetector`

- `__init__()`: Inicializa o detector com parâmetros configuráveis
- `find_hands()`: Detecta mãos na imagem
- `find_position()`: Retorna posições dos landmarks
- `fingers_up()`: Detecta quais dedos estão levantados
- `get_hand_type()`: Identifica se é mão direita ou esquerda

## 🔬 Possíveis Aplicações

- Controle de interface por gestos
- Jogos interativos
- Linguagem de sinais
- Controle de apresentações
- Realidade aumentada
- Contagem e reconhecimento de gestos
- Interação humano-computador

## ⚙️ Configurações Avançadas

Você pode ajustar os parâmetros no código:

```python
detector = HandDetector(
    mode=False,                    # False = usa tracking (mais rápido)
    max_hands=2,                   # Número máximo de mãos
    detection_confidence=0.7,      # Confiança de detecção (0.0 a 1.0)
    tracking_confidence=0.5        # Confiança de rastreamento (0.0 a 1.0)
)
```

## 🐛 Solução de Problemas

### Webcam não abre

- Verifique se a webcam está conectada
- Tente mudar o índice da câmera: `cv2.VideoCapture(1)` ou `cv2.VideoCapture(2)`

### FPS baixo

- Reduza a resolução da câmera
- Aumente o `tracking_confidence`
- Reduza `max_hands` para 1

### Detecção imprecisa

- Melhore a iluminação do ambiente
- Aumente `detection_confidence`
- Evite fundo com cores de pele

## 📚 Tecnologias Utilizadas

- **MediaPipe**: Framework de ML do Google para detecção de mãos
- **OpenCV**: Processamento de imagens e vídeo
- **NumPy**: Manipulação de arrays numéricos

## 📄 Licença

Este projeto é livre para uso educacional e pessoal.

## 👨‍💻 Autor

Desenvolvido com MediaPipe e OpenCV

---

**Divirta-se explorando o reconhecimento de mãos! 👋🖐️✋**
