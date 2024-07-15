# 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor-Classification


## Sumário
- [Introdução](#introdução)
- [Dataset](#dataset)
- [Pré-processamento](#pré-processamento)
- [Instalação](instalação)
- [Execução](#executando)

<div id="introdução"></div>

## Introdução
O tumor cerebral é causado pela proliferação celular rápida e descontrolada no cérebro. É uma das doenças mais mortais que
existem e seu tratamento depende fortemente da detecção precoce para salvar vidas. É necessário um especialista com um conhecimento profundo em doenças cerebrais para identificar manualmente o tipo adequado de tumor. Além disso, processar muitas imagens levam tempo e são cansativas. Portanto, técnicas automáticas de segmentação e classificação são essenciais para acelerar e melhorar o diagnóstico de tumores no cérebro. Este trabalho foca na aplicação dessas técnicas em um problema real para identificar o tipo de câncer a partir de imagens de ressonância magnética.

<div id="dataset"></div>

## :open_file_folder: Dataset

Para a realização deste estudo, utilizamos um conjunto de dados de imagens de ressonância magnética do cérebro, que inclui imagens em três planos anatômicos distintos: axial, sagital e coronal. O conjunto de dados é composto por 3064 imagens do tipo T1ce, provenientes de 233 pacientes diagnosticados com três tipos de tumores cerebrais: glioma, meningioma e tumor pituitário.

<div id="pré-processamento"></div>


## Pré-processamento
O conjunto de dados passou por uma série de transformações. Primeiramente, as imagens, que inicialmente estavam em escala de cinza, foram convertidas para o espaço de cores RGB. Em seguida, todas as imagens foram redimensionadas para uma resolução de 224x224 pixels. Este redimensionamento é fundamental para padronizar o tamanho das imagens e garantir compatibilidade com modelos pré-treinados na ImageNet, que frequentemente utilizam imagens nesta resolução durante o treinamento. Além do redimensionamento, as imagens foram normalizadas utilizando médias de [0.485, 0.456, 0.406] e desvios padrão de [0.229, 0.224, 0.225], conforme padrões estabelecidos pelo conjunto de dados ImageNet. Para enriquecer a diversidade do conjunto de treinamento, foram utilizadas técnicas de aumento de dados. Foram aplicadas diferentes transformações nas imagens, incluindo ajustes aleatórios de brilho, contraste, escala e translação. Essas transformações foram feitas usando a biblioteca [Albumentations](https://github.com/albumentations-team/albumentations/tree/main).

<div id="instalação"></div>

## Instalação

### Access your machine with GPU support and clone the repository
```
# SSH to Linux server via designated port (for tensorboard)
ssh -L 6006:LocalHost:6006 username@server_address

# Clone o repositório
git clone git@github.com:intel-comp-saude-ufes/2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor.git
```

### Nós recomendamos usar um abiente virtual do python para instalar os pacotes necessários
```
cd 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
### Se você desejar usar o mesmo dataset que nós, rode o seguinte script 
**Nota: você deve precisar de uma conta do Kaggle**
```
cd datasets
bash script.sh
```
<div id="executando"></div>

## Execução
**Nota: dicas de como usar o tensor board na branch dev**

-Argumentos:
* --max-epochs: Número máximo de épocas;
* --batch-size: Tamanho dos lotes de treino;
* --cv: Número de folds para validação cruzada;
* --tensor-board: Usar o tensor board para acompanhar os logs em tempo real;
```
cd ..
python3 ./main.py <argumentos>
```


