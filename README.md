# 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor
Vídeo explicativo: [2024-1-P2 - Segmentação e Classificação de Tumores Cerebrais a partir de MRI utilizando CNN's](https://www.youtube.com/watch?v=nvOFCYrG2II).

## Sumário
- [Introdução](#introdução)
- [Dataset](#dataset)
- [Instalação](#instalação)
- [Execução](#executando)
- [DevelopmentTips](#development-tips)

<div id="introdução"></div>

## Introdução
O tumor cerebral é causado pela proliferação celular rápida e descontrolada no cérebro. É uma das doenças mais mortais que
existem e seu tratamento depende fortemente da detecção precoce para salvar vidas. É necessário um especialista com um conhecimento profundo em doenças cerebrais para identificar manualmente o tipo adequado de tumor. Além disso, processar muitas imagens levam tempo e são cansativas. Portanto, técnicas automáticas de segmentação e classificação são essenciais para acelerar e melhorar o diagnóstico de tumores no cérebro. Este trabalho foca na aplicação dessas técnicas em um problema real para identificar o tipo de câncer a partir de imagens de ressonância magnética.

<div id="dataset"></div>

## :open_file_folder: Dataset

Para a realização deste estudo, utilizamos um conjunto de dados de imagens de ressonância magnética do cérebro, que inclui imagens em três planos anatômicos distintos: axial, sagital e coronal. O conjunto de dados é composto por 3064 imagens do tipo T1ce, provenientes de 233 pacientes diagnosticados com três tipos de tumores cerebrais: glioma, meningioma e tumor pituitário. O conjunto pode ser encontrado no seguinte [link](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).

<div id="instalação"></div>

## Instalação

### Clone o repositório
```
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
```
cd datasets
bash script.sh
cd ..
```

<div id="executando"></div>

## Execução

Argumentos:
* --max-epochs: Número máximo de épocas;
* --batch-size: Tamanho dos lotes de treino;
* --cv: Número de folds para validação cruzada;
* --tensor-board: Usar o tensor board para acompanhar os logs em tempo real;
```
python3 ./main.py <argumentos>
```

<div id="development-tips"></div>

## :computer: Development Tips

### :clipboard: Tensorboard

Rodar o tensorboard sem ter que abrir portas etc.:
```bash
    python3 -m tensorboard.main --logdir=runs/run__.../tensorboard
```

### :shell: Executar em segundo plano (ssh)

Colocar a execução em segundo plano e poder fechar o terminal:
```bash
    tmux
    # Inicialize o environment utilizado para rodar o experimento, se necessário.
    conda activate tic
    python3 main.py {args}
```
E para sair pressione ```CTRL+b``` e ```d```.

Para retornar ao terminal:
```bash
    # Provavelmente a janela se chamará 0, mas confira antes
    tmux ls
    tmux attach-session -t 0
```

### :triangular_flag_on_post: Checkpoints

Se o programa falhar, podemos retornar o treino pelo último checkpoint salvo (começa do fold indicado e executa até o fim do resto de folds):
```bash
    python3 main.py {args} --resume=run__.../fold_...
```

Ou se precisamos somente testar novamente um fold:
```bash
    python3 main.py {args} --test=run__.../fold_...
```
