# 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor-Classification

# :computer: Development Tips

## :clipboard: Tensorboard

Rodar o tensorboard sem ter que abrir portas etc.:
```bash
    python3 -m tensorboard.main --logdir=runs/run__.../tensorboard
```

## :shell: Executar em segundo plano (ssh)

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

## :triangular_flag_on_post: Checkpoints

Se o programa falhar, podemos retornar o treino pelo último checkpoint salvo (começa do fold indicado e executa até o fim do resto de folds):
```bash
    python3 main.py {args} --resume=runs/run__.../fold_...
```

Ou se precisamos somente testar novamente um fold:
```bash
    python3 main.py {args} --test=runs/run__.../fold_...
```