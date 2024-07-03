# 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor-Classification

## Usando o WANDB-AI

Para configurar o _wandb_, é necessário instalar ele.
```bash
pip install wandb
```

E então, fazer login.
```bash
wandb login
```

Também é necessário criar um arquivo 'wandb.env' que conterá as informações do projeto e a sua chave API exclusiva. No lugar de _API-KEY_, coloque a chave.
```bash
WANDB_KEY="API-KEY"
WANDB_PROJECT="2024-1-P2-TIC"
```

Com isso, a run será salva online com todas as informações da execução atual.

### Desativando o WANDB-AI
IMPORTANTE: caso esteja rodando casualmente testes, desetive o wandb para que os resultados não sejam reportados na página do projeto.

```bash
wandb offline
```