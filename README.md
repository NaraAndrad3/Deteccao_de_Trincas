# Sistema Inteligente de Detecção de Trincas e Fissuras

Solução desenvolvida para o **Desafio 2 – Detecção de Trincas e Fissuras**, utilizando técnicas de **Deep Learning para Segmentação de Instâncias** com o modelo **YOLOv8n-seg**.

O objetivo do projeto é automatizar a identificação e localização de falhas estruturais em superfícies como concreto, paredes e pavimentos, auxiliando processos de inspeção, manutenção preventiva e monitoramento de infraestrutura.

---

> ⚠️ O modelo treinado não está versionado neste repositório devido ao tamanho do arquivo.
>
> Download do modelo:
> https://drive.google.com/file/d/1MSMmuEL0Czlf2qD3jtb7puPLYexEBd-7/view?usp=drive_link

# Principais Características

* Segmentação automática de trincas e fissuras
* Modelo YOLOv8n-seg treinado especificamente para o problema
* Detecção e localização visual das falhas
* Pipeline completo de preparação de dados
* Divisão automática em treino, validação e teste
* Avaliação quantitativa por métricas de desempenho
* Aplicação Web desenvolvida com Streamlit
* Solução compatível com ambientes de baixo poder computacional

---

# Arquitetura da Solução

```text
Imagem
   ↓
Pré-processamento
   ↓
YOLOv8n-seg
   ↓
Detecção de Objetos
   ↓
Segmentação das Trincas
   ↓
Cálculo das Confianças
   ↓
Visualização dos Resultados
```

---

# Por que YOLOv8-seg?

O conjunto de dados disponibilizado continha imagens anotadas com máscaras poligonais representando a geometria real das trincas.

Diferentemente de abordagens baseadas apenas em Bounding Boxes, a segmentação permite identificar precisamente a região afetada, fornecendo uma representação muito mais fiel das falhas estruturais.

Entre as alternativas avaliadas, o modelo **YOLOv8n-seg** foi escolhido por apresentar:

* Baixo custo computacional
* Boa capacidade de generalização
* Treinamento eficiente em CPU
* Excelente suporte para segmentação de instâncias
* Facilidade de implantação em aplicações web

---

# Estrutura do Projeto

```text
Deteccao_Trincas/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── data.yaml
├── requirements.txt
│
├── dataset/
│
├── dataset_yolo/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── dataset_inspector.py
│   ├── split_dataset.py
│   └── visualize_labels.py
│
├── results/
│
├── runs/
│
└── report/
```

---

#  Exploração do Dataset

Durante a etapa inicial foi realizada uma análise exploratória do conjunto de dados.

## Estatísticas

| Métrica                     | Valor |
| --------------------------- | ----: |
| Imagens                     |  3102 |
| Labels                      |  3102 |
| Classes                     |     1 |
| Objetos por imagem (média)  |  1,33 |
| Objetos por imagem (máximo) |    18 |

Todos os rótulos foram validados e apresentaram consistência com o formato de segmentação do YOLO.

---

# Divisão do Dataset

A divisão foi realizada seguindo uma estratégia amplamente utilizada em problemas supervisionados.

| Conjunto  | Quantidade |
| --------- | ---------: |
| Treino    |       2171 |
| Validação |        620 |
| Teste     |        311 |

Distribuição:

* Treino: 70%
* Validação: 20%
* Teste: 10%

---

#  Instalação

Clone o repositório:

```bash
git clone https://github.com/SEU_USUARIO/Deteccao_Trincas.git
cd Deteccao_Trincas
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

# Treinamento

Execute:

```bash
python train.py
```

Configuração utilizada:

| Parâmetro   | Valor       |
| ----------- | ----------- |
| Modelo      | YOLOv8n-seg |
| Épocas      | 10          |
| Resolução   | 512         |
| Batch Size  | 4           |
| Dispositivo | CPU         |

---

# Inferência

Para executar a detecção em lote:

```bash
python predict.py
```

Os resultados serão salvos automaticamente na pasta de predições.

---

# Aplicação Web

Antes de executar a aplicação, certifique-se de que o arquivo best.pt foi baixado e posicionado corretamente.

Após verificado, xecute:

```bash
streamlit run app.py
```

Funcionalidades disponíveis:

* Upload de imagens
* Segmentação automática
* Visualização das máscaras
* Exibição das confianças
* Interface amigável para inspeção visual

---

## Observação sobre o Dataset

O conjunto de dados utilizado neste projeto não é distribuído juntamente com o repositório devido ao seu tamanho e às restrições de compartilhamento definidas pelo desafio.

Para reproduzir os experimentos, o usuário deve obter o dataset original fornecido pelos organizadores e posicioná-lo na seguinte estrutura:

```text
dataset/
├── images/
└── labels/
```

Após isso, os scripts de preparação e treinamento podem ser executados normalmente.

## Modelo Treinado
O modelo treinado (`best.pt`) foi disponibilizado separadamente devido ao tamanho do arquivo.

Download do modelo:

Link: [Modelo_treinado](https://drive.google.com/file/d/1MSMmuEL0Czlf2qD3jtb7puPLYexEBd-7/view?usp=drive_link)

Após o download, coloque o arquivo na seguinte estrutura:
```text
runs/
└── segment/
    └── runs/
        └── crack_yolov8n_seg_10ep/
            └── weights/
                └── best.pt
```
Para reproduzir os resultados, execute:

```bash
python train.py
```

Ao final do treinamento, os pesos serão gerados automaticamente.



# Resultados Obtidos

## Métricas de Segmentação

| Métrica   | Valor |
| --------- | ----: |
| Precision |  0.70 |
| Recall    |  0.57 |
| mAP50     |  0.57 |
| mAP50-95  |  0.21 |

As métricas demonstram que o modelo foi capaz de identificar corretamente a maior parte das fissuras presentes no conjunto de validação, apresentando boa capacidade de generalização mesmo após um treinamento relativamente curto realizado em CPU.

---

# Relatório Técnico

O relatório completo encontra-se em:

```text
report/Relatorio_Deteccao_Trincas.pdf
```

O documento apresenta:

* Fundamentação teórica
* Exploração do dataset
* Metodologia
* Processo de treinamento
* Avaliação experimental
* Aplicação Web
* Limitações e trabalhos futuros

---

# Trabalhos Futuros

* Treinamento com maior número de épocas
* Avaliação utilizando GPU
* Testes em cenários industriais reais
* Comparação com modelos YOLOv8s-seg e YOLOv8m-seg
* Deploy em ambiente cloud

---

# Autora

**NaraAndrad3**


