# TelecomX: Análise de Evasão de Clientes (Churn)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-%234470AD.svg?style=for-the-badge&logo=Seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

Este repositório contém uma análise detalhada de dados do setor de telecomunicações, focada em entender o fenômeno do **Churn** (evasão de clientes). O projeto utiliza Python e bibliotecas de Data Science para extrair insights valiosos sobre o comportamento dos consumidores.

## 📌 Objetivo do Projeto
Identificar os principais gatilhos que levam um cliente a cancelar seus serviços. A análise busca responder se fatores como o valor da conta mensal, o total gasto acumulado e a quantidade de serviços contratados possuem correlação direta com a saída do cliente.

---

## 📊 Principais Insights

### 1. Distribuição de Serviços vs. Evasão
A análise via boxplot revelou que a mediana de serviços utilizados é idêntica (4 serviços) tanto para clientes que ficaram quanto para os que saíram. No entanto, clientes retidos apresentam uma dispersão maior, chegando a utilizar até 6 serviços na faixa central (IQR), enquanto os que evadem concentram-se em até 5.

### 2. Impacto Financeiro (Valor Mensal e Total)
* **Valor Mensal:** Observou-se uma tendência de aumento na taxa de evasão conforme o valor da fatura mensal sobe, especialmente em faixas intermediárias.
* **Valor Total:** Clientes com menor tempo de casa (e consequentemente menor Valor Total acumulado) representam o maior volume de churn, indicando que a retenção é crítica nos primeiros meses de contrato.

---

## 🛠️ Tecnologias e Ferramentas
* **Linguagem:** Python 3.x
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização de Dados:** Matplotlib, Seaborn
* **Ambiente de Desenvolvimento:** Google Colab

## 🚀 Como Visualizar
1.  Acesse o notebook diretamente pelo link do repositório.
2.  Para executar, recomenda-se abrir o arquivo `.ipynb` no **Google Colab**.

## 📂 Estrutura de Arquivos
* `data/`: Base de dados
* `notebook/`: Notebook principal com todo o código de análise, gráficos e o relatório final.

---

## ✒️ Autor
Desenvolvido por **Vitor Leal Luiz**.
Conecte-se comigo no [LinkedIn](https://www.linkedin.com/in/vitorluizleal/)!
