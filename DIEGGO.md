## 04 de março de 2026
Criação do repositório e envio de 6 PDFs para futura criação de um SmartFAQ.
Os PDFs são manuais em português de carro, eletrônicos, celular, ar condicionado, calculadora e celular.

## 05 de março de 2026
Adição de mais 3 PDFs com tamanho menor devido a falta de informações.

## 06 de março de 2026
Extração dos textos dos PDFs utilizando PyMuPDF.

## 09 de março de 2026
Perguntas feitas sobre o texto "Manual SM-A15X_A16X_A17X_A06X_A075_16_Emb_BR_Rev.2.1"
1. O que o Painel Edge faz?
"Você pode acessar seus aplicativos favoritos e funções dos Painéis Edge."

2. Quais as formas de bloqueio de tela?
"Ao definir um Padrão, PIN, Senha ou seus dados biométricos como modo de bloqueio
de tela"

3. Quais os aplicativos para adquirir novos aplicativos?
"Galaxy Store: Compre e baixe mais aplicativos... Play Store: Compre e baixe mais aplicativos..."

4. Quais os modos de camêra?
"Modo Foto Modo Vídeo Modo Retrato Modo Diversão Modo Pro Modo Panorâmica Modo Food Modo Noturno Modo Macro Modo Câmera lenta Modo Timelapse" Pgs 52 a 56

5. O que acontece quando um número bloqueado me ligar?
"Quando os números bloqueados tentarem ligar, você não receberá notificações. As chamadas serão registradas na lista."

6. Quais ações podem ser feitas em um contato?
"• : efetua uma chamada.
• : envia uma mensagem.
• : efetua uma videochamada.
• : envia um e-mail."

7. Onde verificar cartões compatíveis com o Samsung Wallet?
"Você poderá verificar mais informações, tais como os cartões compatíveis com essa
função no site www.samsung.com/samsung-wallet."

8. Qual o aplicativo padrão para o envio de mensagens?
"Samsung Messages"

9. Quais as opções disponiveis no modo PRO?
"ISO, SPEED, EV, FOCUS, WB" Pg 54

10. O que é possivel de fazer no Samsung Notes?
"Crie notas ao inserir um texto com o teclado ou ao escrever e desenhar na tela. Também
é possível adicionar imagens e gravações de voz às suas notas."

## 10 de março de 2026
Foi testado modelos com o 'sentenceTransformer':
'all-MiniLM-L6-v2',
'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-multilingual-mpnet-base-v2'

## 11 de março de 2026
Começei a usar o LangChain.

## 12 de março de 2026
Faltei.

## 13 de março de 2026
Testes iniciais na maquina remota.

## 16 de março de 2026
Trabalho como assistente de TI para outros alunos.

## 17 de março de 2026
Faltei.

## 18 de março de 2026
Analise inicial de 2 codigos enviados pelo professor sobre langchain.

## 19 de março de 2026
Testando GitLens.
Implementado o codigo de nolegacy para ler um dos PDFs e testado alguns modelos. Pessimos resultados, começei a olhar para RAGs.

## 24 de março de 2026
Assitente de TI.

## 25 de março de 2026
Vibe coding hard no codigo de LangChain. Ainda não testado devido a problemas para usar o gervasio pela HF.

## 26 de março de 2026
Re criei o pipeline para tentar simplificar as coisas. Ainda falta algumas coisas.

## 27 de março de 2026 
Finalizando de executar a pipeline, descobri uma forma mais simples para fazer em uma futura versão (LlamaIndex).

## 31 de março de 2026
LlamaIndex funciona! Pega um PDF e pode validar com um um arquivo de golden Q&A ou responder questões sobre o PDF. Ainda tenho que testar o finetune_reranker.