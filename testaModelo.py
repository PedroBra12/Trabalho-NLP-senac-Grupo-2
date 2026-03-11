from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# The sentences to encode (pergunta + resposta esperada)
sentences = [
    {"pergunta": "Qual a diferença entre roteador Wi-Fi e Wi-Fi Direct?",
     "resposta": "Wi-Fi Direct não há necessidade de rotear é uma conexão direta"},
    {"pergunta": "Após conectar em uma rede Wi-Fi pela primeira vez, preciso reconectar nela toda vez?",
     "resposta": "Não, o aparelho se reconectará automaticamente"},
    {"pergunta": "A conexão bluetooth é segura?",
     "resposta": "A conexão bluetooth não é 100% segura certifique-se de que esteja conectando com aparelho de segurança"},
    {"pergunta": "Não encontrei o dispositivo para parear porque?",
     "resposta": "verifique se o aparelho está com a opção de visibilidade ativa"},
    {"pergunta": "Posso comprar um sorvete com meu aparelho?",
     "resposta": "Pode, mas primeiro deve estar cadastrado em um serviço de pagamento móvel"},
    {"pergunta": "Meu plano de internet está acabando rápido porque?",
     "resposta": "Existem aplicativos que podem estar consumindo dados em segundo plano o que pode justificar a internet acabar mais rápido"},
    {"pergunta": "Posso colocar o youtube para rodar somente no Wi-Fi?",
     "resposta": "Pode sim basta ir na tela de configurações, toque em Conexões → Uso de dados → Redes permitidas para apps. Toque o aplicativo desejado e selecione uma opção de rede"},
    {"pergunta": "O som do meu alto-falante está estourando",
     "resposta": "Ative o Dolby Atmos em Sons e vibração → Qualidade sonora e efeitos. Isso deve resolver seus problemas com aúdio."},
    {"pergunta": "O celular não está recohecendo meu dedo, vou ter digitar minha senha toda hora?",
     "resposta": "Você pode usar o reconhecimento facil para desbloquear a tela"},
    {"pergunta": "Se eu estiver jogando online usando dados móveis compartilhados via Roteador Wi-Fi com outro aparelho, e o Game Booster estiver no modo 'Performance', quais são os possíveis impactos negativos combinados que posso enfrentar no meu aparelho?",
     "resposta": "Ao usar todas essas funções a temperatura do aparelho irá aumentar o que por consequência irá limitar a perfomance do aparelho, além de cobranças adicionais pelo uso do Roteador Wi-Fi."},
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)

# 4. Média das similaridades excluindo a diagonal
import torch
n = similarities.shape[0]
mask = ~torch.eye(n, dtype=torch.bool)
media = similarities[mask].mean().item()
print(f"Média (sem diagonal): {media:.4f}")
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])