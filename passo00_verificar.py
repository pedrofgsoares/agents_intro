"""Passo 0 — Verificar se está tudo pronto para a formação.

O QUE VAIS APRENDER
    Nada sobre agentes, ainda! Este passo só confirma que o teu computador
    consegue falar com um modelo de linguagem (LLM). Verifica três coisas,
    por ordem, e pára na primeira que falhar:
        1. o ficheiro .env existe?
        2. tem as configurações necessárias (incluindo a tua chave)?
        3. uma pergunta ao modelo recebe resposta?

COMO CORRER
    uv run passo00_verificar.py

O QUE OBSERVAR
    Se alguma coisa falhar, o script diz o que está mal e como corrigir.
    Corrige e volta a correr, até veres "Está tudo pronto".

EXPERIMENTA MUDAR
    - Estraga de propósito a chave no .env (apaga uma letra) e corre outra vez:
      que mensagem aparece? (Depois repõe a chave certa!)
    - Muda HF_MODEL para um nome que não existe, ex.: openai/modelo-inventado.

PYTHON NOVO NESTE PASSO
    - import: traz código escrito por outras pessoas (bibliotecas) para usarmos.
    - Variáveis: nomes para guardar valores, ex.: chave = "hf_..."
    - Funções (def): um bloco de código com nome, que se pode chamar várias vezes.
    - if / else: fazer uma coisa ou outra consoante uma condição.
    - try / except: "tenta fazer isto; se der erro, faz aquilo em vez de rebentar".
    - f"...{variavel}...": texto com valores lá dentro (f-string).
    Mais explicações no GLOSSARIO.md.
"""

import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

# A pasta onde está este ficheiro (a pasta do projeto)
PASTA_PROJETO = Path(__file__).parent
FICHEIRO_ENV = PASTA_PROJETO / ".env"

# As configurações de que todos os passos precisam
OBRIGATORIAS = ["HF_API_KEY", "HF_BASE_URL", "HF_MODEL"]


def falhou(problema: str, solucao: str) -> None:
    """Mostra o problema e a solução, e termina o programa."""
    print(f"\n❌ {problema}")
    print(f"👉 {solucao}")
    sys.exit(1)  # 1 = "terminou com erro" (0 seria "correu bem")


# --- 1. O ficheiro .env existe? ----------------------------------------------
print("1. A procurar o ficheiro .env ...", end=" ")
if not FICHEIRO_ENV.exists():
    falhou("Não encontrei o ficheiro .env na pasta do projeto.",
           "Cria-o a partir do exemplo com:  cp .env.exemplo .env\n"
           "   e depois abre-o e coloca a tua chave em HF_API_KEY.")
print("✅")

# --- 2. As configurações estão lá? --------------------------------------------
print("2. A verificar as configurações ...", end=" ")
load_dotenv(FICHEIRO_ENV)  # lê o .env e guarda cada linha NOME=valor em os.environ

em_falta = [nome for nome in OBRIGATORIAS if not os.getenv(nome)]
if em_falta:
    falhou(f"Faltam estas configurações no .env: {', '.join(em_falta)}",
           "Compara o teu .env com o .env.exemplo e acrescenta as linhas em falta.")

chave = os.environ["HF_API_KEY"]
if chave == "hf_coloca_aqui_a_tua_chave":
    falhou("A HF_API_KEY ainda tem o texto do exemplo.",
           "Substitui-o pela tua chave (vê as instruções no início do .env).")
if not chave.startswith("hf_"):
    falhou("A HF_API_KEY não parece uma chave da Hugging Face (devia começar por 'hf_').",
           "Confirma que copiaste a chave inteira, sem espaços nem aspas.")
print("✅")

# --- 3. O modelo responde? -----------------------------------------------------
modelo = os.environ["HF_MODEL"]
print(f"3. A fazer uma pergunta ao modelo {modelo} ...", end=" ", flush=True)

client = openai.OpenAI(base_url=os.environ["HF_BASE_URL"], api_key=chave)

# Cada tipo de erro tem uma causa provável diferente: tratamos cada um à parte.
try:
    resposta = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": "Diz olá numa frase curta, em português de Portugal."}],
    )
except openai.AuthenticationError:
    falhou("A Hugging Face recusou a tua chave (chave inválida ou apagada).",
           "Cria uma chave nova em https://huggingface.co/settings/tokens e coloca-a no .env.")
except openai.PermissionDeniedError:
    falhou("A tua chave não tem permissão para usar modelos.",
           "Na página da chave, ativa 'Make calls to Inference Providers'.")
except openai.RateLimitError:
    falhou("Pedidos a mais em pouco tempo (limite da Hugging Face).",
           "Espera um minuto e volta a tentar.")
except openai.APIConnectionError:
    falhou("Não consegui ligar à Hugging Face.",
           "Verifica a ligação à internet e o valor de HF_BASE_URL no .env.")
except openai.APIStatusError as erro:
    # Outros erros do servidor. O router da Hugging Face responde "model_not_found"
    # a um modelo que não existe; o código 402 significa que acabaram os créditos.
    if "model_not_found" in str(erro) or erro.status_code == 404:
        falhou(f"O modelo '{modelo}' não existe ou não está disponível.",
               "Confirma o nome em HF_MODEL (o do .env.exemplo é openai/gpt-oss-20b).")
    if erro.status_code == 402:
        falhou("Acabaram os créditos da tua conta Hugging Face.",
               "Fala com o formador.")
    falhou(f"A Hugging Face devolveu um erro inesperado ({erro.status_code}).",
           f"Mostra esta mensagem ao formador: {str(erro)[:200]}")
print("✅")

print(f"\nO modelo respondeu: {resposta.choices[0].message.content}")
print(f"(gastou {resposta.usage.total_tokens} tokens)")
print("\n🎉 Está tudo pronto para a formação!")
