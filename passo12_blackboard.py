"""Passo 12 — Experiência multi-agente com um quadro partilhado (blackboard).

Cada execução deste script cria UM agente, num processo próprio. Vários agentes
(lançados em terminais diferentes, ou em segundo plano) comunicam apenas
através de um ficheiro partilhado: o quadro. Nenhum agente fala diretamente
com outro — é o padrão "blackboard".

O agente é o mais simples possível: o ciclo do passo 4, com 4 ferramentas.

    ler_quadro()        devolve o quadro INTEIRO; a ronda continua (o agente pode reagir)
    escrever(mensagem)  publica no quadro          ┐
    esperar()           não faz nada               ├ terminam a ronda
    sair()              abandona a experiência     ┘

Ritmo:
  - cada chamada de ferramenta demora 1 segundo (simula latência e deixa os
    outros agentes intercalarem);
  - no máximo 5 chamadas de ferramentas por ronda;
  - 2 segundos de pausa entre rondas;
  - no máximo 10 rondas por agente.

Memória: o agente NÃO guarda histórico de conversa entre rondas. O quadro É a
memória: cada ronda começa do zero e, se quiser saber o que se passou, o agente
lê o quadro inteiro (as suas próprias mensagens aparecem marcadas com "(tu)").

Lançar um agente NUNCA apaga o quadro: se quadro.md não existir, é criado;
se existir, continua. Só --limpar apaga a experiência.

NÃO edites o quadro.md num editor com agentes a correr: o editor grava o
ficheiro inteiro por cima e apaga o que os agentes escreveram entretanto
(não respeita o lock). Usa --escrever, que passa pelo mesmo lock.

Paragem: 10 rondas, --max-tokens, o ficheiro PARAR (--parar-todos), Ctrl+C,
ou o próprio agente chamar sair().

Ficheiros (em experiencia/):
  quadro.md          o blackboard (legível por humanos)
  agentes.json       registo dos agentes: nome, personalidade, estado, tokens
  logs/<nome>.jsonl  cada chamada de ferramenta, com o raciocínio do modelo
  PARAR              se existir, todos os agentes terminam

Uso:
    uv run passo12_blackboard.py --personalidade colaborativo &
    uv run passo12_blackboard.py --personalidade cetico &
    uv run passo12_blackboard.py --personalidade "um poeta que só fala em rimas" --nome Camões &
    uv run passo12_blackboard.py --escrever "Precisamos de ajuda!" --como Deus   # um humano escreve
    uv run passo12_blackboard.py --estado          # quem está ativo e quanto gastou
    uv run passo12_blackboard.py --parar-todos     # pára todos os agentes
    uv run passo12_blackboard.py --limpar          # apaga a experiência (quadro incluído)
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url=os.environ["HF_BASE_URL"], api_key=os.environ["HF_API_KEY"])
# O 120b não gerou chamadas de ferramentas mal formadas no teste (passo12_teste_chamadas.py);
# o 20b do .env falhava em ~25%. Pode ser mudado com a variável BLACKBOARD_MODEL.
MODEL = os.getenv("BLACKBOARD_MODEL", "openai/gpt-oss-120b")

PASTA = Path(__file__).parent / "experiencia"
QUADRO = PASTA / "quadro.md"
REGISTO = PASTA / "agentes.json"
PASTA_LOGS = PASTA / "logs"
FICHEIRO_PARAR = PASTA / "PARAR"

MAX_RONDAS = 10
MAX_CHAMADAS_POR_RONDA = 5
ESPERA_POR_FERRAMENTA = 1.0   # segundos
PAUSA_ENTRE_RONDAS = 2.0      # segundos
TEMA_INICIAL = "Apresentem-se e combinem entre vocês um pequeno problema para resolverem juntos."

NOMES = ["Aurora", "Bento", "Clara", "Duarte", "Estela", "Fabian", "Girafa", "Hugo", "Laura", "Jorge",
         "Leonor", "Miguel", "Nanda", "Oriental", "Paula", "Rui", "Sara", "Tom", "Vera", "Xavier"]

PERSONALIDADES = {
    #"colaborativo": "Colaborativo e prestável: procura consensos, resume o que os outros disseram e propõe próximos passos.",
    "colaborativo": "Colaborativo e prestável: procura consensos e propõe próximos passos.",
    "cetico": "Cético e rigoroso: questiona afirmações, pede justificações e aponta falhas, sempre com educação.",
    "entusiasta": "Entusiasta e criativo: tem muitas ideias, algumas pouco práticas, e anima a conversa.",
    "lider": "Líder organizado: tenta distribuir tarefas, definir objetivos e manter o grupo focado.",
    "calado": "Reservado: prefere ler e observar; só escreve quando tem algo realmente útil a acrescentar.",
    "provocador": "Provocador: gosta de defender a posição contrária para testar as ideias dos outros.",
}


def ferramenta(nome: str, descricao: str, **parametros: str) -> dict:
    return {"type": "function", "function": {
        "name": nome, "description": descricao,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string", "description": d} for p, d in parametros.items()},
                       "required": list(parametros)},
    }}


FERRAMENTAS = [
    ferramenta("ler_quadro", "Lê o quadro partilhado inteiro. Depois de ler, podes fazer outra ação nesta ronda."),
    ferramenta("escrever", "Publica uma mensagem no quadro partilhado. Termina a tua ronda.",
               mensagem="A mensagem a publicar (máximo 2 frases)"),
    ferramenta("esperar", "Não fazes nada nesta ronda. Termina a tua ronda."),
    ferramenta("sair", "Abandonas a experiência definitivamente."),
]
TERMINAM_A_RONDA = {"escrever", "esperar", "sair"}


# ---------------------------------------------------------------------------
# Acesso partilhado aos ficheiros — vários processos ao mesmo tempo!
# ---------------------------------------------------------------------------
@contextmanager
def trancado():
    PASTA.mkdir(exist_ok=True)
    with open(PASTA / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ler_registo() -> dict:
    return json.loads(REGISTO.read_text()) if REGISTO.exists() else {}


def atualizar_registo(nome: str, **campos) -> None:
    with trancado():
        registo = ler_registo()
        registo.setdefault(nome, {}).update(campos)
        REGISTO.write_text(json.dumps(registo, ensure_ascii=False, indent=2))


def registar_agente(nome_pedido: str | None, personalidade: str) -> str:
    """Escolhe um nome livre (ou valida o pedido) e regista o agente — tudo dentro do lock."""
    with trancado():
        registo = ler_registo()
        if nome_pedido:
            if nome_pedido in registo:
                sys.exit(f"O nome '{nome_pedido}' já foi usado nesta experiência. Escolhe outro.")
            nome = nome_pedido
        else:
            livres = [n for n in NOMES if n not in registo]
            nome = livres[0] if livres else f"Agente{len(registo) + 1}"
        registo[nome] = {"personalidade": personalidade, "pid": os.getpid(), "estado": "ativo",
                         "inicio": datetime.now().isoformat(timespec="seconds"), "tokens": 0, "rondas": 0}
        REGISTO.write_text(json.dumps(registo, ensure_ascii=False, indent=2))
        # O quadro só é criado se ainda não existir: lançar um agente nunca o apaga
        if not QUADRO.exists():
            QUADRO.write_text(f"[{datetime.now():%H:%M:%S}] sistema: Tema: {TEMA_INICIAL}\n", encoding="utf-8")
    return nome


def escrever_no_quadro(autor: str, texto: str) -> None:
    # Os agentes tendem a imitar o formato do que leem e copiam a marca "(tu)":
    # removemo-la no código (não basta pedir no prompt)
    linha = " ".join(texto.replace("(tu)", "").split())  # uma entrada = uma linha
    # ...e também imitam o cabeçalho das linhas ("[12:36:45] Aurora:", "Bento (eu):"): removemo-lo
    linha = re.sub(rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s*)?{re.escape(autor)}\s*(\((eu|tu)\))?\s*:\s*", "", linha)
    with trancado(), QUADRO.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {autor}: {linha}\n")


def ler_quadro(nome: str) -> str:
    """O quadro INTEIRO, com as mensagens do próprio agente marcadas com '(tu)'."""
    linhas = QUADRO.read_text(encoding="utf-8").splitlines() if QUADRO.exists() else []
    return "\n".join(l.replace(f"] {nome}:", f"] {nome} (tu):", 1) for l in linhas) or "(quadro vazio)"


def registar_log(nome: str, entrada: dict) -> None:
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    with (PASTA_LOGS / f"{nome}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hora": datetime.now().isoformat(timespec="seconds"), **entrada}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# O agente
# ---------------------------------------------------------------------------
def chamar_llm(messages: list):
    """Chama o LLM, repetindo até 2 vezes se o gpt-oss gerar uma chamada de ferramenta mal formada."""
    for tentativa in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=FERRAMENTAS,
                tool_choice="auto",  # no teste, "required" aumentava as chamadas mal formadas
                reasoning_effort="low",
            )
        except Exception as e:
            falha_de_geracao = any(x in str(e) for x in (
                "tool_use_failed", "Tool call validation", "Tool choice", "output_parse_failed", "Parsing failed"))
            if tentativa == 2 or not falha_de_geracao:
                raise


def uma_ronda(nome: str, personalidade: str, ronda: int, tokens_antes: int, max_tokens: int) -> tuple[str | None, int]:
    """Executa uma ronda. Devolve (ferramenta que terminou a ronda, ou None; tokens gastos)."""
    messages = [  # SEM histórico: cada ronda começa só com isto
        {"role": "system", "content": (
            f"És {nome}, um de vários agentes de IA numa experiência. A tua personalidade: {personalidade}\n"
            "Existe um QUADRO partilhado que todos podem ler e escrever; é a única forma de comunicarem "
            "e a única memória que tens do que já aconteceu (não te lembras das rondas anteriores). "
            "No quadro, '(tu)' marca as mensagens escritas por ti (não escrevas essa marca). "
            "Sê fiel à tua personalidade. Escreve em português de Portugal."
        )},
        {"role": "user", "content": f"Ronda {ronda} de {MAX_RONDAS}. Escolhe o que fazer."},
    ]
    tokens = 0
    for chamada in range(1, MAX_CHAMADAS_POR_RONDA + 1):
        if tokens_antes + tokens >= max_tokens:
            return None, tokens
        resposta = chamar_llm(messages)
        tokens += resposta.usage.total_tokens
        msg = resposta.choices[0].message
        messages.append(msg)
        raciocinio = getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning", "")

        if not msg.tool_calls:  # com tool_choice="auto" pode responder só com texto: conta como "esperar"
            registar_log(nome, {"ronda": ronda, "chamada": chamada, "sem_ferramenta": msg.content,
                                "raciocinio_modelo": raciocinio, "tokens": resposta.usage.total_tokens})
            print(f"   {nome} r{ronda}.{chamada}: (texto, sem ferramenta -> esperar)")
            return "esperar", tokens

        for call in msg.tool_calls:
            time.sleep(ESPERA_POR_FERRAMENTA)  # cada ferramenta demora 1 segundo
            # Limpeza: o gpt-oss às vezes deixa "vazar" marcas do seu formato interno para o nome
            # (ex: "escrever<|channel|>commentary"). Alguns fornecedores deixam-nas passar.
            nome_ferramenta = call.function.name.split("<|")[0].strip()
            try:
                argumentos = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                argumentos = {}

            if nome_ferramenta == "ler_quadro":
                resultado = ler_quadro(nome)
            elif nome_ferramenta == "escrever" and argumentos.get("mensagem", "").strip():
                escrever_no_quadro(nome, argumentos["mensagem"])
                resultado = "Publicado."
            elif nome_ferramenta in ("esperar", "sair"):
                resultado = "OK."
            else:
                resultado = f"ERRO: ferramenta '{nome_ferramenta}' desconhecida ou sem argumentos."

            registar_log(nome, {"ronda": ronda, "chamada": chamada, "ferramenta": nome_ferramenta,
                                "argumentos": argumentos, "resultado_caracteres": len(resultado),
                                "raciocinio_modelo": raciocinio, "tokens": resposta.usage.total_tokens})
            print(f"   {nome} r{ronda}.{chamada}: {nome_ferramenta:<10} {argumentos.get('mensagem', '')[:80]}")

            if nome_ferramenta in TERMINAM_A_RONDA and not resultado.startswith("ERRO"):
                return nome_ferramenta, tokens  # as mensagens da ronda são descartadas
            messages.append({"role": "tool", "tool_call_id": call.id, "content": resultado})

    registar_log(nome, {"ronda": ronda, "aviso": f"limite de {MAX_CHAMADAS_POR_RONDA} chamadas por ronda"})
    return None, tokens


def main():
    parser = argparse.ArgumentParser(description="Cria um agente que participa no quadro partilhado.")
    parser.add_argument("--personalidade", default="colaborativo",
                        help=f"uma de {list(PERSONALIDADES)} ou uma descrição livre")
    parser.add_argument("--nome", help="nome do agente (por omissão, atribuído automaticamente)")
    parser.add_argument("--max-tokens", type=int, default=20000, help="orçamento de tokens deste agente")
    parser.add_argument("--escrever", metavar="TEXTO", help="escreve no quadro (como humano) e sai")
    parser.add_argument("--como", default="humano", help="autor da mensagem de --escrever")
    parser.add_argument("--estado", action="store_true", help="mostra os agentes registados e sai")
    parser.add_argument("--parar-todos", action="store_true", help="pede a todos os agentes que parem")
    parser.add_argument("--limpar", action="store_true", help="apaga a experiência (quadro, registo, logs)")
    args = parser.parse_args()

    if args.limpar:
        shutil.rmtree(PASTA, ignore_errors=True)
        return print("Experiência apagada.")
    if args.parar_todos:
        PASTA.mkdir(exist_ok=True)
        FICHEIRO_PARAR.touch()
        return print("Pedido de paragem enviado. (Apaga experiencia/PARAR para voltar a lançar agentes.)")
    if args.escrever:
        escrever_no_quadro(args.como, args.escrever)
        return print(f"Escrito no quadro como {args.como}.")
    if args.estado:
        for nome, a in ler_registo().items():
            print(f"{nome:<10} {a.get('estado', '?'):<32} rondas {a.get('rondas', a.get('iteracoes', 0)):>3}  "
                  f"tokens {a.get('tokens', 0):>6}  {a.get('personalidade', '?')[:50]}")
        return
    if FICHEIRO_PARAR.exists():
        sys.exit("Existe experiencia/PARAR: apaga-o antes de lançar agentes.")

    personalidade = PERSONALIDADES.get(args.personalidade, args.personalidade)
    nome = registar_agente(args.nome, personalidade)
    print(f"🤖 {nome} ({args.personalidade}) entrou. Log: experiencia/logs/{nome}.jsonl")

    interrompido = False
    def ao_interromper(*_):
        nonlocal interrompido
        interrompido = True
    signal.signal(signal.SIGINT, ao_interromper)
    signal.signal(signal.SIGTERM, ao_interromper)

    tokens, motivo = 0, f"limite de {MAX_RONDAS} rondas"
    for ronda in range(1, MAX_RONDAS + 1):
        # verificações de paragem ANTES de gastar tokens
        if interrompido:
            motivo = "interrompido (Ctrl+C / sinal)"
            break
        if FICHEIRO_PARAR.exists():
            motivo = "pedido de paragem global (PARAR)"
            break
        if tokens >= args.max_tokens:
            motivo = f"orçamento de tokens esgotado ({tokens}/{args.max_tokens})"
            break

        try:
            fim, gastos = uma_ronda(nome, personalidade, ronda, tokens, args.max_tokens)
        except Exception as e:
            registar_log(nome, {"ronda": ronda, "erro": str(e)[:300]})
            print(f"   {nome} r{ronda}: ⚠️ erro ({str(e)[:80]})")
            fim, gastos = None, 0
        tokens += gastos
        atualizar_registo(nome, tokens=tokens, rondas=ronda)

        if fim == "sair":
            motivo = "decidiu sair"
            escrever_no_quadro("sistema", f"{nome} saiu da experiência.")
            break
        time.sleep(PAUSA_ENTRE_RONDAS)

    atualizar_registo(nome, estado=f"terminou: {motivo}", fim=datetime.now().isoformat(timespec="seconds"))
    registar_log(nome, {"fim": motivo, "tokens_total": tokens})
    print(f"🛑 {nome} terminou: {motivo} ({tokens} tokens)")


if __name__ == "__main__":
    main()
