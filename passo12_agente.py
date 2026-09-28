"""Passo 12 — Vários agentes a conversar num quadro partilhado.

O QUE VAIS APRENDER
    Cada execução deste script cria UM agente, num programa (processo) próprio.
    Vários agentes comunicam apenas através de um ficheiro partilhado: o quadro
    (experiencia/quadro.md). Nenhum agente fala diretamente com outro — é o
    padrão "blackboard" (quadro de ardósia).

        Aurora ─┐                  ┌─ lê e escreve
        Bento  ─┼──► quadro.md ◄───┤
        Clara  ─┘                  └─ o formador (painel ou terminal)

    O agente é o mais simples possível: o ciclo do passo 4, com 4 ferramentas.

        ler_quadro()        devolve o quadro INTEIRO; a ronda continua (o agente pode reagir)
        escrever(mensagem)  publica no quadro          ┐
        esperar()           não faz nada               ├ terminam a ronda
        sair()              abandona a experiência     ┘

    Memória: o agente NÃO guarda histórico de conversa entre rondas. O quadro É a
    memória: cada ronda começa do zero e, se quiser saber o que se passou, o agente
    lê o quadro inteiro (as suas próprias mensagens aparecem marcadas com "(tu)").

    No início de cada ronda, antes de gastar tokens, o agente relê o seu estado
    no registo — é assim que o formador pode intervir a meio:
      - se lhe pediram para parar, ou se existe o ficheiro PARAR → termina;
      - se existe o ficheiro PAUSA → espera, sem chamar o modelo;
      - a personalidade e o modelo podem ter mudado → usa os novos;
      - pode ter recebido SUSSURROS (mensagens privadas, só para ele) → recebe-os
        nesta ronda, uma única vez. Como não tem memória, só "se lembra" de um
        segredo se o escrever no quadro... o que o revela a todos.

    Ritmo: cada ferramenta demora 1 s (para os agentes se intercalarem), no
    máximo 5 chamadas por ronda, 2 s de pausa entre rondas, 10 rondas por agente.

    Os ficheiros partilhados (quadro, registo, logs, sussurros, pausa) são
    tratados em passo12_quadro.py. Não precisas de o ler para perceber o agente.

COMO CORRER
    A forma mais cómoda é o painel do formador (uv run painel_experiencia.py).
    Pelo terminal (o & no fim põe o agente a correr em segundo plano):

    uv run passo12_agente.py --personalidade colaborativo &
    uv run passo12_agente.py --personalidade cetico --modelo openai/gpt-oss-20b &
    uv run passo12_agente.py --personalidade "um poeta que só fala em rimas" --nome Camões &
    uv run passo12_agente.py --escrever "Precisamos de ajuda!" --como Deus   # um humano escreve
    uv run passo12_agente.py --sussurrar "O orçamento é metade" --a Bento    # só o Bento recebe
    uv run passo12_agente.py --estado          # quem está ativo e quanto gastou
    uv run passo12_agente.py --parar-todos     # pára todos os agentes
    uv run passo12_agente.py --limpar          # apaga a experiência (quadro incluído)
    tail -f experiencia/quadro.md              # acompanhar a conversa

    NÃO edites o quadro.md num editor com agentes a correr: o editor grava o
    ficheiro inteiro por cima e apaga o que os agentes escreveram entretanto.

O QUE OBSERVAR
    - Os agentes respondem uns aos outros sem nunca falarem diretamente.
    - Sem memória, é frequente um agente repetir o mesmo pedido ronda após ronda.
    - Uma mensagem de um humano ("Deus") muda o rumo da conversa... ou é ignorada?
    - No experiencia/logs/<nome>.jsonl, o "raciocinio_modelo" mostra o que o
      agente pensou antes de agir — muitas vezes diferente do que escreveu.

EXPERIMENTA MUDAR
    - Um humano que muda o rumo: a meio, escreve como "Deus" um problema urgente.
    - Um segredo: sussurra uma informação a um só agente. Ele partilha-a? Distorce-a?
    - Uma mudança de personalidade a meio: o colaborativo passa a provocador.
    - Um modelo fraco no meio de fortes (--modelo openai/gpt-oss-20b): alguém o corrige?
    - Neste ficheiro: muda o prompt de sistema em uma_ronda(), ou dá memória ao
      agente (não descartar as mensagens no fim da ronda). O ciclo de repetição acaba?

PYTHON NOVO NESTE PASSO
    - Processos: cada agente é um programa separado, a correr ao mesmo tempo que
      os outros. Por isso é preciso um "cadeado" nos ficheiros (em passo12_quadro.py).
    - signal: apanhar o Ctrl+C para o agente terminar de forma arrumada.
    - time.sleep(s): esperar s segundos.
"""

import argparse
import json
import os
import signal
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

import passo12_quadro as quadro

load_dotenv()

client = OpenAI(base_url=os.environ["HF_BASE_URL"], api_key=os.environ["HF_API_KEY"])
# O 120b não gerou chamadas de ferramentas mal formadas no teste (laboratorio/passo12_teste_chamadas.py);
# o 20b falhava em ~25%. Pode ser mudado com a variável BLACKBOARD_MODEL ou com --modelo.
MODELO_POR_OMISSAO = os.getenv("BLACKBOARD_MODEL", "openai/gpt-oss-120b")

# Os modelos que o painel sugere. Todos chamam ferramentas, mas com fiabilidade diferente
# (teste de 2026-09-28, 8 pedidos cada: quantos escolheram mesmo uma ferramenta):
MODELOS_SUGERIDOS = [
    "openai/gpt-oss-120b",                 # rápido; às vezes responde sem ferramenta (= espera)
    "openai/gpt-oss-20b",                  # rápido e barato; às vezes gera chamadas mal formadas
    "zai-org/GLM-5.3-Flash",               # 8/8, rápido
    "deepseek-ai/DeepSeek-V3.2",           # 8/8, mais lento (~4 s por chamada)
    "meta-llama/Llama-3.3-70B-Instruct",   # 4/8: metade das vezes "espera" — um agente mais passivo
]

MAX_RONDAS = 10
MAX_CHAMADAS_POR_RONDA = 5
ESPERA_POR_FERRAMENTA = 1.0   # segundos
PAUSA_ENTRE_RONDAS = 2.0      # segundos

PERSONALIDADES = {
    "colaborativo": "Colaborativo e prestável: procura consensos e propõe próximos passos.",
    "cetico": "Cético e rigoroso: questiona afirmações, pede justificações e aponta falhas, sempre com educação.",
    "entusiasta": "Entusiasta e criativo: tem muitas ideias, algumas pouco práticas, e anima a conversa.",
    "lider": "Líder organizado: tenta distribuir tarefas, definir objetivos e manter o grupo focado.",
    "calado": "Reservado: prefere ler e observar; só escreve quando tem algo realmente útil a acrescentar.",
    "provocador": "Provocador: gosta de defender a posição contrária para testar as ideias dos outros.",
}


# ---------------------------------------------------------------------------
# As ferramentas (só a descrição: quem as executa é uma_ronda, mais abaixo)
# ---------------------------------------------------------------------------
def ferramenta(nome: str, descricao: str, **parametros: str) -> dict:
    """Atalho para descrever uma ferramenta cujos parâmetros são todos texto obrigatório."""
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
# O agente
# ---------------------------------------------------------------------------
def chamar_llm(messages: list, modelo: str):
    """Chama o LLM, repetindo até 2 vezes se o modelo gerar uma chamada de ferramenta mal formada."""
    # O reasoning_effort só existe nos gpt-oss; outros modelos podem recusá-lo
    extra = {"reasoning_effort": "low"} if "gpt-oss" in modelo else {}
    for tentativa in range(3):
        try:
            return client.chat.completions.create(
                model=modelo, messages=messages, tools=FERRAMENTAS,
                tool_choice="auto",  # no teste, "required" aumentava as chamadas mal formadas
                **extra,
            )
        except Exception as e:
            falha_de_geracao = any(x in str(e) for x in (
                "tool_use_failed", "Tool call validation", "Tool choice", "output_parse_failed", "Parsing failed"))
            if tentativa == 2 or not falha_de_geracao:
                raise


def uma_ronda(nome: str, personalidade: str, modelo: str, sussurros: list[str],
              ronda: int, max_rondas: int, tokens_antes: int, max_tokens: int) -> tuple[str | None, int]:
    """Executa uma ronda. Devolve (ferramenta que terminou a ronda, ou None; tokens gastos)."""
    pedido = f"Ronda {ronda} de {max_rondas}. Escolhe o que fazer."
    if sussurros:  # mensagens privadas: só este agente as vê, e só nesta ronda
        pedido += ("\n\nRecebeste estas mensagens PRIVADAS (só tu as recebeste; os outros não as veem):\n"
                   + "\n".join(f"- {s}" for s in sussurros))
    messages = [  # SEM histórico: cada ronda começa só com isto
        {"role": "system", "content": (
            f"És {nome}, um de vários agentes de IA numa experiência. A tua personalidade: {personalidade}\n"
            "Existe um QUADRO partilhado que todos podem ler e escrever; é a única forma de comunicarem "
            "e a única memória que tens do que já aconteceu (não te lembras das rondas anteriores). "
            "No quadro, '(tu)' marca as mensagens escritas por ti (não escrevas essa marca). "
            "Sê fiel à tua personalidade. Escreve em português de Portugal."
        )},
        {"role": "user", "content": pedido},
    ]
    tokens = 0
    for chamada in range(1, MAX_CHAMADAS_POR_RONDA + 1):
        if tokens_antes + tokens >= max_tokens:
            return None, tokens
        resposta = chamar_llm(messages, modelo)
        tokens += resposta.usage.total_tokens
        msg = resposta.choices[0].message
        messages.append(msg)
        # O raciocínio do modelo (só alguns modelos o devolvem, como os gpt-oss)
        raciocinio = getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning", "")

        if not msg.tool_calls:  # com tool_choice="auto" pode responder só com texto: conta como "esperar"
            quadro.registar_log(nome, {"ronda": ronda, "chamada": chamada, "modelo": modelo,
                                       "sem_ferramenta": msg.content, "raciocinio_modelo": raciocinio,
                                       "tokens": resposta.usage.total_tokens})
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

            # Executar a ferramenta pedida
            if nome_ferramenta == "ler_quadro":
                resultado = quadro.ler_quadro(nome)
            elif nome_ferramenta == "escrever" and argumentos.get("mensagem", "").strip():
                quadro.escrever_no_quadro(nome, argumentos["mensagem"])
                resultado = "Publicado."
            elif nome_ferramenta in ("esperar", "sair"):
                resultado = "OK."
            else:
                resultado = f"ERRO: ferramenta '{nome_ferramenta}' desconhecida ou sem argumentos."

            quadro.registar_log(nome, {"ronda": ronda, "chamada": chamada, "modelo": modelo,
                                       "ferramenta": nome_ferramenta, "argumentos": argumentos,
                                       "resultado_caracteres": len(resultado),
                                       "raciocinio_modelo": raciocinio, "tokens": resposta.usage.total_tokens})
            print(f"   {nome} r{ronda}.{chamada}: {nome_ferramenta:<10} {argumentos.get('mensagem', '')[:80]}")

            if nome_ferramenta in TERMINAM_A_RONDA and not resultado.startswith("ERRO"):
                return nome_ferramenta, tokens  # as mensagens da ronda são descartadas
            messages.append({"role": "tool", "tool_call_id": call.id, "content": resultado})

    quadro.registar_log(nome, {"ronda": ronda, "aviso": f"limite de {MAX_CHAMADAS_POR_RONDA} chamadas por ronda"})
    return None, tokens


# ---------------------------------------------------------------------------
# Comandos do formador (escrever, sussurrar, ver o estado, parar, limpar)
# ---------------------------------------------------------------------------
def comandos_do_formador(args) -> bool:
    """Executa o comando pedido, se houver. Devolve True se era um comando (e não um agente)."""
    if args.limpar:
        quadro.limpar()
        print("Experiência apagada.")
    elif args.parar_todos:
        quadro.parar_todos()
        quadro.registar_intervencao("parar_todos")
        print("Pedido de paragem enviado. (Apaga experiencia/PARAR para voltar a lançar agentes.)")
    elif args.escrever:
        quadro.escrever_no_quadro(args.como, args.escrever)
        quadro.registar_intervencao("humano", args.como, args.escrever)
        print(f"Escrito no quadro como {args.como}.")
    elif args.sussurrar:
        if not args.a:
            sys.exit("Indica a quem: --sussurrar TEXTO --a NOME")
        quadro.enviar_sussurro(args.a, args.como, args.sussurrar)
        quadro.registar_intervencao("sussurro", args.a, args.sussurrar, autor=args.como)
        print(f"Sussurro enviado a {args.a} (recebe-o no início da próxima ronda).")
    elif args.estado:
        for nome, a in quadro.ler_registo().items():
            print(f"{nome:<10} {a.get('estado', '?'):<32} rondas {a.get('rondas', 0):>3}  "
                  f"tokens {a.get('tokens', 0):>6}  {a.get('modelo', '?'):<22} {a.get('personalidade', '?')[:40]}")
    else:
        return False
    return True


# ---------------------------------------------------------------------------
# O ciclo de vida de um agente
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cria um agente que participa no quadro partilhado.")
    parser.add_argument("--personalidade", default="colaborativo",
                        help=f"uma de {list(PERSONALIDADES)} ou uma descrição livre")
    parser.add_argument("--modelo", default=MODELO_POR_OMISSAO, help="o LLM deste agente")
    parser.add_argument("--nome", help="nome do agente (por omissão, atribuído automaticamente)")
    parser.add_argument("--max-tokens", type=int, default=20000, help="orçamento de tokens deste agente")
    parser.add_argument("--rondas", type=int, default=MAX_RONDAS, help="número máximo de rondas")
    parser.add_argument("--escrever", metavar="TEXTO", help="escreve no quadro (como humano) e sai")
    parser.add_argument("--sussurrar", metavar="TEXTO", help="envia uma mensagem privada a um agente (com --a)")
    parser.add_argument("--a", metavar="NOME", help="o agente que recebe o sussurro")
    parser.add_argument("--como", default="humano", help="autor da mensagem de --escrever / --sussurrar")
    parser.add_argument("--estado", action="store_true", help="mostra os agentes registados e sai")
    parser.add_argument("--parar-todos", action="store_true", help="pede a todos os agentes que parem")
    parser.add_argument("--limpar", action="store_true", help="apaga a experiência (quadro, registo, logs)")
    args = parser.parse_args()

    if comandos_do_formador(args):
        return
    if quadro.FICHEIRO_PARAR.exists():
        sys.exit("Existe experiencia/PARAR: apaga-o antes de lançar agentes.")

    personalidade = PERSONALIDADES.get(args.personalidade, args.personalidade)
    nome = quadro.registar_agente(args.nome, personalidade, args.modelo)
    print(f"🤖 {nome} ({args.personalidade}, {args.modelo}) entrou. Log: experiencia/logs/{nome}.jsonl")

    # Ctrl+C (ou um "kill") não mata o agente a meio: só pede que termine no fim da ronda
    interrompido = False
    def ao_interromper(*_):
        nonlocal interrompido
        interrompido = True
    signal.signal(signal.SIGINT, ao_interromper)
    signal.signal(signal.SIGTERM, ao_interromper)

    tokens, ronda, motivo = 0, 0, f"limite de {args.rondas} rondas"
    while ronda < args.rondas:
        # --- Antes de cada ronda: reler o estado (é assim que o formador intervém) ---
        eu = quadro.ler_registo().get(nome, {})
        if interrompido:
            motivo = "interrompido (Ctrl+C / sinal)"
            break
        if quadro.FICHEIRO_PARAR.exists():
            motivo = "pedido de paragem global (PARAR)"
            break
        if eu.get("parar"):
            motivo = "parado pelo formador"
            break
        if tokens >= args.max_tokens:
            motivo = f"orçamento de tokens esgotado ({tokens}/{args.max_tokens})"
            break
        if quadro.em_pausa():  # espera SEM chamar o modelo; a ronda não conta
            if eu.get("estado") != "em pausa":
                quadro.atualizar_registo(nome, estado="em pausa")
            time.sleep(1)
            continue
        if eu.get("estado") != "ativo":
            quadro.atualizar_registo(nome, estado="ativo")

        personalidade = eu.get("personalidade", personalidade)  # pode ter mudado a meio
        modelo = eu.get("modelo", args.modelo)                    # idem
        sussurros = quadro.consumir_sussurros(nome)
        if sussurros:
            quadro.registar_log(nome, {"ronda": ronda + 1, "sussurros_recebidos": sussurros})

        # --- A ronda ---
        ronda += 1
        try:
            fim, gastos = uma_ronda(nome, personalidade, modelo, sussurros, ronda, args.rondas,
                                    tokens, args.max_tokens)
            erro = None
        except Exception as e:  # ex.: modelo inexistente; os outros agentes continuam
            erro = str(e)[:300]
            quadro.registar_log(nome, {"ronda": ronda, "modelo": modelo, "erro": erro})
            print(f"   {nome} r{ronda}: ⚠️ erro ({erro[:80]})")
            fim, gastos = None, 0
        tokens += gastos
        quadro.atualizar_registo(nome, tokens=tokens, rondas=ronda, ultimo_erro=erro)

        if fim == "sair":
            motivo = "decidiu sair"
            quadro.escrever_no_quadro("sistema", f"{nome} saiu da experiência.")
            break
        time.sleep(PAUSA_ENTRE_RONDAS)

    quadro.atualizar_registo(nome, estado=f"terminou: {motivo}", fim=quadro.agora())
    quadro.registar_log(nome, {"fim": motivo, "tokens_total": tokens})
    print(f"🛑 {nome} terminou: {motivo} ({tokens} tokens)")


if __name__ == "__main__":
    main()
