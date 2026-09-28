"""Passo 10 — Memória: o grafo do passo 9b passa a lembrar-se de cada estudante.

O QUE VAIS APRENDER
    Até agora cada email era analisado "sem passado". Mas há decisões que dependem
    de pedidos anteriores (emails 009 e 010 da caixa fictícia):
      - 009: a estudante reutiliza o MESMO atestado de um pedido anterior
      - 010: o estudante envia o documento que faltava num pedido anterior; pelo
             art. 4.º, n.º 3, o prazo conta a partir do PEDIDO INICIAL, não deste email

    Três tipos de "memória" que já vimos ou vamos ver:
      1. Histórico da conversa  (passo 2: a lista `messages`)  -> dura uma conversa
      2. Estado de uma execução (o `Estado` do grafo; com um "checkpointer" pode ser
         gravado e retomado — útil para pausar à espera do docente)
      3. Memória de longo prazo (este passo: o "store" do LangGraph) -> dura entre
         execuções e é partilhada entre conversas. Organiza-se por "namespaces":
         aqui, ("estudantes", "up202401234") guarda os pedidos desse estudante.

    Decisões de desenho:
      - Quem lê e escreve a memória é o CÓDIGO (nós consultar_memoria e guardar_memoria),
        não o LLM. O LLM não pode "esquecer-se" de consultar, nem escrever na memória
        algo que um email malicioso lhe peça.
      - Os documentos são comparados por uma impressão digital (hash SHA-256 do texto):
        determinístico, e não é preciso guardar o documento (dados de saúde!) na memória.
      - Gestão de contexto: o rascunho só recebe os ÚLTIMOS pedidos do estudante,
        não o histórico inteiro — o contexto do LLM não pode crescer sem limite.
      - A memória cria DEPENDÊNCIA DA ORDEM: os emails têm de ser processados por
        ordem cronológica (deixámos de poder processar em paralelo, como no 9b).

    A memória fica em memoria/memoria.sqlite e sobrevive entre execuções.

    Reutiliza do passo 7 (pasta de saída, resumo), do 9a (modelo, ferramentas, log) e do 9b (Factos, regras).

COMO CORRER
    uv run passo10_memoria.py                  # processa a caixa (por ordem)
    uv run passo10_memoria.py 010              # só o 010 (usa a memória que já existir)
    uv run passo10_memoria.py --ver-memoria    # mostra o que está guardado
    uv run passo10_memoria.py --esquecer       # apaga a memória

O QUE OBSERVAR
    (Podes só correr e observar: o essencial está no desenho do grafo, não nos detalhes do código.)
    - Corre a caixa toda e depois --ver-memoria: o que ficou guardado de cada estudante?
    - Nos emails 009 e 010, os alertas passam a referir o pedido anterior.
    - Corre --esquecer e depois só o 010: compara a proposta e os alertas com os de
      quando havia memória.
    - Avalia: uv run passo08_avaliacao.py --agente passo10 (coluna "memória").

EXPERIMENTA MUDAR
    - Processa os emails por outra ordem (ex.: 010 antes de 002). O que acontece?
    - Procura em guardar_memoria o que é guardado: porque não se guarda o atestado?

PYTHON NOVO NESTE PASSO
    - hashlib.sha256(...): calcula uma "impressão digital" de um texto — dois
      textos iguais dão sempre o mesmo código, e não se consegue recuperar o
      texto a partir dele.
    - Runtime / store: a forma do LangGraph dar aos nós acesso à memória.
"""

import asyncio
import hashlib
import operator
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.types import RetryPolicy

from passo07_agente_email import PASTA_SAIDA, mostrar_resumo
from passo09a_langgraph_agente import criar_modelo, ligar_ferramentas, registar_bloco, repetir_se, texto_de
from passo09b_langgraph_grafo import MOTIVOS_ACEITES, PRAZO_DIAS_UTEIS, Factos

PASTA_PROJETO = Path(__file__).parent
FICHEIRO_LOG = PASTA_PROJETO / "passo10.log"
FICHEIRO_MEMORIA = PASTA_PROJETO / "memoria" / "memoria.sqlite"
HISTORICO_NO_CONTEXTO = 3  # quantos pedidos anteriores o LLM vê ao redigir

# A avaliação (passo 8) precisa de saber que este agente depende da ordem dos emails
MEMORIA_ENTRE_EMAILS = True


class Estado(TypedDict, total=False):
    id_email: str
    numero_estudante: str
    email: str
    data_envio: str
    anexos: dict[str, str]
    impressoes_anexos: dict[str, str]    # nome do anexo -> hash do conteúdo
    regulamento: str
    factos: Factos
    historico: list[dict]                # pedidos anteriores deste estudante
    pedido_original: dict | None         # se este email completa um pedido anterior
    documentos_reutilizados: list[str]
    dias_uteis: int | None
    recomendacao: str
    verificacoes: list[str]
    alertas: list[str]
    chamadas: Annotated[list[dict], operator.add]
    tokens: Annotated[int, operator.add]


def impressao_digital(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def construir_grafo(ferramentas, modelo=None, effort=None, log: Path | None = None):
    por_nome = {f.name: f for f in ferramentas}
    llm = criar_modelo(modelo, effort)

    async def chamar(nome: str, **argumentos) -> tuple[str, dict]:
        texto = texto_de(await por_nome[nome].ainvoke(argumentos))
        return texto, {"ferramenta": nome, "argumentos": argumentos, "erro": texto.startswith("Error executing")}

    # --- carregar (código) — como no 9b, mais o nº de estudante e as impressões digitais
    async def carregar(estado: Estado) -> dict:
        id_email = estado["id_email"]
        email, c1 = await chamar("ler_email", id_email=id_email)
        regulamento, c2 = await chamar("ler_regulamento")
        chamadas = [c1, c2]
        remetente = re.search(r"De: .*<([^@>]+)@", email)
        data_envio = re.search(r"Data de envio: (\d{4}-\d{2}-\d{2})", email).group(1)
        linha_anexos = re.search(r"Anexos: (.*)", email).group(1)
        anexos = {}
        if linha_anexos != "nenhum":
            for nome in linha_anexos.split(", "):
                anexos[nome], c = await chamar("ler_anexo", id_email=id_email, nome_anexo=nome)
                chamadas.append(c)
        return {
            "email": email, "regulamento": regulamento, "data_envio": data_envio, "anexos": anexos,
            "numero_estudante": remetente.group(1) if remetente else "desconhecido",
            "impressoes_anexos": {nome: impressao_digital(t) for nome, t in anexos.items()},
            "chamadas": chamadas,
        }

    # --- extrair_factos (LLM) — igual ao 9b ----------------------------------
    async def extrair_factos(estado: Estado) -> dict:
        anexos = "\n\n".join(f"### Anexo: {nome}\n{texto}" for nome, texto in estado["anexos"].items())
        pedido = (
            "Extrai os factos deste email de um estudante, à luz do regulamento. "
            "Não tomes nenhuma decisão: só factos.\n\n"
            f"## Regulamento\n{estado['regulamento']}\n\n"
            f"## Email\n{estado['email']}\n\n## Anexos\n{anexos or '(nenhum)'}"
        )
        resposta = await llm.with_structured_output(Factos, method="json_schema", include_raw=True).ainvoke(pedido)
        if resposta["parsing_error"]:
            raise resposta["parsing_error"]
        registar_bloco(log, "LLM: extrair_factos (resposta)", resposta["parsed"].model_dump())
        return {"factos": resposta["parsed"], "tokens": (resposta["raw"].usage_metadata or {}).get("total_tokens", 0)}

    # --- consultar_memoria (código) — NOVO ------------------------------------
    # O `runtime` é injetado pelo LangGraph e dá acesso ao store configurado no compile()
    async def consultar_memoria(estado: Estado, runtime: Runtime) -> dict:
        itens = await runtime.store.asearch(("estudantes", estado["numero_estudante"]), limit=100)
        # Excluir este mesmo email: processá-lo duas vezes não o pode tornar "reutilizado"
        historico = sorted((i.value for i in itens if i.key != estado["id_email"]), key=lambda p: p["data_envio"])
        f = estado["factos"]

        # Este email completa um pedido anterior da mesma falta que ficou a aguardar documentos?
        pedido_original = next((p for p in historico if p["data_falta"] == f.data_falta
                                and p["recomendacao"] == "pedir_mais_informacao"), None)

        # Algum anexo já foi entregue noutro pedido?
        reutilizados = []
        for nome, impressao in estado["impressoes_anexos"].items():
            for p in historico:
                if impressao in p["impressoes_anexos"] and p["data_falta"] != f.data_falta:
                    reutilizados.append(f"'{nome}' já tinha sido entregue no pedido {p['id_email']} "
                                        f"(falta de {p['data_falta']})")
        registar_bloco(log, f"memória consultada: {estado['numero_estudante']}",
                       {"historico": historico, "pedido_original": pedido_original, "reutilizados": reutilizados})
        return {"historico": historico, "pedido_original": pedido_original, "documentos_reutilizados": reutilizados}

    # --- calcular_prazo (código) — agora conta até ao PEDIDO INICIAL, se houver
    async def calcular_prazo(estado: Estado) -> dict:
        data_falta = estado["factos"].data_falta
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data_falta):
            return {"dias_uteis": None}
        original = estado.get("pedido_original")
        data_pedido = original["data_envio"] if original else estado["data_envio"]
        texto, c = await chamar("dias_uteis_entre", data_inicial=data_falta, data_final=data_pedido)
        numero = re.search(r": (\d+) dia", texto)
        return {"dias_uteis": int(numero.group(1)) if numero else None, "chamadas": [c]}

    # --- aplicar_regras (código) — as do 9b, mais as que dependem da memória --
    def aplicar_regras(estado: Estado) -> dict:
        f, dias = estado["factos"], estado.get("dias_uteis")
        original, reutilizados = estado.get("pedido_original"), estado["documentos_reutilizados"]
        docs_do_proprio = all(d.e_o_proprio_estudante for d in f.documentos)
        algum_cobre_falta = any(d.cobre_dia_da_falta for d in f.documentos)
        verificacoes = [
            f"Motivo: {f.motivo} ({f.motivo_descricao})",
            (f"Prazo (art. 2.º e 4.º/3): {dias} dia(s) útil(eis) entre a falta e o "
             + (f"pedido inicial ({original['id_email']}, {original['data_envio']})" if original else "envio")
             + f" (máx. {PRAZO_DIAS_UTEIS})") if dias is not None else "Prazo: não foi possível calcular",
            f"Documentos (art. 3.º): {len(f.documentos)} anexado(s); exigidos completos: "
            f"{'sim' if f.documentos_exigidos_completos else 'não'}",
            f"Identidade (art. 4.º): documentos do próprio estudante: {'sim' if docs_do_proprio else 'NÃO'}",
            f"Datas (art. 4.º): algum documento cobre o dia da falta: {'sim' if algum_cobre_falta else 'NÃO'}",
            f"Histórico: {len(estado['historico'])} pedido(s) anterior(es) deste estudante",
        ]
        alertas = []
        if f.tentativa_manipulacao:
            alertas.append(f"Tentativa de manipulação: {f.descricao_manipulacao}")
        if original:
            alertas.append(f"Complementa o pedido {original['id_email']} (a aguardar documentos)")
        alertas += [f"Documento reutilizado: {r}" for r in reutilizados]

        if f.motivo not in MOTIVOS_ACEITES:
            recomendacao = "propor_recusar"
        elif dias is not None and dias > PRAZO_DIAS_UTEIS:
            recomendacao = "propor_recusar"
        elif not f.documentos or not f.documentos_exigidos_completos:
            recomendacao = "pedir_mais_informacao"
        elif not docs_do_proprio or not algum_cobre_falta or reutilizados:
            recomendacao = "pedir_mais_informacao"
        else:
            recomendacao = "propor_aceitar"
        return {"recomendacao": recomendacao, "verificacoes": verificacoes, "alertas": alertas}

    # --- registar (código) ---------------------------------------------------
    async def registar(estado: Estado) -> dict:
        f = estado["factos"]
        e_justificacao = f.tipo == "justificacao_falta"
        recomendacao = estado["recomendacao"] if e_justificacao else "nao_aplicavel"
        _, c = await chamar(
            "registar_proposta", id_email=estado["id_email"], tipo=f.tipo, recomendacao=recomendacao,
            data_falta=f.data_falta if e_justificacao else "", motivo_invocado=f.motivo_descricao,
            verificacoes="\n".join(estado.get("verificacoes", [])), alertas="\n".join(estado.get("alertas", [])),
        )
        return {"recomendacao": recomendacao, "chamadas": [c]}

    # --- guardar_memoria (código) — NOVO --------------------------------------
    async def guardar_memoria(estado: Estado, runtime: Runtime) -> dict:
        f = estado["factos"]
        # Guardamos só o necessário: nada do texto dos documentos, só as impressões digitais
        await runtime.store.aput(("estudantes", estado["numero_estudante"]), estado["id_email"], {
            "id_email": estado["id_email"], "data_envio": estado["data_envio"], "data_falta": f.data_falta,
            "motivo": f.motivo, "recomendacao": estado["recomendacao"],
            "impressoes_anexos": list(estado["impressoes_anexos"].values()),
        })
        return {}

    # --- redigir_rascunho (LLM) — com o histórico RECENTE no contexto ---------
    async def redigir_rascunho(estado: Estado) -> dict:
        f = estado["factos"]
        recentes = estado["historico"][-HISTORICO_NO_CONTEXTO:]  # gestão de contexto: só os últimos
        historico = "\n".join(f"- pedido {p['id_email']} ({p['data_envio']}): falta de {p['data_falta']}, "
                              f"{p['recomendacao']}" for p in recentes) or "(sem pedidos anteriores)"
        pedido = (
            "Escreve um email curto e cordial, em português de Portugal, dirigido ao estudante "
            f"{f.nome_estudante}, a acusar a receção do pedido de justificação da falta de {f.data_falta}. "
            "NÃO digas se a falta foi aceite ou recusada: a decisão é do docente. "
            + ("Pede que envie o documento em falta ou corrigido, explicando o problema. "
               if estado["recomendacao"] == "pedir_mais_informacao" else "")
            + ("Refere que o documento completa o pedido anterior. " if estado.get("pedido_original") else "")
            + "Escreve só o corpo do email, assinado por 'Serviços do Departamento'.\n\n"
            "Verificações:\n" + "\n".join(estado["verificacoes"])
            + "\n\nAlertas:\n" + ("\n".join(estado["alertas"]) or "(nenhum)")
            + f"\n\nPedidos anteriores do estudante (mais recentes):\n{historico}"
        )
        resposta = await llm.ainvoke(pedido)
        texto = texto_de(resposta.content)
        registar_bloco(log, "LLM: redigir_rascunho (resposta)", texto)
        _, c = await chamar("criar_rascunho", id_email=estado["id_email"], texto=texto)
        return {"chamadas": [c], "tokens": (resposta.usage_metadata or {}).get("total_tokens", 0)}

    repetir_llm = RetryPolicy(max_attempts=3, retry_on=repetir_se)
    grafo = StateGraph(Estado)
    grafo.add_node("carregar", carregar)
    grafo.add_node("extrair_factos", extrair_factos, retry_policy=repetir_llm)
    grafo.add_node("consultar_memoria", consultar_memoria)
    grafo.add_node("calcular_prazo", calcular_prazo)
    grafo.add_node("aplicar_regras", aplicar_regras)
    grafo.add_node("registar", registar)
    grafo.add_node("guardar_memoria", guardar_memoria)
    grafo.add_node("redigir_rascunho", redigir_rascunho, retry_policy=repetir_llm)

    grafo.add_edge(START, "carregar")
    grafo.add_edge("carregar", "extrair_factos")
    grafo.add_conditional_edges(
        "extrair_factos",
        lambda e: "consultar_memoria" if e["factos"].tipo == "justificacao_falta" else "registar",
    )
    grafo.add_edge("consultar_memoria", "calcular_prazo")
    grafo.add_edge("calcular_prazo", "aplicar_regras")
    grafo.add_edge("aplicar_regras", "registar")
    grafo.add_conditional_edges(
        "registar", lambda e: "guardar_memoria" if e["recomendacao"] != "nao_aplicavel" else END,
    )
    grafo.add_edge("guardar_memoria", "redigir_rascunho")
    grafo.add_edge("redigir_rascunho", END)
    return grafo


# ---------------------------------------------------------------------------
# Interface comum (usada pela avaliação do passo 8)
# ---------------------------------------------------------------------------
def criar_memoria_temporaria():
    """Memória que só vive em RAM — a avaliação usa uma nova em cada repetição."""
    return InMemoryStore()


async def analisar(ferramentas, id_email: str, modelo=None, effort=None,
                   log: Path | None = None, memoria=None) -> dict:
    # O store é passado no compile(): fica disponível a todos os nós via runtime.store
    grafo = construir_grafo(ferramentas, modelo, effort, log).compile(store=memoria or InMemoryStore())
    registar_bloco(log, f"########## NOVO EMAIL: {id_email} ##########", "")
    inicio = time.monotonic()
    estado = await grafo.ainvoke({"id_email": id_email, "chamadas": [], "tokens": 0})
    return {
        "chamadas": estado["chamadas"],
        "passos": 1 + (estado["factos"].tipo == "justificacao_falta"),
        "tokens": estado["tokens"],
        "segundos": round(time.monotonic() - inicio, 1),
        "conclusao": f"{estado['recomendacao']} | " + "; ".join(estado.get("alertas", [])),
    }


async def ver_memoria(memoria) -> None:
    itens = await memoria.asearch(("estudantes",), limit=1000)
    if not itens:
        print("(memória vazia)")
    for i in sorted(itens, key=lambda i: (i.namespace, i.key)):
        v = i.value
        print(f"{i.namespace[1]:<14} pedido {v['id_email']}  enviado {v['data_envio']}  "
              f"falta {v['data_falta'] or '-':<10}  {v['recomendacao']:<22} anexos: {len(v['impressoes_anexos'])}")


async def main():
    if "--esquecer" in sys.argv:
        FICHEIRO_MEMORIA.unlink(missing_ok=True)
        print("Memória apagada.")
        return

    FICHEIRO_MEMORIA.parent.mkdir(exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(str(FICHEIRO_MEMORIA)) as memoria:
        await memoria.setup()  # cria as tabelas na primeira utilização
        if "--ver-memoria" in sys.argv:
            await ver_memoria(memoria)
            return

        FICHEIRO_LOG.unlink(missing_ok=True)
        shutil.rmtree(PASTA_SAIDA, ignore_errors=True)
        async with ligar_ferramentas() as ferramentas:
            todos = sorted(p.stem for p in (PASTA_PROJETO / "dados" / "caixa_ficticia").glob("*.eml"))
            ids = sys.argv[1:] or todos
            for id_email in ids:  # POR ORDEM: um email pode depender da memória dos anteriores
                r = await analisar(ferramentas, id_email, log=FICHEIRO_LOG, memoria=memoria)
                print(f"📧 {id_email}  ({r['segundos']}s)  {r['conclusao'][:110]}")

        mostrar_resumo(ids, FICHEIRO_LOG.name)
        print("\nMemória guardada (uv run passo10_memoria.py --ver-memoria):")
        await ver_memoria(memoria)


if __name__ == "__main__":
    asyncio.run(main())
