"""Painel do formador para a experiência multi-agente (passo 12).

Uma página web, para projetar, onde se acompanha a conversa dos agentes e se
intervém nela. Não precisas de perceber este ficheiro: o agente está em
passo12_agente.py. O painel não tem "magia": tudo o que faz é ler e escrever
os mesmos ficheiros da pasta experiencia/ que os agentes usam (ver
passo12_quadro.py). Podes abrir essa pasta durante a experiência e ver.

    Acompanhar: o quadro em direto, o estado de cada agente, o raciocínio
                mais recente de cada um e as intervenções privadas.
    Intervir:   falar como humano (ex.: "Deus"), publicar um evento do narrador,
                sussurrar a um só agente, mudar a personalidade ou o modelo de um agente.
    Gerir:      tema inicial, lançar e parar agentes, pausar, parar todos,
                guardar a experiência para comparar variações, limpar.

Uso:
    uv run painel_experiencia.py
    e abrir no browser o endereço que aparece (http://127.0.0.1:7860).
"""

import html
import re
import subprocess
import sys

import gradio as gr

import passo12_quadro as quadro
from passo12_agente import MAX_RONDAS, MODELO_POR_OMISSAO, MODELOS_SUGERIDOS, PERSONALIDADES

# Os agentes lançados por este painel. Guardamo-los para o sistema operativo
# "arrumar" os que terminam (poll), senão ficariam como processos-fantasma.
PROCESSOS: list[subprocess.Popen] = []
# Nomes já lançados: o agente só se regista quando o processo arranca (1-2 s depois),
# por isso o registo sozinho não chega para evitar nomes repetidos.
LANCADOS: set[str] = set()

CORES = ["#2563eb", "#16a34a", "#c026d3", "#ea580c", "#0891b2", "#9333ea", "#dc2626", "#65a30d"]
MENSAGENS_VISIVEIS = 40
LINHA = re.compile(r"^\[(\d\d:\d\d:\d\d)\] ([^:]+): (.*)$")


# ---------------------------------------------------------------------------
# Ler o estado da experiência (chamado ~1 vez por segundo)
# ---------------------------------------------------------------------------
def agentes_vivos() -> list[str]:
    for p in PROCESSOS:
        p.poll()
    return [n for n, a in quadro.ler_registo().items()
            if not a.get("estado", "").startswith("terminou") and quadro.processo_vivo(a.get("pid"))]


def html_quadro() -> str:
    registo = quadro.ler_registo()
    cor = {nome: CORES[i % len(CORES)] for i, nome in enumerate(registo)}
    linhas = quadro.QUADRO.read_text(encoding="utf-8").splitlines() if quadro.QUADRO.exists() else []
    if not linhas:
        return "<div class='quadro vazio'>O quadro está vazio. Adiciona um agente para começar.</div>"
    partes = []
    if len(linhas) > MENSAGENS_VISIVEIS:
        partes.append(f"<div class='msg sistema'>… {len(linhas) - MENSAGENS_VISIVEIS} mensagens anteriores "
                      "(quadro completo em experiencia/quadro.md)</div>")
    for linha in linhas[-MENSAGENS_VISIVEIS:]:
        m = LINHA.match(linha)
        if not m:
            continue
        hora, autor, texto = m.groups()
        autor_h, texto_h = html.escape(autor), html.escape(texto)
        if autor == "sistema":
            partes.append(f"<div class='msg sistema'><span class='hora'>{hora}</span> ⚙️ {texto_h}</div>")
        elif autor == "narrador":
            partes.append(f"<div class='msg narrador'><span class='hora'>{hora}</span> 📰 <b>Narrador:</b> {texto_h}</div>")
        elif autor in registo:
            partes.append(f"<div class='msg agente' style='border-left-color:{cor[autor]}'>"
                          f"<span class='hora'>{hora}</span> <b style='color:{cor[autor]}'>{autor_h}</b> {texto_h}</div>")
        else:  # um humano (ex.: "Deus")
            partes.append(f"<div class='msg humano'><span class='hora'>{hora}</span> 🗣️ <b>{autor_h}:</b> {texto_h}</div>")
    return "<div class='quadro'>" + "\n".join(partes) + "</div>"


def tabela_agentes() -> list[list]:
    linhas = []
    for nome, a in quadro.ler_registo().items():
        estado = a.get("estado", "?")
        if not estado.startswith("terminou") and not quadro.processo_vivo(a.get("pid")):
            estado = "⚠️ o processo terminou sem avisar"
        elif a.get("ultimo_erro"):
            estado += " ⚠️ erro na última ronda"
        linhas.append([nome, a.get("personalidade", "")[:45], a.get("modelo", "?").split("/")[-1],
                       estado, a.get("rondas", 0), a.get("tokens", 0)])
    return linhas or [["", "", "", "(nenhum agente)", 0, 0]]


def md_raciocinio(mostrar: bool) -> str:
    if not mostrar:
        return "_(raciocínio escondido)_"
    blocos = []
    for nome in quadro.ler_registo():
        e = quadro.ultimo_raciocinio(nome)
        if not e:
            continue
        acao = e.get("ferramenta") or "sem ferramenta"
        texto = (e.get("raciocinio_modelo") or "").strip() or "(este modelo não mostra o raciocínio)"
        blocos.append(f"**{nome}** · ronda {e.get('ronda', '?')} · {acao}\n\n> {texto[:500]}")
    return "\n\n".join(blocos) or "_(ainda sem raciocínios)_"


def md_privado() -> str:
    """Sussurros e mudanças que NÃO aparecem no quadro (os agentes não as veem)."""
    icones = {"sussurro": "🤫", "personalidade": "🎭", "modelo": "🧠", "parar_agente": "⏹", "novo_agente": "➕"}
    linhas = []
    for i in quadro.ler_intervencoes()[-12:]:
        if i["tipo"] not in icones:
            continue
        hora = i["hora"][11:]
        if i["tipo"] == "sussurro":
            linhas.append(f"- {hora} 🤫 **{i.get('autor', '?')} → {i['alvo']}**: {i['conteudo']}")
        elif i["tipo"] == "novo_agente":
            linhas.append(f"- {hora} ➕ **{i['alvo']}** entrou ({i.get('modelo', '').split('/')[-1]})")
        else:
            linhas.append(f"- {hora} {icones[i['tipo']]} **{i['alvo']}**: {i['conteudo'][:80]}")
    return "\n".join(reversed(linhas)) or "_(nada ainda)_"


def md_estado() -> str:
    vivos = agentes_vivos()
    if quadro.FICHEIRO_PARAR.exists():
        situacao = "⏹ **PARAGEM PEDIDA** (os agentes terminam no fim da ronda)"
    elif quadro.em_pausa():
        situacao = "⏸ **EM PAUSA** (cada agente acaba a ronda em curso e depois espera, sem gastar tokens)"
    else:
        situacao = "▶️ **A decorrer**"
    return f"{situacao} · {len(vivos)} agente(s) ativo(s)"


def atualizar(mostrar_raciocinio: bool, nomes_antes: list[str]):
    nomes = list(quadro.ler_registo())
    lista = gr.update(choices=nomes) if nomes != nomes_antes else gr.update()
    return (html_quadro(), tabela_agentes(), md_raciocinio(mostrar_raciocinio), md_privado(),
            md_estado(), lista, nomes)


# ---------------------------------------------------------------------------
# Ações do formador
# ---------------------------------------------------------------------------
def adicionar_agente(personalidade: str, modelo: str, nome: str, max_tokens: float, rondas: float, tema: str):
    personalidade, modelo, nome = (personalidade or "").strip(), (modelo or "").strip(), (nome or "").strip()
    if not personalidade or not modelo:
        raise gr.Error("Escolhe uma personalidade e um modelo.")
    if quadro.FICHEIRO_PARAR.exists():
        if agentes_vivos():
            raise gr.Error("Ainda há agentes a terminar (paragem pedida). Espera uns segundos.")
        quadro.FICHEIRO_PARAR.unlink()  # a paragem anterior já terminou: pode-se continuar
    if quadro.iniciar_quadro(tema.strip() or quadro.TEMA_INICIAL):  # é uma experiência nova
        LANCADOS.clear()
        quadro.registar_intervencao("tema", None, tema.strip() or quadro.TEMA_INICIAL)
    nome = nome or quadro.proximo_nome_livre(excluir=LANCADOS)
    if nome in quadro.ler_registo() or nome in LANCADOS:
        raise gr.Error(f"O nome '{nome}' já foi usado nesta experiência.")
    LANCADOS.add(nome)

    quadro.PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    saida = (quadro.PASTA_LOGS / f"{nome}.out").open("w", encoding="utf-8")
    PROCESSOS.append(subprocess.Popen(
        [sys.executable, str(quadro.PASTA_PROJETO / "passo12_agente.py"), "--nome", nome,
         "--personalidade", personalidade, "--modelo", modelo,
         "--max-tokens", str(int(max_tokens)), "--rondas", str(int(rondas))],
        cwd=quadro.PASTA_PROJETO, stdout=saida, stderr=subprocess.STDOUT,
        start_new_session=True,  # não morre com um Ctrl+C no terminal do painel
    ))
    quadro.registar_intervencao("novo_agente", nome, PERSONALIDADES.get(personalidade, personalidade), modelo=modelo)
    gr.Info(f"{nome} vai entrar na experiência.")
    return ""  # limpa o campo do nome


def falar(autor: str, texto: str):
    autor, texto = (autor or "").strip() or "humano", (texto or "").strip()
    if not texto:
        raise gr.Error("Escreve a mensagem.")
    if autor in quadro.ler_registo() or autor in ("sistema", "narrador"):
        raise gr.Error(f"'{autor}' é um nome reservado ou de um agente. Escolhe outro autor.")
    quadro.iniciar_quadro(quadro.TEMA_INICIAL)
    quadro.escrever_no_quadro(autor, texto)
    quadro.registar_intervencao("humano", autor, texto)
    return ""


def narrar(texto: str):
    texto = (texto or "").strip()
    if not texto:
        raise gr.Error("Escreve o evento.")
    quadro.iniciar_quadro(quadro.TEMA_INICIAL)
    quadro.escrever_no_quadro("narrador", texto)
    quadro.registar_intervencao("narrador", None, texto)
    return ""


def sussurrar(agente: str, autor: str, texto: str):
    texto = (texto or "").strip()
    if not agente or not texto:
        raise gr.Error("Escolhe o agente e escreve a mensagem.")
    autor = (autor or "").strip() or "alguém"
    quadro.enviar_sussurro(agente, autor, texto)
    quadro.registar_intervencao("sussurro", agente, texto, autor=autor)
    gr.Info(f"{agente} recebe o sussurro no início da próxima ronda.")
    return ""


def mudar_agente(agente: str, personalidade: str, modelo: str):
    if not agente:
        raise gr.Error("Escolhe o agente.")
    atual = quadro.ler_registo().get(agente, {})
    personalidade = PERSONALIDADES.get((personalidade or "").strip(), (personalidade or "").strip())
    modelo = (modelo or "").strip()
    mudancas = {}
    if personalidade and personalidade != atual.get("personalidade"):
        mudancas["personalidade"] = personalidade
        quadro.registar_intervencao("personalidade", agente, personalidade, anterior=atual.get("personalidade"))
    if modelo and modelo != atual.get("modelo"):
        mudancas["modelo"] = modelo
        quadro.registar_intervencao("modelo", agente, modelo, anterior=atual.get("modelo"))
    if not mudancas:
        raise gr.Error("Não há nada para mudar (escolhe uma personalidade ou um modelo diferentes).")
    quadro.atualizar_registo(agente, **mudancas)
    gr.Info(f"{agente} muda a partir da próxima ronda.")


def parar_agente(agente: str):
    if not agente:
        raise gr.Error("Escolhe o agente.")
    quadro.pedir_paragem(agente)
    quadro.registar_intervencao("parar_agente", agente, "parado pelo formador")
    gr.Info(f"{agente} termina no fim da ronda.")


def pausar():
    quadro.pausar()
    quadro.registar_intervencao("pausa")


def retomar():
    quadro.retomar()
    quadro.registar_intervencao("retomar")


def parar_todos():
    quadro.parar_todos()
    quadro.registar_intervencao("parar_todos")


def guardar(etiqueta: str):
    if not quadro.PASTA.exists():
        raise gr.Error("Não há nenhuma experiência para guardar.")
    destino = quadro.guardar_experiencia(etiqueta or "")
    gr.Info(f"Guardada em {destino.relative_to(quadro.PASTA_PROJETO)}")
    return ""


def limpar(confirmado: bool):
    if not confirmado:
        raise gr.Error("Marca a confirmação primeiro: apaga o quadro, o registo e os logs.")
    if agentes_vivos():
        raise gr.Error("Ainda há agentes a correr. Usa 'Parar todos' e espera que terminem.")
    quadro.limpar()
    LANCADOS.clear()
    gr.Info("Experiência apagada. (As experiências guardadas não foram tocadas.)")
    return False


# ---------------------------------------------------------------------------
# A página
# ---------------------------------------------------------------------------
CSS = """
body, .gradio-container { font-size: 18px !important; }
.quadro { display: flex; flex-direction: column; gap: 6px; font-size: 19px; line-height: 1.4; }
.quadro.vazio { opacity: .6; padding: 24px; }
.msg { padding: 6px 10px; border-radius: 6px; border-left: 5px solid transparent; }
.msg .hora { font-size: 13px; opacity: .55; font-family: monospace; }
.msg.agente { background: rgba(127,127,127,.08); }
.msg.humano { background: rgba(250,204,21,.22); border-left-color: #ca8a04; }
.msg.narrador { background: rgba(249,115,22,.16); border-left-color: #ea580c; font-style: italic; }
.msg.sistema { opacity: .6; font-size: 15px; }
"""


def construir() -> gr.Blocks:
    personalidades = list(PERSONALIDADES)
    with gr.Blocks(title="Experiência multi-agente", fill_width=True) as painel:
        nomes = gr.State([])
        ativos = agentes_vivos()
        gr.Markdown("## 🤖 Experiência multi-agente — quadro partilhado")
        if ativos:
            gr.Markdown(f"⚠️ **Já há agentes a correr:** {', '.join(ativos)}. O painel mostra-os normalmente.")
        estado = gr.Markdown(md_estado())

        with gr.Row():
            # ---------------- Coluna esquerda: a conversa e as intervenções públicas
            with gr.Column(scale=3):
                vista_quadro = gr.HTML(html_quadro(), js_on_load=None)
                with gr.Tabs():
                    with gr.Tab("🗣️ Falar no quadro"):
                        with gr.Row():
                            autor = gr.Textbox(value="Deus", label="Como quem?", scale=1)
                            texto_falar = gr.Textbox(label="Mensagem (todos os agentes a podem ler)", scale=4)
                        gr.Button("Publicar no quadro", variant="primary").click(
                            falar, [autor, texto_falar], texto_falar)
                    with gr.Tab("📰 Narrador"):
                        texto_narrar = gr.Textbox(label="Evento (ex.: 'Chegou a notícia de que o orçamento foi cortado')")
                        gr.Button("Publicar evento", variant="primary").click(narrar, texto_narrar, texto_narrar)
                    with gr.Tab("🤫 Sussurrar"):
                        with gr.Row():
                            autor_sussurro = gr.Textbox(value="formador", label="De quem?", scale=1)
                            texto_sussurro = gr.Textbox(label="Mensagem PRIVADA para o agente escolhido à direita",
                                                        scale=4)
                        botao_sussurro = gr.Button("Sussurrar", variant="primary")
                    with gr.Tab("🎭 Mudar agente"):
                        with gr.Row():
                            nova_personalidade = gr.Dropdown(personalidades, value=None, allow_custom_value=True,
                                                             label="Nova personalidade (ou escreve uma)")
                            novo_modelo = gr.Dropdown(MODELOS_SUGERIDOS, value=None, allow_custom_value=True,
                                                      label="Novo modelo (ou escreve um)")
                        botao_mudar = gr.Button("Aplicar ao agente escolhido à direita", variant="primary")

            # ---------------- Coluna direita: agentes, controlos e bastidores
            with gr.Column(scale=2):
                tabela = gr.Dataframe(tabela_agentes(), headers=["nome", "personalidade", "modelo", "estado",
                                                                 "ronda", "tokens"],
                                      interactive=False, wrap=True, label="Agentes")
                with gr.Row():
                    agente = gr.Dropdown([], label="Agente escolhido", scale=3)
                    gr.Button("⏹ Parar este", scale=1).click(parar_agente, agente)
                botao_sussurro.click(sussurrar, [agente, autor_sussurro, texto_sussurro], texto_sussurro)
                botao_mudar.click(mudar_agente, [agente, nova_personalidade, novo_modelo])
                with gr.Row():
                    gr.Button("⏸ Pausar").click(pausar)
                    gr.Button("▶️ Retomar").click(retomar)
                    gr.Button("⏹ Parar todos", variant="stop").click(parar_todos)

                with gr.Accordion("➕ Adicionar agente", open=True):
                    tema = gr.Textbox(quadro.TEMA_INICIAL, label="Tema (usado quando entra o primeiro agente)")
                    with gr.Row():
                        personalidade = gr.Dropdown(personalidades, value="colaborativo", allow_custom_value=True,
                                                    label="Personalidade (ou escreve uma)")
                        modelo = gr.Dropdown(MODELOS_SUGERIDOS, value=MODELO_POR_OMISSAO, allow_custom_value=True,
                                             label="Modelo (ou escreve um)")
                    with gr.Row():
                        nome = gr.Textbox(label="Nome (opcional)")
                        rondas = gr.Number(max(MAX_RONDAS, 20), label="Rondas", precision=0)
                        max_tokens = gr.Number(40000, label="Máx. tokens", precision=0)
                    gr.Button("Adicionar agente", variant="primary").click(
                        adicionar_agente, [personalidade, modelo, nome, max_tokens, rondas, tema], nome)

                with gr.Accordion("🤫 Bastidores (o que os agentes não veem no quadro)", open=True):
                    privado = gr.Markdown(md_privado())
                with gr.Accordion("🧠 Raciocínio mais recente", open=True):
                    mostrar = gr.Checkbox(True, label="Mostrar raciocínio")
                    raciocinio = gr.Markdown(md_raciocinio(True))

                with gr.Accordion("💾 Guardar / limpar", open=False):
                    with gr.Row():
                        etiqueta = gr.Textbox(label="Etiqueta (ex.: com-deus)", scale=3)
                        gr.Button("Guardar como…", scale=1).click(guardar, etiqueta, etiqueta)
                    with gr.Row():
                        confirmar = gr.Checkbox(False, label="Sim, quero apagar a experiência atual")
                        gr.Button("🗑 Limpar", variant="stop").click(limpar, confirmar, confirmar)

        # Atualização automática, ~1 vez por segundo (sem indicador de "a carregar")
        gr.Timer(1.0).tick(atualizar, [mostrar, nomes],
                           [vista_quadro, tabela, raciocinio, privado, estado, agente, nomes],
                           show_progress="hidden")
    return painel


if __name__ == "__main__":
    construir().launch(css=CSS)
