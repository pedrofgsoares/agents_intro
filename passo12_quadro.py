"""Passo 12 — A "canalização" da experiência multi-agente.

Este ficheiro NÃO tem nada de IA. Trata dos ficheiros partilhados entre os
agentes (passo12_agente.py) e o painel do formador (painel_experiencia.py).
Podes ler só o passo12_agente.py: é lá que está o agente.

Tudo passa por ficheiros na pasta experiencia/:

    quadro.md             o quadro partilhado (a conversa)
    agentes.json          o registo: quem existe, personalidade, modelo, estado, tokens
    logs/<nome>.jsonl     cada chamada de ferramenta de cada agente, com o raciocínio
    logs/<nome>.out       o que o agente escreveu no terminal (quando lançado pelo painel)
    sussurros/<nome>.txt  mensagens privadas à espera de serem lidas por esse agente
    intervencoes.jsonl    tudo o que o formador fez (escrever, sussurrar, mudar, pausar...)
    PAUSA                 se existir, os agentes esperam sem gastar tokens
    PARAR                 se existir, todos os agentes terminam

Como vários programas (um por agente, mais o painel) mexem nos mesmos ficheiros
ao mesmo tempo, cada escrita é feita "trancada" (trancado()): enquanto um
programa escreve, os outros esperam. Sem isto, duas escritas simultâneas
podiam misturar-se e estragar os ficheiros.
"""

import fcntl
import json
import os
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PASTA_PROJETO = Path(__file__).parent
PASTA = PASTA_PROJETO / "experiencia"
QUADRO = PASTA / "quadro.md"
REGISTO = PASTA / "agentes.json"
PASTA_LOGS = PASTA / "logs"
PASTA_SUSSURROS = PASTA / "sussurros"
INTERVENCOES = PASTA / "intervencoes.jsonl"
FICHEIRO_PARAR = PASTA / "PARAR"
FICHEIRO_PAUSA = PASTA / "PAUSA"
PASTA_GUARDADAS = PASTA_PROJETO / "experiencias_guardadas"

TEMA_INICIAL = "Apresentem-se e combinem entre vocês um pequeno problema para resolverem juntos."

NOMES = ["Aurora", "Bento", "Clara", "Duarte", "Estela", "Fabian", "Girafa", "Hugo", "Laura", "Jorge",
         "Leonor", "Miguel", "Nanda", "Oriental", "Paula", "Rui", "Sara", "Tom", "Vera", "Xavier"]


def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# O "cadeado": só um programa de cada vez escreve nos ficheiros partilhados
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


# ---------------------------------------------------------------------------
# Registo dos agentes (agentes.json)
# ---------------------------------------------------------------------------
def ler_registo() -> dict:
    return json.loads(REGISTO.read_text()) if REGISTO.exists() else {}


def _gravar_registo(registo: dict) -> None:
    REGISTO.write_text(json.dumps(registo, ensure_ascii=False, indent=2))


def atualizar_registo(nome: str, **campos) -> None:
    with trancado():
        registo = ler_registo()
        registo.setdefault(nome, {}).update(campos)
        _gravar_registo(registo)


def proximo_nome_livre(excluir: set[str] = frozenset()) -> str:
    usados = set(ler_registo()) | set(excluir)
    livres = [n for n in NOMES if n not in usados]
    return livres[0] if livres else f"Agente{len(usados) + 1}"


def registar_agente(nome_pedido: str | None, personalidade: str, modelo: str) -> str:
    """Escolhe um nome livre (ou valida o pedido) e regista o agente — tudo dentro do lock."""
    with trancado():
        registo = ler_registo()
        if nome_pedido:
            if nome_pedido in registo:
                raise SystemExit(f"O nome '{nome_pedido}' já foi usado nesta experiência. Escolhe outro.")
            nome = nome_pedido
        else:
            livres = [n for n in NOMES if n not in registo]
            nome = livres[0] if livres else f"Agente{len(registo) + 1}"
        registo[nome] = {"personalidade": personalidade, "modelo": modelo, "pid": os.getpid(),
                         "estado": "ativo", "inicio": agora(), "tokens": 0, "rondas": 0}
        _gravar_registo(registo)
        _iniciar_quadro(TEMA_INICIAL)
    return nome


def processo_vivo(pid: int | None) -> bool:
    """O processo com este número ainda existe? (os.kill com sinal 0 só verifica, não mata)"""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# O quadro (quadro.md)
# ---------------------------------------------------------------------------
def _iniciar_quadro(tema: str) -> None:
    # O quadro só é criado se ainda não existir: lançar um agente nunca o apaga
    if not QUADRO.exists():
        QUADRO.write_text(f"[{datetime.now():%H:%M:%S}] sistema: Tema: {tema}\n", encoding="utf-8")


def iniciar_quadro(tema: str) -> bool:
    """Cria o quadro com este tema, se ainda não existir. Devolve True se o criou."""
    with trancado():
        existia = QUADRO.exists()
        _iniciar_quadro(tema)
    return not existia


def escrever_no_quadro(autor: str, texto: str) -> None:
    # Os agentes tendem a imitar o formato do que leem e copiam a marca "(tu)":
    # removemo-la no código (não basta pedir no prompt)
    linha = " ".join(texto.replace("(tu)", "").split())  # uma entrada = uma linha
    # ...e também imitam o cabeçalho das linhas ("[12:36:45] Aurora:", "Bento (eu):"): removemo-lo
    linha = re.sub(rf"^(\[\d{{2}}:\d{{2}}:\d{{2}}\]\s*)?{re.escape(autor)}\s*(\((eu|tu)\))?\s*:\s*", "", linha)
    with trancado(), QUADRO.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {autor}: {linha}\n")


def ler_quadro(nome: str | None = None) -> str:
    """O quadro INTEIRO; se for dado um nome, as mensagens desse agente ficam marcadas com '(tu)'."""
    linhas = QUADRO.read_text(encoding="utf-8").splitlines() if QUADRO.exists() else []
    if nome:
        linhas = [l.replace(f"] {nome}:", f"] {nome} (tu):", 1) for l in linhas]
    return "\n".join(linhas) or "(quadro vazio)"


# ---------------------------------------------------------------------------
# Logs de cada agente
# ---------------------------------------------------------------------------
def registar_log(nome: str, entrada: dict) -> None:
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    with (PASTA_LOGS / f"{nome}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hora": agora(), **entrada}, ensure_ascii=False) + "\n")


def ultimo_raciocinio(nome: str) -> dict | None:
    """A última entrada do log deste agente que traga raciocínio (lê só o fim do ficheiro)."""
    ficheiro = PASTA_LOGS / f"{nome}.jsonl"
    if not ficheiro.exists():
        return None
    with ficheiro.open("rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - 30_000))  # só os últimos ~30 KB
        linhas = f.read().decode("utf-8", errors="ignore").splitlines()
    for linha in reversed(linhas):
        try:
            entrada = json.loads(linha)
        except json.JSONDecodeError:
            continue  # a primeira linha pode ter ficado cortada a meio
        if "raciocinio_modelo" in entrada:
            return entrada
    return None


# ---------------------------------------------------------------------------
# Sussurros: mensagens privadas para um só agente
# ---------------------------------------------------------------------------
def enviar_sussurro(nome: str, autor: str, texto: str) -> None:
    PASTA_SUSSURROS.mkdir(parents=True, exist_ok=True)
    with trancado(), (PASTA_SUSSURROS / f"{nome}.txt").open("a", encoding="utf-8") as f:
        f.write(f"{autor}: {' '.join(texto.split())}\n")


def consumir_sussurros(nome: str) -> list[str]:
    """Lê e APAGA os sussurros deste agente: cada um é entregue uma única vez."""
    ficheiro = PASTA_SUSSURROS / f"{nome}.txt"
    with trancado():
        if not ficheiro.exists():
            return []
        linhas = ficheiro.read_text(encoding="utf-8").splitlines()
        ficheiro.unlink()
    return [l for l in linhas if l.strip()]


# ---------------------------------------------------------------------------
# Pausa e paragem
# ---------------------------------------------------------------------------
def pausar() -> None:
    PASTA.mkdir(exist_ok=True)
    FICHEIRO_PAUSA.touch()


def retomar() -> None:
    FICHEIRO_PAUSA.unlink(missing_ok=True)


def em_pausa() -> bool:
    return FICHEIRO_PAUSA.exists()


def pedir_paragem(nome: str) -> None:
    """Pede a UM agente que pare (no início da sua próxima ronda)."""
    atualizar_registo(nome, parar=True)


def parar_todos() -> None:
    PASTA.mkdir(exist_ok=True)
    FICHEIRO_PARAR.touch()


# ---------------------------------------------------------------------------
# Registo de intervenções do formador
# ---------------------------------------------------------------------------
def registar_intervencao(tipo: str, alvo: str | None = None, conteudo: str = "", **extra) -> None:
    with trancado(), INTERVENCOES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"hora": agora(), "tipo": tipo, "alvo": alvo, "conteudo": conteudo, **extra},
                           ensure_ascii=False) + "\n")


def ler_intervencoes() -> list[dict]:
    if not INTERVENCOES.exists():
        return []
    return [json.loads(l) for l in INTERVENCOES.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Limpar e guardar experiências
# ---------------------------------------------------------------------------
def limpar() -> None:
    shutil.rmtree(PASTA, ignore_errors=True)


def guardar_experiencia(etiqueta: str) -> Path:
    """Copia experiencia/ para experiencias_guardadas/<AAAAMMDD_HHMM>_<etiqueta>/ (os agentes continuam)."""
    etiqueta = re.sub(r"[^\w-]+", "-", etiqueta.strip()).strip("-") or "sem-etiqueta"
    destino = PASTA_GUARDADAS / f"{datetime.now():%Y%m%d_%H%M}_{etiqueta}"
    if destino.exists():
        destino = destino.with_name(f"{destino.name}_{datetime.now():%S}")
    with trancado():  # ninguém escreve enquanto copiamos
        shutil.copytree(PASTA, destino, ignore=shutil.ignore_patterns(".lock"))
    return destino
