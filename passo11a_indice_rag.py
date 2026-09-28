"""Passo 11a — RAG: indexar os regulamentos e pesquisar neles.

O QUE VAIS APRENDER
    RAG (Retrieval-Augmented Generation) = em vez de dar ao LLM os documentos
    todos, procuramos os EXCERTOS relevantes para a pergunta e só esses entram no
    prompt. Tem três peças, todas neste ficheiro:

      1. DIVIDIR (chunking) — cortar os documentos em pedaços. Cortamos pela
         ESTRUTURA do documento (alínea, artigo), não por número de caracteres:
         cada pedaço é uma regra completa e citável ("Normas FEUP, 3.1 d)").
      2. INDEXAR — preparar os pedaços para pesquisa, de duas formas:
           - por palavras (BM25): rápido, local, exato; falha com sinónimos
             ("estive doente" vs "doença aguda")
           - por significado (embeddings): cada texto vira um vetor de números;
             textos com significado parecido ficam próximos. Apanha sinónimos,
             mas é "difuso" e usa uma API externa (bge-m3 na Hugging Face)
      3. PESQUISAR — devolver os pedaços mais relevantes. A pesquisa HÍBRIDA
         junta as duas listas (Reciprocal Rank Fusion).

    E uma quarta, que muitas vezes se esquece: AVALIAR A RECUPERAÇÃO sozinha.
    Se o excerto certo não é encontrado, o LLM nunca o verá — por melhor que seja.

    Não depende de outros passos. É usado pelo mcp_email.py e pelo passo 11b.

COMO CORRER
    uv run passo11a_indice_rag.py --construir               # (re)cria o índice
    uv run passo11a_indice_rag.py "faltei ao funeral do meu avô"
    uv run passo11a_indice_rag.py --avaliar                  # compara os 3 métodos
    uv run passo11a_indice_rag.py --pedacos                  # mostra como ficou a divisão

O QUE OBSERVAR
    (Podes só correr e observar: o importante são as três ideias acima.)
    - --pedacos: cada pedaço é uma regra completa, com a sua referência.
    - Faz a mesma pergunta com palavras diferentes ("estive doente" / "doença").
    - --avaliar: que método encontra mais vezes a regra certa? O híbrido ganha sempre?

EXPERIMENTA MUDAR
    - Pesquisa coisas que NÃO estão nos regulamentos ("faltei porque choveu"): o
      que devolve? (A pesquisa devolve SEMPRE alguma coisa — mesmo que não sirva.)
    - Muda o número de resultados n na pesquisa.

PYTHON NOVO NESTE PASSO
    - enumerate(lista): percorre a lista com o número de cada elemento.
    - zip(a, b): percorre duas listas lado a lado.
    - math: funções matemáticas (aqui: raízes e logaritmos para as pontuações).
    - re.compile / re.split: expressões regulares para cortar o texto pela estrutura.
"""

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import httpx2 as httpx
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

PASTA_REGULAMENTOS = Path(__file__).parent / "dados" / "regulamentos"
FICHEIRO_INDICE = PASTA_REGULAMENTOS / "indice.json"
MODELO_EMBEDDINGS = "BAAI/bge-m3"

# Como dividir cada documento conhecido. Um PDF novo que não esteja aqui é
# dividido "às cegas" por tamanho — funciona, mas as citações ficam piores.
DOCUMENTOS = {
    "Normas_Regime_de_Assiduidad_e_Faltas_-_FEUP.pdf": {
        "id": "feup-normas", "nome": "Normas FEUP - Regime de Assiduidade e de Faltas (2025)", "divisao": "pontos"},
    "DEI - Regulamento 2025.pdf": {
        "id": "dei", "nome": "Regulamento do DEI-FEUP (versão em preparação, 2025)", "divisao": "artigos"},
    "Regulamento_de_exames_VF.pdf": {
        "id": "fpceup-exames", "nome": "Regulamento de Exames da FPCEUP (2008)", "divisao": "tamanho"},
}

# Linhas de cabeçalho/rodapé que se repetem em cada página e só atrapalham
LIXO = re.compile(
    r"^\s*(Aprovado em reunião do Conselho Pedagógico.*|Estrutura de Coordenação dos.*|"
    r"página\s+\d+\s+de\s+\d+|Página \d+ de \d+|UNIVERSIDADE DO PORTO|"
    r"Faculdade de Psicologia e de Ciências da Educação|Regulamento de exames \d/\d)\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. DIVIDIR
# ---------------------------------------------------------------------------
def texto_do_pdf(caminho: Path) -> list[str]:
    linhas = "\n".join(p.extract_text() or "" for p in PdfReader(caminho).pages).splitlines()
    return [l.strip() for l in linhas if l.strip() and not LIXO.match(l)]


def dividir_por_pontos(linhas: list[str]) -> list[dict]:
    """Normas FEUP: '3. Título' > '3.1 - texto' > 'd) texto'. Um pedaço por alínea (ou por ponto)."""
    pedacos, seccao, ponto, intro_ponto, atual = [], "", "", "", None

    def fechar():
        if atual:
            pedacos.append(atual)

    for linha in linhas:
        if m := re.match(r"^(\d)\.\s+(\D.*)$", linha):              # "3. Relevação de faltas"
            fechar()
            seccao, ponto, intro_ponto = f"{m.group(1)}. {m.group(2)}", m.group(1), ""
            atual = {"referencia": m.group(1), "texto": linha}
        elif m := re.match(r"^(\d\.\d)\s*-\s*(.*)$", linha):         # "3.1 - Constituem ..."
            fechar()
            ponto, intro_ponto = m.group(1), m.group(2)
            atual = {"referencia": ponto, "texto": linha}
        elif m := re.match(r"^([a-i])\)\s*(.*)$", linha):            # "d) Doença aguda ..."
            fechar()
            # CONTEXTO: a alínea sozinha ("d) Doença aguda...") não diz que é um motivo
            # de relevação. Juntamos o título da secção e a frase de abertura do ponto.
            atual = {"referencia": f"{ponto} {m.group(1)})",
                     "texto": f"[{seccao} > {ponto}: {intro_ponto}]\n{m.group(1)}) {m.group(2)}"}
        elif atual:
            if atual["referencia"] == ponto and not atual["texto"].count("\n"):
                intro_ponto += " " + linha                            # continuação da abertura do ponto
            atual["texto"] += "\n" + linha
    fechar()
    return pedacos


def dividir_por_artigos(linhas: list[str]) -> list[dict]:
    """Regulamento do DEI: 'Artigo 10º' + título na linha seguinte. Ignora o índice inicial."""
    texto = "\n".join(linhas)
    inicio = texto.find("Artigo 1º")
    partes = re.split(r"\n(?=Artigo\s+\d+\s*º)", "\n" + texto[inicio:])
    pedacos = []
    for parte in partes:
        if m := re.match(r"\s*Artigo\s+(\d+)\s*º\s*\n([^\n]*)", parte):
            pedacos.append({"referencia": f"Artigo {m.group(1)}.º ({m.group(2).strip()})", "texto": parte.strip()})
    return pedacos


def dividir_por_tamanho(linhas: list[str], alvo: int = 700) -> list[dict]:
    """Sem estrutura reconhecível: blocos de ~700 caracteres, sem cortar frases a meio."""
    frases = re.split(r"(?<=[.;:])\s+", " ".join(linhas))
    pedacos, atual = [], ""
    for frase in frases:
        if len(atual) + len(frase) > alvo and atual:
            pedacos.append(atual)
            atual = ""
        atual += frase + " "
    if atual.strip():
        pedacos.append(atual)
    return [{"referencia": f"bloco {i + 1}", "texto": t.strip()} for i, t in enumerate(pedacos)]


def dividir_documentos() -> list[dict]:
    # Os PDFs não vêm com o material (git): cada formando descarrega-os (ver README).
    # As Normas FEUP são indispensáveis; os outros dois são distratores (opcionais).
    normas = next(iter(DOCUMENTOS))
    if not (PASTA_REGULAMENTOS / normas).exists():
        raise FileNotFoundError(
            f"Faltam regulamentos: não encontrei dados/regulamentos/{normas}. "
            "Descarrega os PDFs e grava-os com os nomes exatos indicados em 'Regulamentos' no README.")
    divisores = {"pontos": dividir_por_pontos, "artigos": dividir_por_artigos, "tamanho": dividir_por_tamanho}
    pedacos = []
    for pdf in sorted(PASTA_REGULAMENTOS.glob("*.pdf")):
        info = DOCUMENTOS.get(pdf.name, {"id": pdf.stem, "nome": pdf.stem, "divisao": "tamanho"})
        for p in divisores[info["divisao"]](texto_do_pdf(pdf)):
            pedacos.append({"id": f"{info['id']}:{p['referencia']}", "documento_id": info["id"],
                            "documento": info["nome"], **p})
    return pedacos


# ---------------------------------------------------------------------------
# 2. INDEXAR — por palavras (BM25)
# ---------------------------------------------------------------------------
PALAVRAS_VAZIAS = set("""a o as os um uma de do da dos das em no na nos nas por pelo pela para com sem
que se ao aos e ou é foi ser sao seu sua seus suas este esta isso me eu meu minha lhe mais nao
como tem ter ja ate sobre entre quando onde qual quais""".split())


def palavras(texto: str) -> list[str]:
    # minúsculas, sem acentos, sem palavras vazias, e um "stemming" grosseiro:
    # só os primeiros 6 caracteres ("faltas"/"faltou" -> "faltas"/"faltou"; "justificação"/"justificar" -> "justif")
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w[:6] for w in re.findall(r"[a-z0-9]+", t) if w not in PALAVRAS_VAZIAS and len(w) > 2]


class BM25:
    """O algoritmo clássico dos motores de pesquisa, em 20 linhas.

    Um pedaço pontua mais se contiver as palavras da pergunta, sobretudo as
    RARAS (idf: 'atestado' vale mais que 'faltas', que está em todo o lado),
    com retorno decrescente (k1) e penalizando pedaços longos (b).
    """

    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(palavras(t)) for t in textos]
        self.tamanhos = [sum(d.values()) for d in self.docs]
        self.media = sum(self.tamanhos) / len(self.docs)
        n = len(self.docs)
        contagem = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in contagem.items()}
        self.k1, self.b = k1, b

    def pontuar(self, pergunta: str) -> list[float]:
        termos = palavras(pergunta)
        return [
            sum(self.idf.get(w, 0) * d[w] * (self.k1 + 1)
                / (d[w] + self.k1 * (1 - self.b + self.b * tam / self.media)) for w in termos if w in d)
            for d, tam in zip(self.docs, self.tamanhos)
        ]


# ---------------------------------------------------------------------------
# 2. INDEXAR — por significado (embeddings)
# ---------------------------------------------------------------------------
def embeddings(textos: list[str]) -> list[list[float]]:
    url = f"https://router.huggingface.co/hf-inference/models/{MODELO_EMBEDDINGS}/pipeline/feature-extraction"
    vetores = []
    for i in range(0, len(textos), 16):  # em lotes, para não exceder limites do pedido
        r = httpx.post(url, headers={"Authorization": f"Bearer {os.environ['HF_API_KEY']}"},
                       json={"inputs": textos[i:i + 16]}, timeout=120)
        r.raise_for_status()
        vetores += r.json()
    # normalizar (comprimento 1): assim o produto interno É a semelhança do cosseno
    return [[x / math.sqrt(sum(v * v for v in vetor)) for x in vetor] for vetor in vetores]


def assinatura_pdfs() -> str:
    """Muda se algum PDF mudar -> o índice tem de ser reconstruído."""
    h = hashlib.sha256(MODELO_EMBEDDINGS.encode())
    for pdf in sorted(PASTA_REGULAMENTOS.glob("*.pdf")):
        h.update(pdf.name.encode() + pdf.read_bytes())
    return h.hexdigest()[:16]


def construir_indice() -> dict:
    pedacos = dividir_documentos()
    vetores = embeddings([p["texto"] for p in pedacos])
    indice = {"assinatura": assinatura_pdfs(), "modelo": MODELO_EMBEDDINGS,
              "pedacos": [{**p, "vetor": v} for p, v in zip(pedacos, vetores)]}
    FICHEIRO_INDICE.write_text(json.dumps(indice, ensure_ascii=False))
    return indice


class Indice:
    def __init__(self):
        dados = json.loads(FICHEIRO_INDICE.read_text()) if FICHEIRO_INDICE.exists() else None
        if not dados or dados["assinatura"] != assinatura_pdfs():
            dados = construir_indice()   # só volta a pagar embeddings se os PDFs mudarem
        self.pedacos = dados["pedacos"]
        self.bm25 = BM25([p["texto"] for p in self.pedacos])

    # -----------------------------------------------------------------------
    # 3. PESQUISAR
    # -----------------------------------------------------------------------
    def _ordenar(self, pontuacoes: list[float]) -> list[int]:
        return sorted(range(len(pontuacoes)), key=lambda i: -pontuacoes[i])

    def por_palavras(self, pergunta: str) -> list[int]:
        return self._ordenar(self.bm25.pontuar(pergunta))

    def por_significado(self, pergunta: str) -> list[int]:
        q = embeddings([pergunta])[0]
        return self._ordenar([sum(a * b for a, b in zip(q, p["vetor"])) for p in self.pedacos])

    def hibrida(self, pergunta: str) -> list[int]:
        # Reciprocal Rank Fusion: não soma pontuações (escalas diferentes!), soma POSIÇÕES.
        # Ficar em 1.º numa lista vale 1/61, em 10.º vale 1/70...
        pontos = Counter()
        for ordem in (self.por_palavras(pergunta), self.por_significado(pergunta)):
            for posicao, i in enumerate(ordem):
                pontos[i] += 1 / (60 + posicao + 1)
        return [i for i, _ in pontos.most_common()]

    def pesquisar(self, pergunta: str, n: int = 5, metodo: str = "hibrida") -> list[dict]:
        ordem = {"palavras": self.por_palavras, "significado": self.por_significado, "hibrida": self.hibrida}[metodo]
        return [{k: v for k, v in self.pedacos[i].items() if k != "vetor"} for i in ordem(pergunta)[:n]]


# ---------------------------------------------------------------------------
# 4. AVALIAR A RECUPERAÇÃO
# Perguntas escritas como um ESTUDANTE as escreveria (não com as palavras do
# regulamento!) e os pedaços que deviam aparecer.
# ---------------------------------------------------------------------------
PERGUNTAS_TESTE = [
    ("Faltei à aula porque estive doente, envio o atestado do médico", {"feup-normas:3.1 d)"}),
    ("Até quando posso enviar a justificação da falta?", {"feup-normas:3.2"}),
    ("Faltei porque fui ao funeral do meu avô", {"feup-normas:3.1 b)"}),
    ("O meu pai faleceu na semana passada", {"feup-normas:3.1 a)"}),
    ("Sou trabalhador-estudante e tive de ir trabalhar", {"feup-normas:1.3"}),
    ("Fiquei em casa a cuidar da minha mãe que estava doente", {"feup-normas:3.1 e)"}),
    ("Tive de ir às urgências do hospital", {"feup-normas:3.1 c)"}),
    ("Fui notificado para ir a tribunal como testemunha", {"feup-normas:3.1 h)"}),
    ("Estive numa reunião da associação de estudantes", {"feup-normas:3.1 g)"}),
    ("Representei a faculdade num torneio de futebol", {"feup-normas:3.1 f)"}),
    ("Pediram-me o papel original do atestado, quanto tempo tenho?", {"feup-normas:3.4"}),
    ("Quantas faltas posso dar no máximo numa cadeira?", {"feup-normas:1.2 a)"}),
    ("Faltei a um teste a meio do semestre", {"feup-normas:2.1", "feup-normas:2.2"}),
]


def avaliar(indice: Indice, n: int = 3) -> None:
    print(f"{'pergunta':<52} {'palavras':>9} {'signif.':>9} {'híbrida':>9}")
    totais = Counter()
    for pergunta, esperados in PERGUNTAS_TESTE:
        linha = f"{pergunta[:50]:<52}"
        for metodo in ("palavras", "significado", "hibrida"):
            ids = [p["id"] for p in indice.pesquisar(pergunta, n=10, metodo=metodo)]
            posicao = next((i + 1 for i, x in enumerate(ids) if x in esperados), None)
            acertou = posicao is not None and posicao <= n
            totais[metodo, "acertos"] += acertou
            totais[metodo, "mrr"] += 1 / posicao if posicao else 0
            linha += f" {('✅ ' if acertou else '❌ ') + (str(posicao) if posicao else '>10'):>9}"
        print(linha)
    print("-" * 82)
    k = len(PERGUNTAS_TESTE)
    print(f"{f'encontrado no top-{n}':<52}" + "".join(f" {totais[m, 'acertos'] / k:>8.0%}" for m in ("palavras", "significado", "hibrida")))
    print(f"{'MRR (1 = sempre em 1.º lugar)':<52}" + "".join(f" {totais[m, 'mrr'] / k:>9.2f}" for m in ("palavras", "significado", "hibrida")))


if __name__ == "__main__":
    if "--construir" in sys.argv:
        FICHEIRO_INDICE.unlink(missing_ok=True)
    if "--pedacos" in sys.argv:
        for p in dividir_documentos():
            print(f"── {p['id']}  ({len(p['texto'])} car.)\n{p['texto'][:220]}\n")
        sys.exit()
    indice = Indice()
    print(f"Índice: {len(indice.pedacos)} pedaços de {len({p['documento_id'] for p in indice.pedacos})} documentos\n")
    if "--avaliar" in sys.argv:
        avaliar(indice)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        for metodo in ("palavras", "significado", "hibrida"):
            print(f"### {metodo}")
            for p in indice.pesquisar(sys.argv[1], n=3, metodo=metodo):
                print(f"   {p['id']:<45} {p['texto'].splitlines()[-1][:70]}")
