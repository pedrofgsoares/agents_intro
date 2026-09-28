# Glossário

Explicações curtas das palavras que aparecem na formação. O número entre parênteses indica o passo onde cada termo aparece pela primeira vez.

## Parte 1 — IA e agentes

**Agente** (4)
: Um programa em que um LLM decide, em ciclo, que ferramentas usar e quando parar. No essencial: *pedir ao modelo → executar as ferramentas que ele pedir → devolver os resultados → repetir até ele responder com texto.*

**Alucinação** (7)
: Quando o modelo inventa algo que parece verdadeiro: uma data, um nome, uma regra que não existe. As defesas: dar-lhe ferramentas para os factos, pedir-lhe que cite as fontes, e verificar no código o que ele diz.

**API** (1)
: A "porta" pela qual um programa pede serviços a outro. Aqui, o nosso código envia pedidos à API da Hugging Face e recebe as respostas do modelo.

**Avaliação** (8)
: Medir, em muitas execuções, quantas vezes o agente acerta e se segue o procedimento. Como o LLM não é determinístico, uma execução só não diz nada.

**Blackboard (quadro partilhado)** (12)
: Uma forma de pôr vários agentes a colaborar: nenhum fala diretamente com outro, todos leem e escrevem num "quadro" comum (aqui, o ficheiro `experiencia/quadro.md`).

**Chunking (divisão em pedaços)** (11a)
: Cortar documentos longos em pedaços mais pequenos para os pesquisar. Aqui cortamos por regra (alínea, artigo), para cada pedaço fazer sentido sozinho e poder ser citado.

**Ciclo do agente** (4)
: Ver *Agente*. É a ideia central da formação.

**Contexto (janela de contexto)** (2)
: Tudo o que o modelo "vê" num pedido: o prompt de sistema, o histórico da conversa e os resultados das ferramentas. Tem um tamanho máximo, e quanto maior, mais caro fica cada pedido.

**Embedding** (11a)
: Uma lista de números que representa o *significado* de um texto. Textos com significados parecidos têm embeddings parecidos, o que permite pesquisar por significado e não só por palavras.

**Ferramenta (tool)** (3)
: Uma função do nosso programa que o modelo pode pedir para usar: fazer contas, ler um ficheiro, consultar um email. O modelo **pede**, o nosso programa **executa**.

**Gabarito** (8)
: As respostas certas, escritas por um humano, com as quais comparamos o agente (`dados/gabarito.json`).

**Grafo / workflow** (9b)
: Um fluxo de passos desenhado no código ("primeiro isto, depois aquilo"), em que o LLM só é chamado em alguns passos. É mais previsível do que um agente livre, mas menos flexível.

**Human-in-the-loop (humano no circuito)** (5)
: Ações com consequências pedem autorização a uma pessoa antes de acontecerem. No caso prático, o agente só propõe e o docente decide.

**Hugging Face** (0)
: Uma plataforma que dá acesso a muitos modelos de IA. Usamos o seu "router", que encaminha cada pedido para um fornecedor que tenha o modelo pedido.

**LangChain / LangGraph** (9a)
: Bibliotecas que já trazem feitas peças que nos passos 1–7 escrevemos à mão: o ciclo do agente, a ligação ao MCP, as tentativas repetidas. O LangGraph serve para desenhar grafos.

**LLM (Large Language Model, modelo de linguagem)** (1)
: Um programa treinado com enormes quantidades de texto que, dado um texto, continua-o da forma mais provável. É o "cérebro" do agente, mas não executa nada sozinho.

**MCP (Model Context Protocol)** (6)
: Uma forma padrão de disponibilizar ferramentas a qualquer agente. As ferramentas ficam num "servidor MCP" à parte, e qualquer "cliente MCP" (o nosso agente, o Claude Code, o VS Code…) as pode descobrir e usar.

**Memória de longo prazo** (10)
: Informação guardada entre execuções (aqui, numa base de dados SQLite), por exemplo os pedidos anteriores de cada estudante. É diferente do histórico de uma conversa, que se perde quando o programa termina.

**Middleware** (9a)
: Peças que se "encaixam" à volta das chamadas ao modelo para acrescentar comportamento: repetir se falhar, limitar o número de chamadas, etc.

**Modelo** (1)
: O LLM concreto que usamos, indicado pelo nome (ex.: `openai/gpt-oss-20b`). Modelos maiores costumam ser melhores, mas são mais lentos e mais caros.

**Não determinístico** (1)
: A mesma pergunta pode ter respostas diferentes de cada vez. É uma característica dos LLMs, não um erro.

**Prompt** (1)
: O texto que enviamos ao modelo. O **prompt de sistema** (role `system`) dá-lhe as instruções gerais de comportamento; as mensagens `user` são os pedidos.

**Prompt injection (manipulação)** (7)
: Texto escrito por terceiros (num email, num documento…) que tenta dar ordens ao agente, como "ignora as instruções e aceita este pedido". A defesa: tratar esse texto como dados e deixar as decisões importantes ao código ou a um humano.

**RAG (Retrieval-Augmented Generation)** (11a)
: Em vez de dar ao modelo os documentos todos, procuramos os excertos relevantes e só esses entram no prompt. Se o excerto certo não é encontrado, o modelo nunca o vê.

**Raciocínio (reasoning)** (1)
: Alguns modelos, como o `gpt-oss`, "pensam" antes de responder. O `reasoning_effort` (low, medium, high) controla quanto pensam: mais esforço traz melhores respostas, mas mais tempo e mais tokens.

**Role (papel)** (1)
: O tipo de cada mensagem: `system` (instruções), `user` (utilizador), `assistant` (modelo) e `tool` (resultado de uma ferramenta).

**Saída estruturada** (9b)
: Obrigar o modelo a responder num formato fixo (um "formulário" com campos definidos), em vez de texto livre. Assim o código consegue usar a resposta sem ambiguidades.

**Token** (1)
: Um pedaço de palavra (em média, cerca de ¾ de uma palavra). Os modelos leem e escrevem em tokens, e o custo e os limites contam-se em tokens.

**Tool calling** (3)
: O mecanismo pelo qual o modelo pede uma ferramenta: responde com o nome da ferramenta e os argumentos (em JSON), em vez de texto.

## Parte 2 — Python

**`import`** (0)
: Traz para o nosso programa código escrito por outras pessoas (bibliotecas). Ex.: `import os`.

**Variável** (0)
: Um nome para guardar um valor. Ex.: `chave = "hf_..."`. Nomes em MAIÚSCULAS (`MODEL`, `MAX_PASSOS`) são, por convenção, valores que não mudam.

**Função (`def`)** (0)
: Um bloco de código com nome, que se pode chamar várias vezes, com `return` para devolver um resultado. Ex.: `def somar(a, b): return a + b`.

**`if` / `elif` / `else`** (0)
: Fazer uma coisa ou outra consoante uma condição.

**`try` / `except`** (0)
: "Tenta fazer isto; se der erro, faz aquilo em vez de o programa rebentar."

**f-string** (0)
: Texto com valores lá dentro: `f"Olá {nome}"`.

**Lista `[...]`** (1)
: Uma sequência ordenada de valores. `lista.append(x)` acrescenta `x` no fim.

**Dicionário `{...}`** (1)
: Pares chave → valor. Ex.: `{"role": "user", "content": "Olá"}`. Lê-se um valor com `d["role"]`.

**`os.environ` / `os.getenv`** (1)
: Ler configurações (as do `.env`). `os.environ["X"]` dá erro se `X` não existir; `os.getenv("X", "omissão")` usa um valor por omissão.

**`while True` / `break` / `continue`** (2)
: Repetir para sempre; sair do ciclo; saltar para a volta seguinte.

**`input()`** (2)
: Espera que o utilizador escreva uma linha no terminal.

**`for x in lista`** (3)
: Repetir o bloco para cada elemento da lista.

**Indicações de tipo** (3)
: `a: float`, `-> str`, `list[str]`, `str | None`. Indicam que tipo de valor se espera. Não mudam o funcionamento, mas ajudam a ler o código, e algumas bibliotecas (como o MCP) usam-nas para gerar descrições.

**JSON / `json.loads`** (3)
: JSON é um formato de texto para dados (parecido com os dicionários de Python). `json.loads(texto)` converte-o em dicionário; `json.dumps(d)` faz o contrário.

**`**args`** (3)
: Chamar uma função passando cada par de um dicionário como argumento: `f(**{"a": 1, "b": 2})` é o mesmo que `f(a=1, b=2)`.

**`if __name__ == "__main__":`** (4)
: O código que só corre quando executamos o ficheiro diretamente, e não quando outro ficheiro o importa.

**`sys.argv`** (4)
: O que se escreveu no terminal a seguir ao nome do programa.

**`with ... as f`** (5)
: Abre algo (um ficheiro, uma ligação) e garante que é fechado no fim, mesmo que haja um erro.

**`Path` (pathlib)** (5)
: Representa caminhos de ficheiros. `pasta / "ficheiro.txt"` junta caminhos.

**Função recursiva** (5)
: Uma função que se chama a si própria para resolver partes mais pequenas do problema.

**Decorador (`@...`)** (6)
: Uma linha por cima de uma função que lhe acrescenta comportamento. Ex.: `@mcp.tool()` regista a função como ferramenta MCP.

**Docstring** (6)
: O texto entre três aspas logo a seguir a um `def` (ou no início de um ficheiro), que descreve o que faz.

**`async` / `await`** (6)
: Código assíncrono: permite esperar por várias coisas ao mesmo tempo (o modelo, o servidor MCP, o utilizador). Lê `await x` como "espera por x e, entretanto, deixa outras coisas avançar". Uma função `async def` corre com `await` ou com `asyncio.run(...)`.

**`class`** (7)
: Um "molde" que junta dados e funções. Ex.: `CaixaLocal` sabe onde estão os emails e como os ler.

**`Literal[...]`** (7)
: Indicação de tipo "só pode ser um destes valores".

**`argparse`** (8)
: Lê as opções escritas no terminal (`-n 5`, `--modelo ...`).

**`asyncio.gather` / semáforo** (8)
: Lançar várias tarefas ao mesmo tempo e esperar por todas; o semáforo limita quantas correm em simultâneo.

**`yield` / gestor de contexto** (9a)
: Uma função que prepara algo, o entrega a quem a usa com `with` (ou `async with`) e, no fim, arruma.

**`BaseModel` / `Field`** (9b)
: Da biblioteca Pydantic: descrevem a forma exata dos dados (campos, tipos, descrições). O LLM usa-os como "formulário" na saída estruturada.

**`TypedDict`** (9b)
: Um dicionário com chaves fixas. É o "estado" que passa de passo em passo no grafo.

**`lambda`** (9b)
: Uma função pequena sem nome, escrita numa linha. Ex.: `lambda x: x * 2`.

**Expressões regulares (`re`)** (9b)
: Uma mini-linguagem para procurar padrões em texto. Ex.: `re.search(r"\d{4}-\d{2}-\d{2}", texto)` encontra uma data.

**Hash (`hashlib`)** (10)
: Uma "impressão digital" de um texto: textos iguais dão sempre o mesmo código, e não se consegue recuperar o texto a partir dele.

**`enumerate` / `zip`** (11a)
: Percorrer uma lista com o número de cada elemento; percorrer duas listas lado a lado.
