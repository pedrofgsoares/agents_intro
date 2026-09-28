# Criar agentes de IA, passo a passo

Material de uma formação em 4 sessões. Começamos com uma simples pergunta a um modelo de linguagem (LLM) e acabamos com vários agentes a conversar entre si. Pelo meio construímos um agente para um caso real: analisar pedidos de justificação de faltas.

Cada passo é um ficheiro Python curto e numerado (`passo01_...py`, `passo02_...py`, …), que acrescenta **uma** ideia ao anterior. Não é preciso saber programar bem: cada ficheiro começa com um cabeçalho que explica o que vais aprender, como o correr, o que observar e o que experimentar mudar.

> 💡 Palavras que não conheças (token, MCP, RAG, `async`…) estão explicadas no [GLOSSARIO.md](GLOSSARIO.md).

---

## 1. Preparar o computador (uma vez só)

Precisas de **Ubuntu** ou de **WSL** (o Ubuntu dentro do Windows). Todos os comandos abaixo são escritos num terminal.

**1.1 — Instalar o `git` e o `uv`**

O `git` serve para descarregar (e depois atualizar) este material. O `uv` instala o Python e as bibliotecas de que precisamos, sem mexer no resto do sistema.

```bash
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Fecha e volta a abrir o terminal, para o `uv` ficar disponível.

**1.2 — Descarregar o material**

```bash
git clone <ENDEREÇO-DO-REPOSITÓRIO> agentes
cd agentes
uv sync
```

(O formador indica o endereço.) O `uv sync` descarrega tudo o que é preciso e demora um ou dois minutos da primeira vez.

**1.3 — Criar a tua chave da Hugging Face**

Os modelos que usamos correm nos servidores da [Hugging Face](https://huggingface.co). Para os usar precisas de uma conta e de uma chave pessoal:

1. Cria uma conta em <https://huggingface.co/join> (é gratuita).
2. Vai a <https://huggingface.co/settings/tokens> → **Create new token**.
3. Escolhe o tipo **Fine-grained**, dá-lhe um nome (ex.: `formacao`) e ativa **Make calls to Inference Providers**.
4. Cria a chave e copia-a. Começa por `hf_`.

**1.4 — Configurar o `.env`**

O ficheiro `.env` guarda as tuas configurações e a tua chave. Cria-o a partir do exemplo:

```bash
cp .env.exemplo .env
```

Abre o `.env` num editor (por exemplo `code .env` ou `nano .env`) e cola a tua chave na linha `HF_API_KEY=`. As outras linhas já têm valores que servem.

> ⚠️ O `.env` tem a **tua** chave: não o partilhes. O git já está configurado para nunca o enviar.

**1.5 — Verificar**

```bash
uv run passo00_verificar.py
```

Se aparecer **🎉 Está tudo pronto para a formação!**, podes avançar. Se não, o script diz o que corrigir.

---

## 2. O percurso

Corre cada passo, lê o cabeçalho do ficheiro e experimenta as sugestões da secção **EXPERIMENTA MUDAR**.

### Sessão 1 — O que é um agente?

| Passo | O que aprendes | Como correr |
|---|---|---|
| 00 | Verificar que está tudo pronto | `uv run passo00_verificar.py` |
| 01 | Fazer uma pergunta a um LLM; o papel do prompt de sistema | `uv run passo01_chamada_simples.py` |
| 02 | A "memória" de um chat é só a lista de mensagens | `uv run passo02_chat_com_memoria.py` |
| 03 | Ferramentas: o modelo **pede**, o nosso programa **executa** | `uv run passo03_tool_calling.py` |
| 04 | **O ciclo do agente**: pedir → executar → repetir até ter resposta | `uv run passo04_ciclo_do_agente.py` |
| 05 | Um agente de conversa com ferramentas seguras e autorização humana | `uv run passo05_agente_conversacional.py` |

### Sessão 2 — Ferramentas externas e um caso real

| Passo | O que aprendes | Como correr |
|---|---|---|
| 06 | MCP: as ferramentas passam para um servidor à parte (`mcp_servidor.py`) | `uv run passo06_agente_mcp.py` |
| 07 | O caso prático: um agente que analisa pedidos de justificação de faltas | `uv run passo07_agente_email.py` |
| 08 | Avaliar um agente: medir quantas vezes acerta, em vez de confiar | `uv run passo08_avaliacao.py` |

> O caso prático usa só **dados inventados**: 10 emails em `dados/caixa_ficticia/` e um regulamento em `dados/regulamento_faltas.md`. As respostas certas estão em `dados/gabarito.json`.
>
> ⚠️ O passo 08 gasta bastantes tokens (corre o agente 30 vezes). Para experimentar com menos: `uv run passo08_avaliacao.py -n 1 --emails 004 006`.

### Sessão 3 — Frameworks, memória e conhecimento

Estes passos são mais longos. Não é preciso perceber todas as linhas: o cabeçalho de cada um diz o que observar, e é isso que importa. Quando o cabeçalho diz *"Podes só correr e observar"*, é mesmo isso.

| Passo | O que aprendes | Como correr |
|---|---|---|
| 09a | O agente do passo 7 reescrito com uma framework (LangChain) | `uv run passo09a_langgraph_agente.py` |
| 09b | Um **grafo**: o código decide o caminho, o LLM só extrai factos. Acerta mais e gasta menos | `uv run passo09b_langgraph_grafo.py` |
| 10 | Memória de longo prazo: lembrar pedidos anteriores de cada estudante | `uv run passo10_memoria.py` |
| 11a | RAG: pesquisar nos regulamentos reais (em `dados/regulamentos/`) | `uv run passo11a_indice_rag.py "faltei por doença"` |
| 11b | O grafo com os regulamentos reais e citações verificadas | `uv run passo11b_grafo_rag.py` |

Para comparar versões do agente, usa o passo 08: `uv run passo08_avaliacao.py --agente passo09b` (ou `passo09a`, `passo10`, `passo11b`).

#### Regulamentos (antes dos passos 11a e 11b)

Os passos 11a e 11b pesquisam em regulamentos reais, em PDF, que **não vêm com o material**. Descarrega-os dos endereços que o formador indicar e grava-os na pasta `dados/regulamentos/` **com estes nomes exatos** (o programa reconhece cada documento pelo nome):

| Documento | Nome do ficheiro |
|---|---|
| Normas FEUP — Regime de Assiduidade e de Faltas (2025) | `Normas_Regime_de_Assiduidad_e_Faltas_-_FEUP.pdf` |
| Regulamento do DEI-FEUP (2025) | `DEI - Regulamento 2025.pdf` |
| Regulamento de exames da FPCEUP | `Regulamento_de_exames_VF.pdf` |

Só as Normas FEUP se aplicam às faltas. Os outros dois estão lá de propósito, como "distratores": a pesquisa tem de encontrar a regra certa no meio de documentos parecidos.

Para confirmar que estão todos:

```bash
ls dados/regulamentos/
```

As Normas FEUP são indispensáveis: se faltarem, o passo 11a avisa. Os distratores são opcionais, mas sem eles a pesquisa fica fácil demais.

### Sessão 4 — Vários agentes

| Passo | O que aprendes | Como correr |
|---|---|---|
| 12 | Vários agentes, cada um com a sua personalidade, conversam num quadro partilhado | ver abaixo |

Cada agente é um programa à parte. Abre um terminal e lança dois ou três (o `&` no fim põe-no a correr em segundo plano):

```bash
uv run passo12_blackboard.py --personalidade colaborativo &
uv run passo12_blackboard.py --personalidade cetico &
uv run passo12_blackboard.py --personalidade "um poeta que só fala em rimas" --nome Camões &
```

Noutro terminal, acompanha a conversa e intervém:

```bash
tail -f experiencia/quadro.md                                   # ver a conversa em direto
uv run passo12_blackboard.py --escrever "Mudança de planos!" --como Deus
uv run passo12_blackboard.py --estado                           # quem está ativo e quanto gastou
uv run passo12_blackboard.py --parar-todos                      # parar todos
uv run passo12_blackboard.py --limpar                           # apagar e recomeçar
```

Nesta sessão, o formador conduz a experiência principal no ecrã.

---

## 3. Experimentar sem medo

A melhor forma de aprender é mudar o código e ver o que acontece. Para não estragares os originais, **faz uma cópia com um nome começado por `meu_`**:

```bash
cp passo03_tool_calling.py meu_passo03.py
uv run meu_passo03.py
```

Os ficheiros `meu_*.py` são teus: o git ignora-os e nunca entram em conflito com as atualizações.

Se estragares um passo original, recupera-o com:

```bash
git checkout -- passo03_tool_calling.py
```

## 4. Atualizar o material

Se o formador publicar correções entre sessões:

```bash
git pull
uv sync
```

Se o `git pull` se queixar de alterações locais, é porque mudaste um ficheiro original. Guarda as tuas alterações numa cópia `meu_*.py` e depois corre `git checkout -- <ficheiro>` antes de repetires o `git pull`.

---

## 5. O que está em cada pasta

```
passo00 … passo12      os passos da formação
mcp_servidor.py        servidor MCP do passo 6 (ferramentas gerais)
mcp_email.py           servidor MCP do caso prático (passos 7 a 11)
dados/                 emails fictícios, regulamento fictício e respostas certas
dados/regulamentos/    onde gravas os PDFs dos regulamentos reais (passo 11)
.env.exemplo           modelo do teu .env
GLOSSARIO.md           o significado das palavras técnicas
```

Ao correr os passos são criados ficheiros e pastas. Podes abri-los para perceber o que aconteceu, e apagá-los à vontade:

| Criado por | O quê |
|---|---|
| `passoNN.log` | tudo o que foi enviado ao modelo e recebido dele (passos 5, 6, 7, 9a, 9b, 10, 11) |
| `notas/` | notas que autorizaste o agente do passo 5 a escrever |
| `saida/` | propostas e rascunhos de resposta do caso prático |
| `avaliacao/` | resultados detalhados do passo 8 |
| `memoria/` | a memória de longo prazo (passos 10 e 11b) |
| `dados/regulamentos/indice.json` | o índice de pesquisa do passo 11a |
| `experiencia/` | o quadro, o registo e os logs dos agentes do passo 12 |

## 6. Problemas comuns

| Sintoma | O que fazer |
|---|---|
| `uv: command not found` | Fecha e volta a abrir o terminal depois de instalar o `uv` |
| `FileNotFoundError: Faltam regulamentos` | Descarrega os PDFs (ver "Regulamentos" na sessão 3) |
| `KeyError: 'HF_API_KEY'` | Falta o `.env`, ou estás noutra pasta: faz `cd` para a pasta do material |
| Um erro vermelho a meio de um passo | Corre `uv run passo00_verificar.py`: diz se o problema é a chave, o modelo ou a ligação |
| `Tool call validation failed` ou `<\|channel\|>` | O modelo gerou uma chamada de ferramenta mal formada. Acontece com o `gpt-oss-20b`: volta a correr |
| O agente dá respostas diferentes de cada vez | É normal: um LLM não é determinístico. O passo 08 mostra como lidar com isso |
