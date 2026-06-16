<h1 align="center"><font color="gree">Executando LLMs locais com llama serve</font></h1>


<font color="pink">Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro</font>


Este documento descreve como executar um LLM local usando `llama.cpp` por meio do comando `llama serve`, abrir a interface web local e testar o modelo `unsloth/Qwen3.6-35B-A3B-GGUF`.

O fluxo documentado aqui foi testado com o modelo quantizado `UD-Q4_K_M` e com a UI web exposta em `http://127.0.0.1:8080`.


## <font color="red">O que e o `llama.cpp`</font>

`llama.cpp` e um motor de inferencia para `LLMs` escrito em `C/C++`. Ele permite executar modelos localmente em CPU, GPU ou uma combinacao de `CPU + GPU`, usando formatos como `GPT-Generated Unified Format` (GGUF). Este é um formato binário para armazenar modelos de linguagem grandes e seus pesos, otimizado para inferência eficiente en CPU e GPU de consumo.

O site oficial [llama.app](https://llama.app/) fornece um instalador simples para Linux, macOS e outros ambientes. No teste descrito neste documento, o instalador disponibilizou o comando `llama`, e o servidor foi iniciado com:

```bash
llama serve
```

Neste documento, todos os comandos usam `llama serve`, que foi o comando instalado e testado localmente via `llama.app`.

## <font color="red">Modelo escolhido</font>

Modelo:

```text
unsloth/Qwen3.6-35B-A3B-GGUF
```

Quantização executada:

```text
UD-Q4_K_M
```

Comando usado:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192
```

## <font color="red">Por que usar este modelo?</font>

O `Qwen3.6-35B-A3B` e um modelo `Mixture-of-Experts` (MoE). Segundo a descricao publica do ecossistema `llama.app`, ele e um modelo de 35B parametros totais com cerca de 3B parametros ativos por token. Isso e relevante porque modelos MoE podem oferecer qualidade proxima a modelos maiores, mas com custo de inferencia menor por token do que um modelo denso de tamanho equivalente.

Em termos práticos, este modelo serve para conversar, escrever, explicar conceitos, auxiliar em tarefas de programação, analisar trechos de código, raciocinar sobre problemas em varias etapas e testar inferencia local de um modelo grande sem depender de uma API externa. Ele e especialmente interessante para estudos de LLMs locais porque combina tres caracteristicas importantes: `arquitetura MoE`, `foco em reasoning/coding` e `disponibilidade em GGUF quantizado`.

O modelo também é interessante para testes locais porque:

- Tem foco em coding e reasoning, ou seja, tende a ser mais útil em tarefas que exigem explicação, análise, planejamento e geração de código.
- Está disponível em formato `GGUF` pela `Unsloth`.
- Possui várias quantizações, permitindo escolher entre menor uso de memória e maior qualidade.
- Pode ser servido localmente via `llama.cpp`, com UI web e API local.

Sobre janela de contexto: a model card oficial do `Qwen` informa que o `Qwen3.6-35B-A3B` tem contexto padrao/nativo de `262.144 tokens`. O log do `llama serve` tambem reportou `n_ctx_train = 262144`. No teste deste documento, usamos `--ctx-size 8192` de proposito. Isso nao usa toda a capacidade do modelo, mas reduz consumo de memoria e torna o primeiro teste mais simples e estavel.

Resumo das vantagens para este experimento:

- Modelo grande, mas com inferência mais eficiente por ser `MoE`.
- Bom candidato para testar raciocínio, coding e respostas mais elaboradas.
- Pode rodar localmente com `llama.cpp`, mantendo os dados na máquina.
- Tem quantizações menores para máquinas com menos memória.
- Permite testar UI web local e API local no mesmo servidor.

Limites importantes:

- Mesmo quantizado, ainda é um modelo pesado.
- A `velocidade` depende fortemente de `GPU`, `CPU`, `RAM`, `drivers` e `parâmetros de contexto`.
- Usar `--ctx-size 8192` e uma escolha conservadora; contextos maiores podem exigir mais memória. A documentação oficial do `Qwen` recomenda contextos maiores para preservar melhor as capacidades de `thinking` em tarefas complexas, mas isso aumenta o custo de memória.

## <font color="red">Sobre a quantização `UD-Q4_K_M`</font>

A página do Hugging Face lista várias versões `GGUF` do modelo, incluindo:

- `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`: aproximadamente 12.3 GB.
- `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`: aproximadamente 22.1 GB.
- `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`: aproximadamente 22.4 GB.
- `Qwen3.6-35B-A3B-Q8_0.gguf`: aproximadamente 36.9 GB.

Neste teste foi usada a quantização `UD-Q4_K_M`, que e uma opção de 4 bits. Em geral, quantizações de `4 bits` tentam manter uma boa relação entre qualidade e consumo de memória. Quantizações menores, como `UD-Q2_K_XL`, tendem a consumir menos memória e disco, mas podem reduzir a qualidade das respostas.

Para um primeiro teste em uma máquina com menos memória, uma opção mais conservadora seria usar o mesmo modelo com a quantização `UD-Q2_K_XL`:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL \
  --ctx-size 8192
```

Essa alternativa não troca a família do modelo. Ela troca a variante quantizada usada pelo `llama.cpp`. Em outras palavras, continuamos executando `Qwen3.6-35B-A3B-GGUF`, mas com um arquivo menor e mais agressivamente quantizado.

## <font color="red">Requisitos práticos</font>

Para reproduzir o teste com a quantização `UD-Q4_K_M`, considere:

- Linux com acesso ao terminal.
- Espaço livre em disco suficiente para baixar o arquivo `GGUF`. A versão `UD-Q4_K_M` tem cerca de `22 GB`.
- RAM suficiente para carregar parte do modelo, cache e overhead do servidor.
- GPU NVIDIA e CUDA ajudam bastante, mas `llama.cpp` também pode executar em CPU ou fazer offload parcial.
- Conexão com a internet para baixar o instalador e o modelo do Hugging Face.

No teste local documentado aqui, o `llama serve` detectou uma GPU NVIDIA e CUDA:

```text
CUDA0: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU (11774 MiB, 11522 MiB free)
CPU: Intel(R) Core(TM) Ultra 9 285HX (63748 MiB, 63748 MiB free)
```

Ou seja, a maquina tinha aproximadamente:

- 11.7 GiB de VRAM.
- 63.7 GiB de RAM disponivel no momento da inicializacao.
- CPU com 24 threads logicas reportadas pelo servidor.

## <font color="red">Instalação do `llama.cpp`</font>

O caminho mais simples é usar o instalador oficial do `llama.app`:

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Durante o teste, o instalador executou uma etapa de detecção de CUDA e concluiu com sucesso:

```text
Version: b9585
Probing CUDA...
Downloading cuda-probe...
Found: 120
Downloading llama...
Installation completed successfully

Run the following command to start it:

  llama serve
```

Depois da instalação, verifique se o comando está disponível:

```bash
command -v llama
```

Se esse comando não retornar um caminho para o executável `llama`, feche e abra novamente o terminal, ou verifique se o instalador adicionou o binário ao `PATH` da sua shell.

## <font color="red">Executando o servidor com o modelo `Qwen`</font>

Execute:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192
```

O ambiente virtual Python do projeto não é obrigatório para executar esse comando. Ele foi usado no terminal durante o teste porque já estava ativo, mas o `llama serve` instalado pelo `llama.app` funciona como ferramenta de sistema/usuario, fora do projeto Python.

Explicação dos argumentos:

- `llama serve`: inicia o servidor local do `llama.cpp`.
- `-hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M`: baixa e carrega o modelo diretamente do Hugging Face, usando a variante `UD-Q4_K_M`.
- `--ctx-size 8192`: limita o contexto inicial a 8192 tokens. Isso reduz consumo de memória em relação ao contexto máximo do modelo.

Ao executar pela primeira vez, o modelo será baixado para o cache local do Hugging Face. No teste, o arquivo carregado foi:

```text
<path_on_your_notebook_where_the_model_downloaded>/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

O servidor também carregou o arquivo multimodal:

```text
mmproj-BF16.gguf
```

Esse arquivo é usado para capacidades multimodais, como processamento de imagem, quando suportadas pelo modelo e pela UI.

## <font color="red">Abrindo a UI web</font>

Quando o modelo carregar corretamente, o terminal deve mostrar algo parecido com:

```text
llama_server: model loaded
llama_server: server is listening on http://127.0.0.1:8080
```

Abra no navegador:

```text
http://127.0.0.1:8080
```

A partir dessa UI, é possível conversar com o modelo localmente.

## <font color="red">Observações importantes no meus logs</font>

### <font color="blue">Modelo carregado com sucesso</font>

O log confirmou:

```text
llama_server: loading model
llama_server: model loaded
llama_server: server is listening on http://127.0.0.1:8080
```

Isso indica que o modelo foi carregado e a UI/API local ficou disponível.

### <font color="blue">Contexto usado no teste</font>

O modelo possui contexto de treinamento maior, mas o teste limitou a execução para 8192 tokens:

```text
n_ctx_seq (8192) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
```

Isso não é erro. Significa apenas que o teste usou uma janela de contexto menor que a capacidade máxima do modelo. Essa escolha reduz o consumo de memória e é adequada para validar o setup inicial.

### <font color="blue">Uso multimodal</font>

O servidor carregou o `mmproj-BF16.gguf`:

```text
loaded multimodal model, '.../mmproj-BF16.gguf'
```

Também apareceu o aviso:

```text
Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

Para uso apenas texto, esse aviso normalmente não impede o teste. Para tarefas com imagem, pode ser útil adicionar:

```bash
--image-min-tokens 1024
```

### <font color="blue">Thinking mode</font>

A documentação oficial do `Qwen` informa que os modelos `Qwen3.6` operam em `thinking mode` por padrão. Isso significa que o modelo pode gerar conteúdo de raciocínio antes da resposta final, o que pode melhorar respostas complexas, mas também pode aumentar a quantidade de tokens gerados e o tempo de resposta.

No teste local, o `llama serve` mostrou:

```text
chat template, thinking = 1
```

Para obter respostas mais diretas, é possível desabilitar esse comportamento via `chat_template_kwargs`:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --alias qwen3.6-q4 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Também é possível enviar essa preferência por chamada na API, usando `chat_template_kwargs` no corpo JSON, quando o cliente a o endpoint suportarem esse campo.

### <font color="blue">Prompt cache</font>

O servidor ativou cache de prompt:

```text
prompt cache is enabled, size limit: 8192 MiB
use `--cache-ram 0` to disable the prompt cache
```

Esse cache pode melhorar interações repetidas, mas também consome RAM. Se for necessário reduzir consumo de memória, teste:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --cache-ram 0
```

### <font color="blue">Desempenho observado</font>

Nos logs, uma das respostas apresentou:

```text
prompt eval time = 1866.74 ms / 129 tokens (69.10 tokens per second)
eval time = 34333.39 ms / 436 tokens (12.70 tokens per second)
total time = 36200.13 ms / 565 tokens
```

Outras respostas ficaram próximas de:

```text
eval time = 6113.80 ms / 80 tokens (13.09 tokens per second)
eval time = 3533.97 ms / 46 tokens (13.02 tokens per second)
eval time = 3965.28 ms / 52 tokens (13.11 tokens per second)
```

Portanto, no hardware testado, com `UD-Q4_K_M` e `--ctx-size 8192`, a geração ficou em torno de 13 tokens por segundo. Esse número é uma observação local, não uma garantia geral. O desempenho pode mudar conforme GPU, CPU, RAM, drivers, tamanho do prompt, tamanho do contexto, temperatura, paralelismo e outros parâmetros.

## <font color="red">Comando recomendado para repetir o teste</font>

Para repetir o teste documentado:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192
```

Depois abra:

```text
http://127.0.0.1:8080
```

## <font color="red">Comando alternativo mais leve</font>

Se faltar memória, se o download for muito grande ou se o objetivo for apenas validar a UI, use a quantização `UD-Q2_K_XL`:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL \
  --ctx-size 8192
```

Esse comando executa o mesmo modelo base, `Qwen3.6-35B-A3B-GGUF`, mas troca a quantização de `UD-Q4_K_M` para `UD-Q2_K_XL`. De acordo com a listagem do Hugging Face, o arquivo `UD-Q2_K_XL` tem aproximadamente 12.3 GB, enquanto o `UD-Q4_K_M` tem aproximadamente 22.1 GB.

Use `UD-Q2_K_XL` quando:

- Você quer apenas confirmar que o `llama serve` e a UI web estão funcionando.
- A máquina tem menos `RAM`, menos `VRAM` ou menos espaço em disco.
- O download de 22 GB da versão `Q4` e pesado demais para o primeiro teste.
- Você aceita uma possível perda de qualidade em troca de menor consumo.

`Trade-off` principal:

- `UD-Q2_K_XL`: menor arquivo, menor consumo, mais fácil de testar, qualidade potencialmente inferior.
- `UD-Q4_K_M`: <font color="cyan">arquivo maior, maior consumo, melhor equilíbrio esperado entre qualidade e tamanho.</font>

Para documentação e reprodução, o teste principal deste repositório usa `UD-Q4_K_M`. A variante `UD-Q2_K_XL` fica como exemplo recomendado para validar o ambiente em máquinas mais limitadas.

## <font color="red">Comando com ajustes úteis</font>

Para reduzir uso de RAM com prompt cache desativado:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --cache-ram 0
```

Para testar tarefas multimodais com imagens:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --image-min-tokens 1024
```

## <font color="red">API local</font>

O servidor local também expõe uma API HTTP. Para uso interativo, a forma mais simples é abrir a UI web:

```text
http://127.0.0.1:8080
```

Para integrar com ferramentas que esperam uma API no estilo `OpenAI`, use a base:

```text
http://127.0.0.1:8080/v1
```

Antes de chamar `/v1/chat/completions`, verifique qual nome de modelo o servidor está expondo:

```bash
curl http://127.0.0.1:8080/v1/models
```

Use no campo `"model"` exatamente o valor retornado em `data[].id` ou um alias registrado pelo servidor. Se o valor não bater, o servidor pode responder com erro parecido com:

```json
{
  "error": {
    "code": 400,
    "message": "model 'unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M' not found",
    "type": "invalid_request_error"
  }
}
```

Uma forma simples de evitar ambiguidade e iniciar o servidor com um alias explícito:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --alias qwen3.6-q4
```

Depois use esse alias no `curl`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-q4",
    "max_tokens": 80,
    "messages": [
      {
        "role": "user",
        "content": "Explique em uma frase, com no maximo 30 palavras, o que e o llama.cpp."
      }
    ]
  }'
```

O campo `max_tokens` limita a quantidade máxima de tokens gerados na resposta. Isso é útil em testes iniciais porque respostas longas podem demorar, especialmente em modelos grandes executando parcialmente em CPU/RAM.

## <font color="red">Troubleshooting</font>

### <font color="blue">Porta 8080 ocupada</font>

Se a porta `8080` já estiver em uso, especifique outra porta:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --port 8081
```

Depois abra:

```text
http://127.0.0.1:8081
```

### <font color="blue">`model not found` na API `/v1/chat/completions`</font>

Esse erro significa que o valor enviado no campo `"model"` não corresponde ao nome/alias exposto pelo servidor. Verifique com:

```bash
curl http://127.0.0.1:8080/v1/models
```

Depois use exatamente o `id` retornado ou reinicie o servidor com um alias explícito:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --ctx-size 8192 \
  --alias qwen3.6-q4
```

Nesse caso, o JSON da chamada deve usar:

```json
{
  "model": "qwen3.6-q4"
}
```

### <font color="blue">Falta de memória</font>

Tente, nesta ordem:

1. Reduzir a quantização para `UD-Q2_K_XL`.
2. Reduzir `--ctx-size`.
3. Desativar prompt cache com `--cache-ram 0`.
4. Fechar aplicações que usam VRAM/RAM.
5. Reiniciar o servidor.

Exemplo:

```bash
llama serve \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL \
  --ctx-size 4096 \
  --cache-ram 0
```

### <font color="blue">O modelo baixou, mas quero saber onde ficou</font>

O `llama serve -hf ...` usa cache local do `Hugging Face`. No teste, o modelo ficou em:

```text
~/.cache/huggingface/hub/
```

Isso evita baixar o mesmo arquivo novamente em execuções futuras.

## <font color="red">Conclusão</font>

O caminho mais simples para executar esse `LLM` localmente foi:

1. Instalar o `llama.cpp` via `llama.app`.
2. Rodar `llama serve` apontando diretamente para o modelo no `Hugging Face`.
3. Limitar o contexto inicial com `--ctx-size 8192`.
4. Abrir a UI em `http://127.0.0.1:8080`.

No hardware testado, o modelo `Qwen3.6-35B-A3B-GGUF` com quantização `UD-Q4_K_M` carregou com sucesso e gerou respostas em torno de 13 tokens por segundo. O resultado mostra que é possível testar um modelo `MoE` grande localmente, desde que haja memória suficiente e que se aceite o `trade-off` entre tamanho do modelo, quantização, contexto e velocidade.


Links de estudo:

* [llama.app](https://llama.app/)

* [GitHub: llama.cpp](https://github.com/ggml-org/llama.cpp)

* [Hugging Face: Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)

* [Unsloth AI](https://huggingface.co/unsloth)



Thank God!