# `MagnusFormer`
# `MagnusFormer`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Ronnypetson Souza da Silva  | 211848  | Ciência da Computação|


## Descrição Resumida do Projeto
A Transformer-based autoregressive language model for chess PGN generation resembling human games.
> Descrição do tema do projeto, incluindo contexto gerador, motivação.

O tema do projeto é a geração de partidas de Xadrez (ref.), usando a notação PGN (ref.), que sejam semelhantes à partidas entre profissionais do jogo. Para tanto, deseja-se usar modelos de linguagem autorregressivos baseados em Transformer (ref.), tais como BERT (ref.). Partidas de Xadrez sintéticas semelhante às que humanos jogam podem ser usadas na ficção da literatura e do cinema (ref. Gambito da Rainha, Simpsons, Death Note, Harry Potter), com geração condicionada e muita flexibilidade. Por ser compatível com modelos de linguagem e por possuir muitas partidas públicas que a usam, a notação PGN é uma escolha adequada para a tarefa em questão.

Os modelos de linguagem autorregressivos baseados em Transformer têm mudado o estado-da-arte em várias tarefas de lingaugem natural, tais como (ref.). Também têm sido usados na geração de código em linguagens de programação (ref.), imagens vetorizadas tipo SVG (ref.) e arquivos no formato MIDI (ref.). A princípio, uma base de textos em  qualquer linguagem, natual ou não, com um vocabulário de tamanho razoável (menor ou comparável com o vocabulário de uma linguagem natural como o Inglês), pode ser modelada com esse tipo de técnica. Isso inclui os textos nos arquivos PGN de Xadrez, que possuem um vocabulário bem menor do que o do Inglês - cerca de 14700 palavras.

> Descrição do objetivo principal do projeto.

O objetivo principal do projeto é a criação de um modelo generativo que seja capaz de produzir sequências de lances de Xadrez no formato PGN, de modo que gere partidas de Xadrez semelhantes à partidas jogadas por seres humanos no nível profissional (partidas jogadas em torneios). O modelo deve ser capaz de gerar partidas do início como também de completar partidas já começadas.

> Esclarecer qual será a saída do modelo de síntese, ou generativo.

> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
