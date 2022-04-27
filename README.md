# `MagnusFormer`: treinando Transformers para gerar partidas de Xadrez
# `MagnusFormer`: training Transformers to generate Chess games

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Ronnypetson Souza da Silva  | 211848  | Ciência da Computação|


## Descrição Resumida do Projeto

### Tema do projeto, contexto e motivação

O tema do projeto é a geração de partidas de Xadrez, usando a notação PGN (*Portable Game Notation*) (ref.), que sejam semelhantes à partidas entre profissionais do jogo. Para tanto, deseja-se usar modelos de linguagem autorregressivos baseados em Transformer (ref.), tais como BERT (ref.). Partidas de Xadrez sintéticas semelhante às que humanos jogam podem ser usadas na ficção da literatura e do cinema (ref. Gambito da Rainha, Simpsons, Death Note, Harry Potter), com geração condicionada e flexibilidade. Por ser compatível com modelos de linguagem e por possuir muitas partidas públicas que a usam, a notação PGN é uma escolha adequada para a tarefa em questão.

Os modelos de linguagem autorregressivos baseados em Transformer têm mudado o estado-da-arte em várias tarefas de lingaugem natural, tais como (ref.). Também têm sido usados na geração de código em linguagens de programação (ref.), imagens vetorizadas tipo SVG (ref.) e arquivos no formato MIDI (ref.). A princípio, uma base de textos em  qualquer linguagem, natual ou não, com um vocabulário de tamanho razoável (menor ou comparável com o vocabulário de uma linguagem natural como o Inglês), pode ser modelada com esse tipo de técnica. Isso inclui os textos nos arquivos PGN de Xadrez, que possuem um vocabulário - cerca de 14700 *tokens* - bem menor do que o do Inglês.

### Objetivo

O objetivo principal do projeto é a criação de um modelo generativo que seja capaz de produzir sequências de lances de Xadrez no formato PGN, de modo que gere partidas de Xadrez semelhantes à partidas jogadas por seres humanos no nível profissional (partidas jogadas em torneios). O modelo deve ser capaz de gerar partidas do início como também de completar partidas já começadas.

### Saída do modelo

A saída do modelo será uma sequência de anotações de lances de Xadrez numa versão simplificada do formato PGN. Nesse formato, cada lance especifica uma peça no tabuleiro e a nova casa que deve ocupar. As peças, exceto peões, são representados por letras da seguinte forma:

* Peão: ♙♟
* Cavalo: N (kNight) ♘♞
* Bispo: B (Bishop) ♗♝
* Torre: R (Rook) ♖♜
* Dama: Q (Queen) ♕♛
* Rei: K (King) ♔♚

As linhas e colunas do tabuleiro são representadas pelos números de ```1``` à ```8``` e pelas letras de ```a``` à ```h```, como mostra a figura abaixo. Dessa maneira, quando queremos dizer "mover o cavalo para a casa f3", escrevemos ```Nf3``` na notação PGN. Quando a peça a ser movida é ambígua, pode-se especificar a coluna e/ou a linha da casa que a peça ocupa caso seja necessário, como por exemplo ```Ngf3```, ```N5f3``` ou ```Ne5f3```. Esses lances também representam o movimento de um cavalo para a casa ```f3```.

<p align="center">
  <img src='https://upload.wikimedia.org/wikipedia/commons/b/b6/SCD_algebraic_notation.svg' width='300'>
</p>

Um arquivo PGN completo tem o seguinte formato

```
[Event "F/S Return Match"]
[Site "Belgrade, Serbia JUG"]
[Date "1992.11.04"]
[Round "29"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 {This opening is called the Ruy Lopez.}
4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7
11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5
Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6
23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5
hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5
35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6
Nf2 42. g4 Bd3 43. Re6 1/2-1/2
```

Entretanto, será usada apenas a parte que especifica os movimentos (*movetext*), sem a enumeração das rodadas e sem comentários, resultando numa notação similar à seguinte:

```
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7
c4 c6 cxb5 axb5 Nc3 Bb7 Bg5 b4 Nb1 h6 Bh4 c5 dxe5 Nxe4 Bxe7 Qxe7 exd6 Qf6 Nbd2 Nxd6
Nc4 Nxc4 Bxc4 Nb6 Ne5 Rae8 Bxf7+ Rxf7 Nxf7 Rxe1+ Qxe1 Kxf7 Qe3 Qg5 Qxg5 hxg5
b3 Ke6 a3 Kd6 axb4 cxb4 Ra5 Nd5 f3 Bc8 Kf2 Bf5 Ra7 g6 Ra6+ Kc5 Ke1 Nf4 g3 Nxh3 
Kd2 Kb5 Rd6 Kc5 Ra6 Nf2 g4 Bd3 Re6 1/2-1/2
```

A partida pode ser recuperada usando um leitor de PGN:

<p align="center">
  <img src='/src/visualization/board.gif' width='300'>
</p>

### Vídeo de apresentação

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:

> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.

### Base de dados
A base de dados a ser usada é um conjunto de várias partidas em torneios, disponível no site [The Week in Chess](https://theweekinchess.com/zips). Ao todo são mais de 1400 arqivos *zip*, cada um contendo partidas de torneios profissionais ao redor do mundo em cada semana. Todas as partidas estão no formato PGN.

> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.

### Abordagens de modelagem generativa
A ideia inicial é usar as anotações de lances (ex: ```Nf3```, ```e4```, ```Be7```) como palavras individuais em um texto para servir de tokens. Daí um modelo autorregressivo tipo BERT seria treinado a partir dessa tokenização. Uma vez que o modelo é treinado, a geração de novas partidas ou complementos de partidas iniciadas pode ocorrer com a geração de um token por vez ou através de *beam search*, onde os próximos **k** lances são escolhidos de modo a maximizar a probabilidade de acordo com o modelo. A desvantagem do beam search está no custo computacional elevado, crescendo exponencialmente com o tamanho de **k**.

> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
