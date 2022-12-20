# MUSS-ptBR - Simplificador textual para português

Autor: Raphael Assis (contato.raphael.assis@gmail.com)

## Introdução do problema

Para entender mais sobre a tarefa de simplificação textual leia [este artigo](https://direct.mit.edu/coli/article/46/1/135/93384/Data-Driven-Sentence-Simplification-Survey-and).

## Infraestrutura utilizada

Para a realização deste trabalho utilizou-se a plataforma [Google Cloud](https://cloud.google.com/). Esta plataforma disponibiliza todos os recursos necessários para a implementação deste trabalho e ainda oferece 300 dólares  de créditos (~1770 reais  em 08/2022) para testar os serviços antes de começar a pagar pela utilização. Por conta disso, este trabalho pode ser replicado integralmente somente se utilizando dos créditos gratuitos oferecidos pelo Google 🤩!

A infraestrutura utilizada foi a seguinte:

Máquina com 8 vCPUs, 52 GB de memória (n1-highmem-8), 2 TB de HDD (disco de inicialização) e 1 GPU NVIDIA Tesla T4. O sistema operacional utilizado foi o Ubuntu 20.04 LTS para arquitetura x86 de 64 bits. Essa configuração resulta em um custo de US$ 0,69 por hora.

Obs: O disco de inicialização não precisa possuir tanto volume de armazenamento. É possível economizar ainda mais utilizando um disco separado para manter os dados da VM e utilizar um disco de inicialização de uns 10Gb. Você pode ver mais detalhes sobre isso [neste tutorial](https://cloud.google.com/compute/docs/disks/add-persistent-disk?hl=pt-br). Entretanto, ao utilizar um disco de inicialização com bastante volume de armazenamento há menos configurações para realizar na VM. 

## Configuração do projeto na VM

Após criar e iniciar a VM é preciso clonar o projeto do Github e configurar as dependências do projeto. Além disso, como a VM inicia com uma imagem limpa do Linux é necessário atualizar alguns programas. Os passos necessários são os seguintes:

1. Execute `sudo apt-get update`
2. Execute `sudo apt-get install python3-pip`
3. Execute `sudo apt-get install zip`
4. Execute `sudo apt install unzip`
5. Execute `sudo apt install python3.8-venv`
6. Execute `sudo apt-get install build-essential cmake`
7. Execute `sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev`
8. Clone o código do Github: `git clone git@github.com:facebookresearch/muss.git`
9. Navegue até a pasta do projeto: `cd muss/`
10. Execute `pip install -e .`
11. Siga [este tutorial](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=pt-br#verify-driver-install) para instalar os drivers da GPU na VM.
12. Siga [este tutorial](https://cloud.google.com/compute/docs/gpus/monitor-gpus#use-virtualenv_1) para configurar a telemetria da GPU na VM e poder monitorar seu desempenho durante a execução dos treinamentos do modelo.
13. Suba os arquivos com o corpus de texto para a VM. Veja [este tutorial](https://cloud.google.com/compute/docs/instances/transfer-files?hl=pt-br#upload-to-the-vm) de como enviar e receber arquivos para a VM. A pasta onde os arquivos serão salvos não importa (por padrão é uma pasta com o seu nome de usuário em \home), pois o muss recebe o path como parâmetro.

Após realizar todos os passos acima, a VM estará configurada e pronta para o uso. 

## Adaptando o muss para um novo idioma

O Multilingual Unsupervised Sentence Simplification (MUSS) é um modelo de linguagem baseado em BART e mBART que realiza simplificação textual. Neste projeto, há tanto scripts para produzir uma base de dados de paráfrases para o treinamento do modelo quanto scripts para treinar e avaliar o modelo.

### Fase de mineração de paráfrases

Nesta fase, realiza-se o pré-processamento dos textos coletados e produção de paráfrases para realizar o treinamento do modelo. O objetivo desta fase é obter pares de frases com representem a versão complexa e simplificada de uma sentença. O resultado dessa fase é uma pasta com os arquivos  test.complex, test.simple, train.complex, train.simple, valid.complex e valid.simple. Ambos arquivos são no formato txt, sendo cada linha composta por uma uma sentença. Dessa forma, a sentença da linha 1 do arquivo test.complex é a versão complexa da sentença da linha 1 do arquivo test.simple.

Exemplo de arquivo com sentenças complexas: 

```
Um lado dos conflitos armados é composto principalmente pelos militares sudaneses e pelos Janjaweed, um grupo de milícias sudanesas recrutado principalmente das tribos afro-árabes Abbala da região norte de Rizeigat, no Sudão.
Jeddah é a principal porta de entrada para Meca, a cidade mais sagrada do Islã, que os muçulmanos sãos são obrigados a visitar pelo menos uma vez na vida.
Acredita-se que a Grande Mancha Escura represente um buraco no convés de nuvens de metano de Netuno.
Seu próximo trabalho, sábado, segue um dia especialmente agitado na vida de um neurocirurgião de sucesso.
A tarântula, o personagem trapaceiro, girou uma corda preta e, prendendo-a à bola, rastejou rapidamente para o leste, puxando a corda com toda a força.
Lá ele morreu seis semanas depois, em 13 de janeiro de 888.
Eles são culturalmente semelhantes aos povos costeiros de Papua Nova Guiné.
```

Exemplo de arquivo com sentenças simples: 

```
Um lado da guerra é composto principalmente pelos militares sudaneses e pelos Janjaweed. O Janjaweed é um grupo de milícia sudanesa que vem principalmente das tribos afro-árabes Abbala da região norte de Rizeigat, no Sudão.
Jeddah é a porta de entrada para Meca, a cidade mais sagrada do Islã, que os muçulmanos devem visitar uma vez na vida.
Acredita-se que a Grande Mancha Escura seja um buraco nas nuvens de metano de Netuno.
Sábado segue um dia agitado na vida de um neurocirurgião.
A tarântula, que é complicada, girou um cordão preto para se juntar a uma bola e puxá-la para o leste com toda a sua força.
Ele morreu lá seis semanas depois, em 13 de janeiro de 888.
Eles são semelhantes ao povo da Papua Nova Guiné que vive na costa.
```

Para iniciar essa fase é necessário obter uma base de dados de textos para produzir as paráfrases de treinamento. No projeto original, utilizou-se textos minerados pelo programa [ccnet](https://github.com/facebookresearch/cc_net), que minera textos do Common Crawl. Recomendamos utilizá-lo para extração destes textos, pois ele garante que os textos coletados serão de alta qualidade. Entretanto, não é obrigatório sua utilização. O muss apenas espera que o formato dos textos de entrada sejam similares aos shards que o ccnet gera.

O formato dos arquivos de entrada do script de mineração de paráfrases é um conjunto de JSONs separados por quebra de linha, sendo a última linha do arquivo vazia. Cada JSON precisa conter o campo `raw_content` com o texto que deve ser processado. O ccnet fornece mais dados além deste, porém o muss só espera este campo.

Exemplo de shard gerado pelo ccnet:

```
{"url": "http://aapsocidental.blogspot.com/2018/05/autoridades-de-ocupacao-marroquinas.html", "date_download": "2019-01-16T00:32:20Z", "digest": "sha1:G4UHEYCPVGMKCO37M67XGN7Y5QJ7U7GM", "length": 2022, "nlines": 10, "source_domain": "aapsocidental.blogspot.com", "title": "Sahara Ocidental Informação: Autoridades de ocupação marroquinas transferem arbitrariamente o ativista saharaui Hassanna Duihi", "raw_content": "Autoridades de ocupação marroquinas transferem arbitrariamente o ativista saharaui Hassanna Duihi\n27 de maio, 2018 - Liga para la Protección de los Presos Saharauis en las cárceles marroquíes - As autoridades de ocupação marroquinas tomaram a decisão de transferir arbitrariamente o vice-presidente da Liga para a Proteção dos Presos Saharauis em cárceres marroquinos, Hassanna Duihi, de El Aaiún para a cidade ocupada de Bojador, após a sentença proferida pelo Tribunal de Apelação em Marraquexe, apesar da sentença do tribunal de primeira instância tenha decretado a anulação da decisão da transferência arbitrária..", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 290, "original_length": 13580, "line_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "language": "pt", "language_score": 0.96, "perplexity": 157.6, "bucket": "head"}
{"url": "http://aguaslindasdegoias.go.gov.br/2018/11/22/", "date_download": "2019-01-15T23:49:20Z", "digest": "sha1:FIBOHPZ7BYGPBRTEFUHQJB7UMBEKPVGK", "length": 309, "nlines": 2, "source_domain": "aguaslindasdegoias.go.gov.br", "title": "Águas Lindas de Goiás | 2018 novembro 22", "raw_content": "Pregão n° 051/2018 – Seleção da melhor proposta para a aquisição de brita, emulsão RR-1C, conforme especificações previstas no termo de Referência\nPregão n° 054/2018 – Aquisição de câmaras fotográficas digitais que serão utilizados pela Secretaria Municipal de Habitação e Integração Fundiária deste município", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 182, "original_length": 4288, "line_ids": [96, 102], "language": "pt", "language_score": 1.0, "perplexity": 66.7, "bucket": "head"}
{"url": "http://alvopesquisas.com.br/ipixunadopara.asp", "date_download": "2019-01-16T00:17:51Z", "digest": "sha1:3YRVHGS4JQBJ5YWXOTIPT7XX7TJB3U4T", "length": 1232, "nlines": 5, "source_domain": "alvopesquisas.com.br", "title": "Bem-vindo à @lvo Pesquisas!!!", "raw_content": "Em 1958 chegou à região o pioneiro Manoel do Carmo, que, juntamente com sua família, utilizou a via fluvial. O primeiro passo foi construir uma morada e, em seguida o roçado. No seu rastro vieram Irineu Farias, Antonio Cipriano e Manoel Henrique.\nNa esteira do pioneirismo surgiu a primeira casa de comércio,em 1960, de Vicente Fortunato. Instalado em 01 de janeiro de 1993.", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 75, "original_length": 2459, "line_ids": [51, 52, 53, 54, 55], "language": "pt", "language_score": 0.97, "perplexity": 89.1, "bucket": "head"}
{"url": "http://azeitedoalentejo.pt/cepaal/", "date_download": "2019-01-15T23:41:11Z", "digest": "sha1:EH2JEC2BSIVKJU5GPXU6IYRXX7DZWYF4", "length": 1337, "nlines": 10, "source_domain": "azeitedoalentejo.pt", "title": "CEPAAL – Azeite do Alentejo", "raw_content": "O CEPAAL\nO CEPAAL – Centro de Estudos e Promoção do Azeite do Alentejo nasceu em 1999 e é uma associação sem fins lucrativos, sedeada em Moura, que tem como objetivo valorizar e promover o Azeite do Alentejo dentro e fora de Portugal.\nTem entre os seus associados 26 produtores e 13 instituições ligadas ao sector olivícola e oleícola, incluindo organismos do Estado, municípios e universidades.\nNo âmbito das suas atividades, desenvolve ações de promoção do Azeite do Alentejo e é a entidade responsável pela organização do Concurso Nacional de Azeites de Portugal, integrado na Feira Nacional de Agricultura, e pelo Concurso de Azeite Virgem da Feira Nacional de Olivicultura, sendo também a entidade responsável pela organização do Congresso Nacional do Azeite.", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 147, "original_length": 4332, "line_ids": [16, 19, 20, 21, 23, 25, 27, 28, 56, 84], "language": "pt", "language_score": 0.98, "perplexity": 169.4, "bucket": "head"}

```

Exemplo de shard gerado manualmente:

```
{"raw_content": "Tempo de Entrega Até 2 dias após confirmação de pagamento\n1. O prazo de validade para a utilização do Vale Presente é de 03 meses (Três meses) a contar da data de sua compra, que constará do e-mail enviado pela Loja Soma de Dois, ao cliente após a confirmação do pagamento.\n2. O Vale Presente consiste num código que será enviado ao cliente por e-mail, de acordo com seu cadastro no site da Loja Soma de Dois (https://somadedois.com.br), indicando o valor, a data da compra, o código que deverá ser utilizado e o link relativo a este regulamento de utilização.\n3."}
{"raw_content": "Um rotor de elevado desempenho com a diversidade e eficiência para qualquer local\nUtilize uma chave de fendas ou a chave Hunter para, de forma fácil e simples, ajustar o arco de irrigação conforme necessário.\nO FloStop fecha a vazão de água dos aspersores individualmente enquanto o sistema continua a funcionar. Esta situação é ideal para a substituição de bocais ou para desligar aspersores específicos durante trabalhos de manutenção e/ou instalação."}
{"raw_content": "LEI ORGÂNICA DO MUNICÍPIO DE BOM CONSELHO 1990\nSeção I – Disposições Gerais 1° a 4°\nSeção II – Da Divisão Administrativa do Município 5° a 9°\nCap. II – Da Competência do Município 10 a 13\nSeção II – Da Competência Comum 11 a 13\nTÍT. II – DA ORGANIZAÇÃO DOS PODERES 15 A 89\nCap. I – Do Poder Legislativo 15 a 69\nSeção I – Da Câmara Municipal 15 a 16\nSeção II – Das Atribuições da Câmara Municipal 17 a 18\nSeção III – Do Funcionamento da Câmara 19 a 32\nSeção V – Das Comissões 39 a 40\nSeção VI – Do Processo Legislativo 41 a 56\nSub. I – Disposições Gerais 41 a 42\nSub. II – Das Emendas à Lei Orgânica 43\nSub. IV – Dos Decretos legislativos e das Resoluções 55 a 56\nSeção VIII – Dos Vereadores 59 a 69\nCap. II – Do Poder Executivo 70 a 89\nSeção II – Do Prefeito e do Vice-Prefeito 73 a 78\nSeção III – Da Competência do Prefeito 79 a 80\nSeção IV – Da responsabilidade do Prefeito 81 a 83\nSeção V – Dos Auxiliares Diretos do Prefeito 84 a 89\nCap. II – Da Administração Pública 91 a 116\nSeção II – Das Obras e Serviços Municipais 94 a 100\nSeção III – Dos Bens Municipais 101 a 110\nSeção IV – Dos Servidores Públicos 111 a 114\nSeção V – Da Segurança Publica 115 a 116\nCap. III – Da Estrutura Administrativa 117\nCap. IV – Dos atos Municipais 118 a 122\nSeção I – Da Publicidade dos Atos Municipais 118 a 120\nCap. I – Dos Tributos Municipais 123 a 133\nCap. II – Dos Preços Públicos 134 a 135\n"}
{"raw_content": "É com muita satisfação que começo essa coluna sobre Dança e Nutrição! Sou muito grata pelo convite da Dryelle e espero a cada semana poder trazer temas importantes para que nós, bailarinos, tenhamos boa saúde e bom desempenho por meio de uma alimentação saudável e, o mais importante, sem neuras!\nVou começar falando sobre a importância da nutrição nas modalidades de dança que, embora sejam uma linda expressão artística, também são atividades físicas que requerem muito desempenho físico. Em geral, começa-se a praticar na infância e na adolescência, mas atualmente muitos adultos também aderiram a essas modalidades.\nCada tipo de dança desenvolve aptidões físicas específicas que exigem dos bailarinos resistência muscular e esquelética, osteoarticular, flexibilidade, bom condicionamento cardiorrespiratório e uma composição corporal magra e esguia.\n"}
```

O problema em utilizar o ccnet é que o custo computacional para minerar os textos é bastante elevado, sendo bastante maior do que o custo necessário para treinar o MUSS. Por este motivo, para o treinamento do MUSS para português utilizou-se o dataset do ccnet compartilhado [aqui](https://data.statmt.org/cc-100/). Neste repositório, há 116 arquivos contendo o conteúdo minerado para esses idiomas. Cada arquivo é um único txt comprido, podendo ter mais de 80Gb. O arquivo para o idioma português contém 13Gb comprimido e mais de 50Gb descomprimido. Estes arquivos são formatados contendo o texto de cada site minerado pelo ccnet separados por uma linha em banco e a última linha do arquivo é vazia. Dessa forma, é como se fosse um arquivo gigante onde cada parágrafo é o texto completo de cada site minerado.

```
Site1_Linha1
Site1_Linha2
Site1_LinhaN

Site2_Linha1
Site2_Linha2
Site2_LinhaN

SiteN_Linha1
SiteN_LinhaN

```

Para manipular este arquivo, primeiro realizou-se sua divisão em arquivos com 1.500.000 linhas cada utilizando a ferramenta para Windows [Text File Split](https://www.microsoft.com/store/productId/9PFNL897RKKM), totalizando 228 arquivos, porém pode-se utilizar qualquer script customizado para essa tarefa. Em seguida, converteu-se cada arquivo para o formato do ccnet. O script Python utilizado para realizar essa formatação está ilustrado abaixo. 

```python
import json
import gzip

for i in range(1,228):
    source_file = open(f'cc100/cc100-{i}', 'r', encoding='utf-8')
    out_file = gzip.open(f'cc100_mined/pt_head_{i:04d}.json.gz', 'wt', encoding='utf-8')
    textSentence = ''
  
    while 1:   
        
        line = source_file.readline()           
        if len(line) == 0:  
            break

        if line != '\n':
            textSentence += line

        if(line == '\n'):
            textObj = {"raw_content": textSentence}
            jsonLine = json.dumps(textObj, ensure_ascii=False) + '\n'
            out_file.write(jsonLine)
            textSentence = ''
            continue

    print(f'Arquivo cc100-{i} concluido!')
    source_file.close()
    out_file.close()

print('leitura concluida com sucesso!!')
  
source_file.close()
out_file.close()
```

O funcionamento do script é bastante simples, ele apenas abre cada shard e lê o texto até encontrar uma linha em branco; converte esse texto em um JSON adicionando o texto ao campo `raw_content` e escreve esse conteúdo no arquivo formatado. Para executar esses passos de formatação do dataset do ccnet para o formato esperado para o MUSS, garanta que seu computador possua armazenamento suficiente. Para o dataset de português, foi necessário em torno de 150 Gb de armazenamento para realizar essa operação. Após a produção dos arquivos formatados pode-se deletar os demais arquivos baixados e processados. 

Após finalizar a coleta dos textos e formatá-los corretamente, agora é necessário adaptar o código do muss para atender ao novo idioma. Nessa fase de mineração de paráfrases, será necessário alterar os arquivos `muss/scripts/mine_sequences.py`, `muss/mining/preprocessing.py` e `muss/text.py`. 

No arquivo `muss/scripts/mine_sequences.py` adicione ao objeto `n_shards`o número de shards que deseja minerar, o que equivale ao número de shards produzidos na formatação discutida acima ou minerados utilizando o ccnet. Para a adaptação para português, utilizou-se 6 shard contendo 1.500.000 linhas cada arquivo, totalizando 9 milhões de linhas de texto.

No arquivo `muss/text.py` adapte as funções `get_spacy_model` e `get_sentence_tokenize` para o novo idioma. 

No arquivo `muss/mining/preprocessing.py` adapte a função `has_low_lm_prob` ao novo idioma. Essa função é responsável por filtrar as sentenças produzidas de acordo com um modelo de perplexidade. Essa etapa não é obrigatória, então pode-se forçar essa função a sempre retornar o valor False, porém é fortemente recomendado que seja utilizado um modelo similar para essa tarefa. Para a adaptação para português, utilizou-se o modelo Kenlm disponível no [huggingface](https://huggingface.co/edugp/kenlm).

Feito isso, já está tudo pronto para executar o script `muss/scripts/mine_sequences.py.` Como esse script utiliza muitos recursos computacionais e é bastante demorado, execute-o de forma a [não depender da conexão SSH](https://www.linuxdescomplicado.com.br/2017/07/saiba-como-manter-um-comando-executando-mesmo-depois-de-encerrar-uma-sessao-remota-ssh.html) com a VM. Para isso utilize o comando nohup, conforme ilustrado abaixo. Ao utilizar o nohup, a saída do comando é escrita no arquivo nohup.out. 

```bash
nohup python3 scripts/mine_sequences.py &
```

Para acompanhar a execução do programa, basta monitorar o desempenho da VM pela central de monitoramento da GCloud. Os processos gerados pelo script geram logs na pasta experiments, que é criada automaticamente na primeira execução. Monitorando os arquivos gerados nessa pasta você conseguirá acompanhar o processamento do script.

Esse script irá demorar bastante para executar e poderá gerar alguns problemas de estouro e vazamento de memória. Na adaptação para português, esse script demorou cerca de 36 horas. Algumas adaptações precisaram ser feitas para evitar vazamentos de memória e discussão sobre isso foi feita aqui [nesta issue](https://github.com/facebookresearch/muss/issues/32). Além disso, o parâmetro `max_tokens` da função `get_laser_embeddings` foi alterada para 800.

### Treinamento do modelo

Nesta fase, o objetivo é treinar modelo mBART utilizando as paráfrases produzidas na etapa anterior. O MUSS utiliza o modelo mBART pré treinado e configura o treinamento utilizando o [ACCESS](https://github.com/facebookresearch/access).

Para adicionar um novo idioma, será necessário adaptar os arquivos `muss/fairseq/main.py, muss/mining/training.py e scripts/train_model.py`.

No arquivo `muss/fairseq/main.py` adicione o novo idioma a função `get_language_from_dataset` e mude o valor da variável `TEST_DATASET`. O valor da variável TEST_DATASET deve ser o nome de um dataset manual que será utilizado pelo fairseq para testar a qualidade do modelo durante o treinamento. Esse valor deve corresponder ao nome de uma pasta dentro da pasta resources/datasets que contenha os arquivos test.complex, test.simple, valid.complex e valid.simple. Esses arquivos devem ser diferentes dos gerados na fase anterior e devem ter alta qualidade. Existem diversos benchmarks públicos que podem ser utilizados para essa finalidade como o [Asset](https://github.com/facebookresearch/asset), [Alector](https://github.com/psawa/alector_corpus/tree/master/corpus), [Porsimples](https://github.com/sidleal/porsimplessent), entre outros. Escolha algum deles e traduza para o idioma desejado.

No arquivo `muss/mining/training.py`  mude o valor da variável `TEST_DATASET`igual a etapa anterior e adicione o dicionário do mBART para o novo idioma ao dicionário `MBART_DICT_FILENAME`. Se você utilizar o mbart pré-treinado em 25 idiomas, o nome do dicionário será “`dict.txt`” e se utilizar o [mBART prétreinado em 50 idiomas](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models) o valor  será “`dict.{target_lang}.txt`” (Veja a lista de target_lang [aqui](https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/ML50_langs.txt)). Como o mBART50 é a versão mais recente, recomenda-se utilizá-la.

No arquivo `scripts/train_model.py`, a princípio não será necessário modificar nada, mas caso dê algum problema de estouro ou vazamento de memória pode ser necessário alterar os parâmetros de configuração. Caso já possua um modelo treinado você pode passar o path do modelo no parâmetro `restore_file_path` da função `get_mbart_kwargs.`

Feito isso basta executar o comando:

```bash
nohup python3 scripts/train_model.py NAME_OF_DATASET --language LANGUAGE_OF_TRANNING &
```

O treinamento do modelo em português durou 18 horas.

### Simplificação de sentenças

Após produzir  e treinar um modelo de simplificação textual, pode-se utilizá-lo para simplificar textos quaisquer. Para isso, será necessário adaptar o arquivo `muss/simplify.py.` Adicione o nome do arquivo do modelo treinado no dicionário ALLOWED_MODEL_NAMES. O nome do modelo deve ser o mesmo nome da pasta que está dentro da pasta `muss-ptBR\resources\models` e possui os arquivos model.pt, sentencepiece.bpe.model, dict.complex.txt e dict.simple.txt. Em seguida, adicione o novo idioma nas funções is_model_using_mbart e get_mbart_languages_from_model. 

Feito isso, basta executar o comando:

```bash
 python3 scripts/simplify.py FILE_PATH_TO_SIMPLIFY --model-name MODEL_NAME
```

### Detalhes do procedimento de treinamento

Fase de mineração dos dados:

- Computar os embeddings demora em torno de 36 min para cada shard
- Treinar o index demora em torno de 20 min por shard
- Criar o base index demora em torno de 46 min por shard

## ACCESS

O Controllable Sentence Simplification permite controlar 4 parâmetros da simplificação realizada pelo modelo. Esses parâmteros variam de 0.05 a 2.00, com passo de 0.05. Os detalhes desses parãmetros pode ser consultado no [artigo original](https://arxiv.org/pdf/2005.00352.pdf) do muss.

Estes parâmetros podem ser configurados no arquivo `muss/simplify.py` através da variável TOKENS_RATIO. Variando esses valores o resultado da simplificação muda.

```python
TOKENS_RATIO = {
    "LengthRatioPreprocessor": 1.0,
    "ReplaceOnlyLevenshteinPreprocessor": 0.5,
    "WordRankRatioPreprocessor": 0.9,
    "DependencyTreeDepthRatioPreprocessor": 0.8,
}
```