# MUSS-ptBR - Textual Simplifier for Portuguese

Author: Raphael Assis (contato.raphael.assis@gmail.com)

## Introduction of the problem

To understand more about the task of textual simplification read [this article](https://direct.mit.edu/coli/article/46/1/135/93384/Data-Driven-Sentence-Simplification-Survey-and).

## Infrastructure used

To carry out this work, the [Google Cloud](https://cloud.google.com/) platform was used . This platform provides all the necessary resources for the implementation of this work and also offers 300 dollars of credits (~1770 reais in 08/2022) to test the services before starting to pay for the use. Because of this, this work can be fully replicated only using the free credits offered by Google.

The infrastructure used was as follows:

Machine with 8 vCPUs, 52 GB memory (n1-highmem-8), 2 TB HDD (boot disk) and 1 NVIDIA Tesla T4 GPU. The operating system used was Ubuntu 20.04 LTS for 64-bit x86 architecture. This configuration results in a cost of $0.69 per hour.

Note: The boot disk does not need to have that much storage volume. You can save even more by using a separate disk to hold the VM data and using a boot disk of about 10Gb. You can see more details about this in [this tutorial](https://cloud.google.com/compute/docs/disks/add-persistent-disk?hl=pt-br). However, when using a boot disk with a lot of storage there is less configuration to perform on the VM.

## Project configuration in the VM

After creating and starting the VM, you need to clone the project from Github and configure the project's dependencies. Also, as the VM starts with a clean Linux image, it is necessary to update some programs. The necessary steps are as follows:

1. Execute `sudo apt-get update`
2. Execute `sudo apt-get install python3-pip`
3. Execute `sudo apt-get install zip`
4. Execute `sudo apt install unzip`
5. Execute `sudo apt install python3.8-venv`
6. Execute `sudo apt-get install build-essential cmake`
7. Execute `sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev`
8. Clone the code from Github: `git clone git@github.com:facebookresearch/muss.git`
9. Navigate to the project folder: `cd muss/`
10. Execute `pip install -e .`
11. Follow  [this tutorial](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=pt-br#verify-driver-install) to install GPU drivers on VM.
12. Follow  [this tutorial](https://cloud.google.com/compute/docs/gpus/monitor-gpus#use-virtualenv_1) to configure GPU telemetry on the VM so you can monitor its performance while running model trainings.
13. pload the files with the text corpus to the VM. See [this tutorial](https://cloud.google.com/compute/docs/instances/transfer-files?hl=pt-br#upload-to-the-vm) on how to send and receive files to the VM. The folder where the files will be saved doesn't matter (by default it's a folder with your username in \home), because muss takes the path as a parameter.

After performing all the steps above, the VM will be configured and ready to use.

## Adapting muss to a new language

The Multilingual Unsupervised Sentence Simplification (MUSS) is a language model based on BART and mBART that performs textual simplification. In this project, there are both scripts to produce a database of paraphrases for model training and scripts to train and evaluate the model.

### Paraphrase mining phase

In this phase, the pre-processing of the collected texts and the production of paraphrases are carried out to carry out the training of the model. The objective of this phase is to obtain pairs of sentences that represent the complex and simplified version of a sentence. The result of this phase is a folder with the files test.complex, test.simple, train.complex, train.simple, valid.complex and valid.simple. Both files are in txt format, each line consisting of a sentence. Thus, the sentence on line 1 of the test.complex file is the complex version of the sentence on line 1 of the test.simple file.

Example file with complex sentences: 

```
One side of the armed conflicts is made up mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited primarily from the Afro-Arab Abbala tribes of the northern Rizeigat region of Sudan.
Jeddah is the main gateway to Mecca, Islam's holiest city, which sane Muslims are obligated to visit at least once in their lifetime.
The Great Dark Spot is believed to represent a hole in Neptune's methane cloud deck.
His next job, Saturday, follows an especially eventful day in the life of a successful neurosurgeon.
The tarantula, the trickster character, spun a black rope and, attaching it to the ball, quickly crawled east, pulling the rope with all his might.
There he died six weeks later, on January 13, 888.
They are culturally similar to the coastal peoples of Papua New Guinea.
```

Example file with simple sentences: 

```
One side of the war is made up mainly of the Sudanese military and the Janjaweed. The Janjaweed is a Sudanese militia group that comes mainly from the Afro-Arab Abbala tribes of the northern Rizeigat region of Sudan.
Jeddah is the gateway to Mecca, Islam's holiest city, which Muslims must visit once in their lifetime.
The Great Dark Spot is believed to be a hole in Neptune's methane clouds.
Saturday follows an eventful day in the life of a neurosurgeon.
The tarantula, which is tricky, spun a black cord to join a ball and pull it east with all its might.
He died there six weeks later, on January 13, 888.
They are similar to the people of Papua New Guinea who live on the coast.
```

To start this phase, it is necessary to obtain a database of texts to produce the training paraphrases. The original project used texts mined by the [ccnet](https://github.com/facebookresearch/cc_net) program , which mines Common Crawl texts. We recommend using it to extract these texts, as it guarantees that the collected texts will be of high quality. However, its use is not mandatory. muss just expects the format of the input texts to be similar to the shards that ccnet generates.

The format of the paraphrase mining script input files is a set of line-separated JSONs, with the last line of the file being empty. Each JSON needs to contain the field `raw_content` with the text that must be processed. ccnet provides more data than this, but muss only expects this field.

Example shard generated by ccnet:

```
{"url": "http://aapsocidental.blogspot.com/2018/05/autoridades-de-ocupacao-marroquinas.html", "date_download": "2019-01-16T00:32:20Z", "digest": "sha1:G4UHEYCPVGMKCO37M67XGN7Y5QJ7U7GM", "length": 2022, "nlines": 10, "source_domain": "aapsocidental.blogspot.com", "title": "Sahara Ocidental Informação: Autoridades de ocupação marroquinas transferem arbitrariamente o ativista saharaui Hassanna Duihi", "raw_content": "Autoridades de ocupação marroquinas transferem arbitrariamente o ativista saharaui Hassanna Duihi\n27 de maio, 2018 - Liga para la Protección de los Presos Saharauis en las cárceles marroquíes - As autoridades de ocupação marroquinas tomaram a decisão de transferir arbitrariamente o vice-presidente da Liga para a Proteção dos Presos Saharauis em cárceres marroquinos, Hassanna Duihi, de El Aaiún para a cidade ocupada de Bojador, após a sentença proferida pelo Tribunal de Apelação em Marraquexe, apesar da sentença do tribunal de primeira instância tenha decretado a anulação da decisão da transferência arbitrária..", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 290, "original_length": 13580, "line_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "language": "pt", "language_score": 0.96, "perplexity": 157.6, "bucket": "head"}
{"url": "http://aguaslindasdegoias.go.gov.br/2018/11/22/", "date_download": "2019-01-15T23:49:20Z", "digest": "sha1:FIBOHPZ7BYGPBRTEFUHQJB7UMBEKPVGK", "length": 309, "nlines": 2, "source_domain": "aguaslindasdegoias.go.gov.br", "title": "Águas Lindas de Goiás | 2018 novembro 22", "raw_content": "Pregão n° 051/2018 – Seleção da melhor proposta para a aquisição de brita, emulsão RR-1C, conforme especificações previstas no termo de Referência\nPregão n° 054/2018 – Aquisição de câmaras fotográficas digitais que serão utilizados pela Secretaria Municipal de Habitação e Integração Fundiária deste município", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 182, "original_length": 4288, "line_ids": [96, 102], "language": "pt", "language_score": 1.0, "perplexity": 66.7, "bucket": "head"}
{"url": "http://alvopesquisas.com.br/ipixunadopara.asp", "date_download": "2019-01-16T00:17:51Z", "digest": "sha1:3YRVHGS4JQBJ5YWXOTIPT7XX7TJB3U4T", "length": 1232, "nlines": 5, "source_domain": "alvopesquisas.com.br", "title": "Bem-vindo à @lvo Pesquisas!!!", "raw_content": "Em 1958 chegou à região o pioneiro Manoel do Carmo, que, juntamente com sua família, utilizou a via fluvial. O primeiro passo foi construir uma morada e, em seguida o roçado. No seu rastro vieram Irineu Farias, Antonio Cipriano e Manoel Henrique.\nNa esteira do pioneirismo surgiu a primeira casa de comércio,em 1960, de Vicente Fortunato. Instalado em 01 de janeiro de 1993.", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 75, "original_length": 2459, "line_ids": [51, 52, 53, 54, 55], "language": "pt", "language_score": 0.97, "perplexity": 89.1, "bucket": "head"}
{"url": "http://azeitedoalentejo.pt/cepaal/", "date_download": "2019-01-15T23:41:11Z", "digest": "sha1:EH2JEC2BSIVKJU5GPXU6IYRXX7DZWYF4", "length": 1337, "nlines": 10, "source_domain": "azeitedoalentejo.pt", "title": "CEPAAL – Azeite do Alentejo", "raw_content": "O CEPAAL\nO CEPAAL – Centro de Estudos e Promoção do Azeite do Alentejo nasceu em 1999 e é uma associação sem fins lucrativos, sedeada em Moura, que tem como objetivo valorizar e promover o Azeite do Alentejo dentro e fora de Portugal.\nTem entre os seus associados 26 produtores e 13 instituições ligadas ao sector olivícola e oleícola, incluindo organismos do Estado, municípios e universidades.\nNo âmbito das suas atividades, desenvolve ações de promoção do Azeite do Alentejo e é a entidade responsável pela organização do Concurso Nacional de Azeites de Portugal, integrado na Feira Nacional de Agricultura, e pelo Concurso de Azeite Virgem da Feira Nacional de Olivicultura, sendo também a entidade responsável pela organização do Congresso Nacional do Azeite.", "cc_segment": "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/wet/CC-MAIN-20190115225438-20190116011438-00000.warc.wet.gz", "original_nlines": 147, "original_length": 4332, "line_ids": [16, 19, 20, 21, 23, 25, 27, 28, 56, 84], "language": "pt", "language_score": 0.98, "perplexity": 169.4, "bucket": "head"}

```

Example of manually generated shard:

```
{"raw_content": "Tempo de Entrega Até 2 dias após confirmação de pagamento\n1. O prazo de validade para a utilização do Vale Presente é de 03 meses (Três meses) a contar da data de sua compra, que constará do e-mail enviado pela Loja Soma de Dois, ao cliente após a confirmação do pagamento.\n2. O Vale Presente consiste num código que será enviado ao cliente por e-mail, de acordo com seu cadastro no site da Loja Soma de Dois (https://somadedois.com.br), indicando o valor, a data da compra, o código que deverá ser utilizado e o link relativo a este regulamento de utilização.\n3."}
{"raw_content": "Um rotor de elevado desempenho com a diversidade e eficiência para qualquer local\nUtilize uma chave de fendas ou a chave Hunter para, de forma fácil e simples, ajustar o arco de irrigação conforme necessário.\nO FloStop fecha a vazão de água dos aspersores individualmente enquanto o sistema continua a funcionar. Esta situação é ideal para a substituição de bocais ou para desligar aspersores específicos durante trabalhos de manutenção e/ou instalação."}
{"raw_content": "LEI ORGÂNICA DO MUNICÍPIO DE BOM CONSELHO 1990\nSeção I – Disposições Gerais 1° a 4°\nSeção II – Da Divisão Administrativa do Município 5° a 9°\nCap. II – Da Competência do Município 10 a 13\nSeção II – Da Competência Comum 11 a 13\nTÍT. II – DA ORGANIZAÇÃO DOS PODERES 15 A 89\nCap. I – Do Poder Legislativo 15 a 69\nSeção I – Da Câmara Municipal 15 a 16\nSeção II – Das Atribuições da Câmara Municipal 17 a 18\nSeção III – Do Funcionamento da Câmara 19 a 32\nSeção V – Das Comissões 39 a 40\nSeção VI – Do Processo Legislativo 41 a 56\nSub. I – Disposições Gerais 41 a 42\nSub. II – Das Emendas à Lei Orgânica 43\nSub. IV – Dos Decretos legislativos e das Resoluções 55 a 56\nSeção VIII – Dos Vereadores 59 a 69\nCap. II – Do Poder Executivo 70 a 89\nSeção II – Do Prefeito e do Vice-Prefeito 73 a 78\nSeção III – Da Competência do Prefeito 79 a 80\nSeção IV – Da responsabilidade do Prefeito 81 a 83\nSeção V – Dos Auxiliares Diretos do Prefeito 84 a 89\nCap. II – Da Administração Pública 91 a 116\nSeção II – Das Obras e Serviços Municipais 94 a 100\nSeção III – Dos Bens Municipais 101 a 110\nSeção IV – Dos Servidores Públicos 111 a 114\nSeção V – Da Segurança Publica 115 a 116\nCap. III – Da Estrutura Administrativa 117\nCap. IV – Dos atos Municipais 118 a 122\nSeção I – Da Publicidade dos Atos Municipais 118 a 120\nCap. I – Dos Tributos Municipais 123 a 133\nCap. II – Dos Preços Públicos 134 a 135\n"}
{"raw_content": "É com muita satisfação que começo essa coluna sobre Dança e Nutrição! Sou muito grata pelo convite da Dryelle e espero a cada semana poder trazer temas importantes para que nós, bailarinos, tenhamos boa saúde e bom desempenho por meio de uma alimentação saudável e, o mais importante, sem neuras!\nVou começar falando sobre a importância da nutrição nas modalidades de dança que, embora sejam uma linda expressão artística, também são atividades físicas que requerem muito desempenho físico. Em geral, começa-se a praticar na infância e na adolescência, mas atualmente muitos adultos também aderiram a essas modalidades.\nCada tipo de dança desenvolve aptidões físicas específicas que exigem dos bailarinos resistência muscular e esquelética, osteoarticular, flexibilidade, bom condicionamento cardiorrespiratório e uma composição corporal magra e esguia.\n"}
```

The problem with using ccnet is that the computational cost to mine the texts is quite high, being much higher than the cost needed to train the MUSS. For this reason, for training the MUSS in Portuguese, the ccnet dataset shared [here](https://data.statmt.org/cc-100/) was used . In this repository, there are 116 files containing the mined content for these languages. Each file is a single long txt, which can be over 80Gb. The file for the Portuguese language contains 13Gb compressed and more than 50Gb uncompressed. These files are formatted containing the text of each site mined by ccnet separated by a line in bank and the last line of the file is empty. That way, it's like a giant file where each paragraph is the complete text of each site mined.

```
Site1_Line1
Site1_Line2
Site1_LineN

Site2_Line1
Site2_Line2
Site2_LineN

SiteN_Line1
SiteN_LineN

```

o manipulate this file, it was first divided into files with 1,500,000 lines each using the Windows [Text File Split](https://www.microsoft.com/store/productId/9PFNL897RKKM) tool, totaling 228 files, however any customized script can be used for this task. Then each file was converted to ccnet format. The Python script used to perform this formatting is illustrated below.

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

How the script works is quite simple, it just opens each shard and reads the text until it finds a blank line; converts this text to a JSON by adding the text to the field `raw_content` and writes this content to the formatted file. To perform these steps to format the ccnet dataset to the expected format for the MUSS, ensure your computer has enough storage. For the Portuguese dataset, around 150 Gb of storage was needed to perform this operation. After producing the formatted files, you can delete the other downloaded and processed files.

After finishing collecting the texts and formatting them correctly, it is now necessary to adapt the muss code to suit the new language. In this phase of paraphrase mining, it will be necessary to change the `muss/scripts/mine_sequences.py`, `muss/mining/preprocessing.py` and `muss/text.py`.

In the file `muss/scripts/mine_sequences.py` add to the object `n_shards` the number of shards you want to mine, which is equivalent to the number of shards produced in the format discussed above or mined using ccnet. For the adaptation to Portuguese, 6 shards containing 1,500,000 lines each file were used, totaling 9 million lines of text.

In the file `muss/text.py` adapt the functions `get_spacy_model` and `get_sentence_tokenize` to the new language.

In the file `muss/mining/preprocessing.py`, adapt the function `has_low_lm_prob` to the new language. This function is responsible for filtering the sentences produced according to a perplexity model. This step is not mandatory, so you can force this function to always return the False value, but it is strongly recommended that you use a similar model for this task. For the Portuguese adaptation, the Kenlm model available on [huggingface](https://huggingface.co/edugp/kenlm) was used.

That done, everything is ready to run the script `muss/scripts/mine_sequences.py`. As this script uses a lot of computational resources and is quite time-consuming, run it in a way that [it does not depend on the SSH connection](https://www.linuxdescomplicado.com.br/2017/07/saiba-como-manter-um-comando-executando-mesmo-depois-de-encerrar-uma-sessao-remota-ssh.html) to the VM. To do so, use the nohup command, as illustrated below. When using nohup, the output of the command is written to the nohup.out file. 

```bash
nohup python3 scripts/mine_sequences.py &
```

To monitor the execution of the program, simply monitor the VM's performance through the GCloud monitoring center. Processes generated by the script generate logs in the experiments folder, which is automatically created on first run. By monitoring the files generated in this folder you will be able to follow the script processing.

This script will take a long time to run and may cause some overflow and memory leak issues. In the Portuguese adaptation, this script took about 36 hours. Some adaptations needed to be made to avoid memory leaks and discussion about that was done here [in this issue](https://github.com/facebookresearch/muss/issues/32). Also, the `max_tokens` function parameter `get_laser_embeddings` was changed to 800.

### Model training

In this phase, the objective is to train the mBART model using the paraphrases produced in the previous step. MUSS takes the pre-trained mBART model and sets up training using [ACCESS](https://github.com/facebookresearch/access).

To add a new language, you will need to adapt the `muss/fairseq/main.py, muss/mining/training.py and scripts/train_model.py`.

In the file `muss/fairseq/main.py` add the new language to the function `get_language_from_dataset` and change the value of the variable `TEST_DATASET`. The value of the TEST_DATASET variable must be the name of a manual dataset that will be used by fairseq to test the model's quality during training. This value must match the name of a folder within the resources/datasets folder that contains the test.complex, test.simple, valid.complex, and valid.simple files. These files must be different from those generated in the previous phase and must be of high quality. There are several public benchmarks that can be used for this purpose, such as [Asset](https://github.com/facebookresearch/asset), [Alector](https://github.com/psawa/alector_corpus/tree/master/corpus), [Porsimples](https://github.com/sidleal/porsimplessent), among others. Choose any of them and translate it into the desired language.

In the file `muss/mining/training.py` change the value of the variable  `TEST_DATASET` equal to the previous step and add the mBART dictionary for the new language to the dictionary  `MBART_DICT_FILENAME`.  If you use pre-trained mBART in 25 languages, the dictionary name will be `dict.txt` and if you use pre-trained [mBART in 50 languages](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models) the value will be `dict.{target_lang}.txt` (See target_lang list [aqui](https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/ML50_langs.txt)).As mBART50 is the latest version, it is recommended to use it.

In the file `scripts/train_model.py`, at first it will not be necessary to modify anything, but if there is a problem with overflow or memory leaks, it may be necessary to change the configuration parameters. If you already have a trained model, you can pass the model's path in `restore_file_path` parameter in the function `get_mbart_kwargs.`

Once that's done, just run the command:

```bash
nohup python3 scripts/train_model.py NAME_OF_DATASET --language LANGUAGE_OF_TRANNING &
```

Model training in Portuguese lasted 18 hours.

### Simplification of sentences

After producing and training a textual simplification model, it can be used to simplify any text. For this, you will need to adapt the file `muss/simplify.py.` Add the trained model file name in the ALLOWED_MODEL_NAMES dictionary. The name of the model must be the same as the name of the folder that is inside the folder `muss-ptBR\resources\models` and has the files model.pt, sentencepiece.bpe.model, dict.complex.txt and dict.simple.txt. Then add the new language in the is_model_using_mbart and get_mbart_languages_from_model functions.

Once that's done, just run the command:

```bash
 python3 scripts/simplify.py FILE_PATH_TO_SIMPLIFY --model-name MODEL_NAME
```

### Training procedure details

Data mining phase:

- Computing the embeddings takes around 36 min for each shard
- Training index takes around 20 min per shard
- Creating the base index takes around 46 min per shard

## ACCESS

The Controllable Sentence Simplification allows controlling 4 parameters of the simplification performed by the model. These parameters range from 0.05 to 2.00, with a step of 0.05. The details of these parameters can be found in the [original muss article](https://arxiv.org/pdf/2005.00352.pdf).

These parameters can be configured in the file `muss/simplify.py` hrough the TOKENS_RATIO variable. By varying these values ​​the result of the simplification changes.

```python
TOKENS_RATIO = {
    "LengthRatioPreprocessor": 1.0,
    "ReplaceOnlyLevenshteinPreprocessor": 0.5,
    "WordRankRatioPreprocessor": 0.9,
    "DependencyTreeDepthRatioPreprocessor": 0.8,
}
```