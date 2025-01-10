This repository contains the code of the paper "Improving the Quality of Document Embeddings with Post-OCR Correction on a Multilingual Historical Corpus" (Piras, Rosell, Casals, 2025)

The dataset, which includes scanned documents of the Biblioteca Nacional de Catalunya, together with their OCR texts, can be found at https://zenodo.org/records/14628712.

### Abstract

The digitization of libraries and archives promises to revolutionize the way cultural heritage assets are managed and accessed. However, historical documents still present many open challenges, including, but not limited to, difficulties in the Optical Character Recognition (OCR) and the search for an effective indexation of the corpus. In this paper, we tackle these challenges through a joint approach, which improves the quality of the document embeddings with a post-OCR correction algorithm. A combination of a spellchecker and a BERT-based word-predictor is used to fix the raw texts originally extracted from scanned documents. Subsequently, the BERT model is applied on the processed texts to represent them in a vector space. The quality of the resulting embeddings is assessed based on a clustering task, thus showing, for the first time in the literature, the impact of OCR quality on document clustering. The evaluation, carried out on a new dataset of multilingual historical documents from early 20th century belonging to the Biblioteca Nacional de Catalunya, shows that the post-OCR corrector effectively improves the quality of the extracted texts, which leads to better embeddings and, consequently, to better document clusters.

## Notebook prototyping setup

For prototyping purposes, the notebook setup can be used. The setup for the
prototyping system involves building the docker image, preparing the volumes
and running the Jupyter service.

### Create environment file 

The command to generate the environment file (.env) is the following:

```commandline
printf "UID=$(id -u)\nGID=$(id -g)\n" > .env 
```

This command writes the user id and group id in a file. For default Docker 
gets the environment variables from this file.

### Docker image

From the root of the project (/metaglam) do the following:

```commandline
docker build -f Dockerfile_jupyter -t metaglam_jupyter_environment .
```

With this we have obtained a docker image which will take all necessary source
files, and will offer a Jupyter lab service synchronised with them.

#### Check the volumes

The system needs to interact with some folders in order to be able to take
necessary data, train models or predict. From the root folder (/metaglam), the
volumes are mounted as follows:

```yaml
    volumes:
      - ./src:/home/jovyan/work/src
      - ./notebooks:/home/jovyan/work/notebooks
      - ./data/raw:/data/external
      - ./data/raw:/data/interim
      - ./data/raw:/data/models
      - ./data/raw:/data/processed
      - ./data/raw:/data/raw
```

For prototyping purposes, it is not advisable to change the above volumes for
consistency across projects.

### Start/shutdown the system

When ready, start the system from the root folder as follows:

```commandline
docker compose -f docker-compose_jupyter.yml up
```

Shutdown the system as follows:

```commandline
docker compose -f docker-compose_jupyter.yml down
```

### Utility to start/shutdown the system

To facilitate system startup and shutdown, there is a script available. 
Please execute the following command from the root folder:

```commandline
make deploy_jupyter
```

This command initiates the API and displays the logs in the command line. 
To shutdown the system, press "CTRL + C".

### Notebook

The notebook service is accessible by default through the port 8888. It
requires an access token, which is displayed when the system starts. The main
notebook entry can be accessed by directly entering the link shown in
the console.

E.g.

```
http://127.0.0.1:8888/lab?token=metaglam
```

