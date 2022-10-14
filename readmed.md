Analyser App

This app offers a fullt trained Pegasus Large and a Transformer to allow classification of a given text ans its reason. The detection can be made using an API endpoint.

Installation

Local PC:

* git clone https://github.com/augsaksham/AnalyserApp
* cd dev_app/project
* pip install -r requirements.txt
* python -m spacy download en_core_web_sm
* python -m spacy download en
* Goto "saved_mode" dir and donload the pretrained model from readme
* Open console and input : uvicorn main:app
* This will start a local server at localhost:8000

Goto localhost:8000/docs to see the API Documentation

Docker

* git clone https://github.com/augsaksham/AnalyserApp
* cd dev_app/project
* docker build . -t dev_nlp
* docker run -d --name dev_curr -p 80:80 dev_nlp
This will start a local server at localhost:80

Goto localhost:80/docs to see the API Documentation

Features

* Robust classification
* Documented API for easier usage
* Mulitple Models

Traning and Evaluation :

To see the results of training and evalaution heat to comparison.ipynb file.
