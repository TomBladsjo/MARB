# MARB 
### – A Dataset for Studying the Social Dimensions of Reporting Bias in Language Models.

This repo contains the MARB dataset, presented at the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP).
The current file contains an introduction to the dataset, as well as instructions on how to download and use it.
If you want to know more about the dataset, feel free to check out [the paper](https://aclanthology.org/2025.gebnlp-1.5/) and [the datacard](https://github.com/TomBladsjo/MARB/blob/main/datacard.md).

### About

Reporting bias (the human tendency to not mention obvious or redundant information) and social bias (societal attitudes toward specific demographic groups) have both been shown to propagate from human text data to language models trained on such data. 
In descriptions of people, reporting bias can manifest as a tendency to over report on attributes that deviate from the norm.
Despite this, the intersection of social bias and reporting bias remains underexplored.
The MARB dataset was developed to begin to fill this gap by studying the interaction between social biases and reporting bias in both human text and language models.
Unlike many existing benchmark datasets, MARB does not rely on artificially constructed templates or crowdworkers to create contrasting examples. 
Instead, the templates used in MARB are based on naturally occurring written language from the 2021 version of the enTenTen corpus [(Jakubíček et al., 2013)](https://www.sketchengine.eu/ententen-english-corpus/). 

### Dataset

The dataset consists of 28.5K template sequences – 9.5K containing each of the phrases "a person", "a woman", "a man" – and their modifications.
It covers three categories of sensitive attributes: *Disability*, *Race* and *Queerness*.
Each category comes with a list of expressions pertaining to the category (for example, the expressions in the *Race* category are "Native American", "Asian", "Black"/"black", "Hispanic", "Native Hawaiian" and "white"). 
Each of these expressions are inserted as modifiers to each person-word ("a person" -> "an Asian person"), resulting in a total of nearly 1M modified sequences.
These can be used to investigate whether a model expects to see certain attributes mentioned more than others.

### Usage

##### Dataset

The full dataset can be found at `data/data.tar.gz`. 
To create a MARB-style dataset from your own data, use `code/create_dataset.py`.

##### Model evaluation

To evaluate a model on the dataset, run:

`python code/test_models.py <model> <path to dataset directory> <path to directory for result files>`

where `model` is the model name as a string (case insensitive).
Available models at the moment are (MLMs:) 'BERT', 'Roberta', 'Albert', (Generative:) 'GPT2', 'Bloom', 'OPT', 'Mistral'.
To see all available options, run
`python code/test_models.py -h`.

The output results (as per-sequence log-likelihods or perplexities or as full dataset perplexity, default is per-sequence perplexity) will be stored as CSV files (one for each model and category).


### Citing MARB

If you use the MARB dataset, please cite [the paper describing it](https://aclanthology.org/2025.gebnlp-1.5/).
The standard citation can be found below.

> Tom Södahl Bladsjö and Ricardo Muñoz Sánchez. 2025. Introducing MARB — A Dataset for Studying the Social Dimensions of Reporting Bias in Language Models. In Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 59–74, Vienna, Austria. Association for Computational Linguistics.

Other references related to MARB can be found below.
Please cite them if you find them useful.

> Tom Södahl Bladsjö. 2024. Don't Mention the Norm. Masters Thesis, Gothenburg, Sweden. University of Gothenburg.
