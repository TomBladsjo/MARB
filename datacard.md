# Data Card

This datacard is based on the template used to document the resources that form part of SuperLim.
Said template can be found [here](https://github.com/spraakbanken/SuperLim).


## 1. Identifying Information

### General Information

- **Title:** MARB
- **Subtitle:** A dataset for studying Marked Attribute Reporting Bias
- **Created by:** Tom Södahl Bladsjö; & Ricardo Muñoz Sánchez
- **Publisher(s):** N/A
- **License(s):** MIT License
- **Funded by:** N/A
- **Link(s)/permanent identifier(s):**
    - https://github.com/TomBladsjo/MARB/
    - https://doi.org/10.18653/v1/2025.gebnlp-1.5


### Abstract

Reporting bias (the human tendency to not mention obvious or redundant information) and social bias (societal attitudes toward specific demographic groups) have both been shown to propagate from human text data to language models trained on such data.
However, the two phenomena have not previously been studied in combination.
The MARB dataset was developed to begin to fill this gap by studying the interaction between social biases and reporting bias in language models.
Unlike many existing benchmark datasets, MARB does not rely on artificially constructed templates or crowdworkers to create contrasting examples. Instead, the templates used in MARB are based on naturally occurring written language from the 2021 version of the enTenTen corpus [1].


### Citation

> Tom Södahl Bladsjö and Ricardo Muñoz Sánchez. 2025. Introducing MARB — A Dataset for Studying the Social Dimensions of Reporting Bias in Language Models. In Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 59–74, Vienna, Austria. Association for Computational Linguistics.

> Tom Södahl Bladsjö. 2024. Don't Mention the Norm. Masters Thesis, Gothenburg, Sweden. University of Gothenburg.


## 2. Usage	

- **Key applications:** Identify and study reporting bias with regards to human attributes in masked and autoregressive language models
- **Intended task(s)/usage(s):** While MARB can be used to investigate a range of different research questions, it is mainly intended as an intrinsic diagnostic for out-of-the-box language models: How likely is the model to predict the sets of sequences containing different attribute descriptors and person-words? Are there systematic differences depending on the attribute mentioned?
- **Recommended evaluation measures:** Perplexity, Rank biserial correlation (r)
- **Dataset function(s):** Diagnostics
- **Recommended split(s):** Test data only


## 3. Data	

- **Primary data:** Text
- **Language:** English

### Dataset in numbers

28500 template sequences, modified to contain each expression in each category, resulting in a total of 940500 unique sequences.

### Nature of the content

The dataset consists of nearly 28.5K template sequences.
9.5K containing each of the phrases "a person", "a woman", "a man", and their modifications.
The templates are based on naturally occurring written language from the 2021 version of the enTenTen corpus [1].

It covers three categories of sensitive attributes: Disability, Race and Queerness.
Each category comes with a list of expressions pertaining to the category (for example, the expressions in the Race category are "Native American", "Asian", "Black"/"black", "Hispanic", "Native Hawaiian" and "white").
Each of these expressions are inserted as modifiers to each person-word ("a person" -> "an Asian person"), resulting in a total of nearly 1M modified sequences.
These can be used to investigate whether a model expects to see certain attributes mentioned more than others.

### Format

#### Datasets
CSV files for each category of 28501 rows each (first row is column names), with one test item per row.
The first two columns are "person_word", indicating which of the words "person", "woman", "man" is the focus of the test item, and "original", containing the unmodified template sequence.
The following columns are named after the attributes of interest, and each contain a version of the template sequence modified to contain a descriptor for that attribute.
Thus, the number of columns will differ depending on the number of attributes included in that category. 

#### Categories
One CSV file for each category containing the columns "phrase", "a person", "a woman", "a man", where the first contains the term or phrase on its own (e.g. "Asian") and the following contains the noun phrase to substitute in each case ("a person" -> "an Asian person"). 

#### Originals
The originals folder contains text files with original sequences (one sequence per row).
These can be used with the code/create_dataset.py script and category files to create MARB-style datasets for more categories of terms. 

#### Metadata
CSV file containing information about the original sequences, where the row indices correspond to those of "originals.csv" as well as each of the dataset files.
The  "search_phrase" column shows the regular expression used to extract the sequence from the EnTenTen 2021 corpus [1].
The following columns ("token_number", "document_number", "URL", "website" and "crawl_date") contain information about the sequence obtained from [SketchEngine](www.sketchengine.eu).


### Data source(s)

Template sequences were retrieved from the 2021 version of the enTenTen corpus [1].


### Data collection method(s)

Template sequences were extracted using the [concordance tool](https://www.sketchengine.eu/guide/concordance-a-tool-to-search-a-corpus/) at SketchEngine.


### Data selection and filtering

A random sample of 10K sequences was retrieved containing each person word ("person", "woman" and "man").
The 500 shortest sequences for each person word were removed after preprocessing to mitigate the effects of sequence length on evaluation results.


### Data preprocessing

Each sequence was preprocessed to remove context outside of sentence boundaries.
Then for each sentence, the noun phrase containing the person word of interest was identified using regular expressions, and modified versions of the sentence with inserted attribute descriptors were created.


### Data labeling

Modified sequences were labeled automatically with the inserted attribute.


### Annotator characteristics

N/A


## 4. Ethics and Caveats	

### Ethical considerations

This work deals with language categorizing people based on sensitive attributes such as race, gender identity and sexuality.
This is a sensitive topic, and care should be taken not to oversimplify complex real-world power structures or to confuse real-life demographic groups with the words used to describe them.


### Things to watch out for

There are often many ways to refer to a specific social group, and they carry different connotations and underlying assumptions.
For example, both terminology and ontological definitions relating to disability are contested, and there is great variation in the language used both by in-group and out group members.
Additionally, in many cases there is a complete lack of established terms describing normative attributes, such as not having a disability. The attribute descriptors included in MARB should be seen as a sample rather than a comprehensive list of the language used to refer to these groups. 

Furthermore, most common metrics for how well a model predicts a sequence (such as perplexity) are affected by factors such as sequence length and model vocabulary.
While steps have been taken to mitigate these effects (especially that of sequence length), they will still be present to some extent, particularly in cases where an attribute descriptor is divided into many tokens.
When using the dataset for evaluation, take care to analyse your results with this in mind.

Finally, while MARB can be used to identify certain types of bias in a model, a lack of visible bias using MARB is no guarantee that the model is not biased.


### 5. About Documentation	

- **Data last updated:** 20250211
- **Which changes have been made, compared to the previous version:**
    - Shorter sequences have been removed to mitigate the effect of sequence length on the results.
    - The dataset creation script and the format of the category files have been changed to make it easier to modify and extend the existing dataset.
- **Access to previous versions:** Email [tom.sodahl@gmail.com](mailto:tom.sodahl@gmail.com)
- **This document created:** 20250605
- **This document last updated:** 20250819
- **Where to look for further details:** See [2]
- **Documentation template version:** [1.0](https://github.com/spraakbanken/SuperLim/blob/main/documentation_sheet_template_v1.0.tsv)


### 6. Other	

- **Related projects:** N/A
- **References:**
    1. Miloš Jakubíček, Adam Kilgarriff, Vojtěch Kovář, Pavel Rychlý, Vít Suchomel. 2013. The TenTen Corpus Family. In _7th International Corpus Linguistics Conference CL 2013_, Lancaster, United Kingdom.
    2. Tom Södahl Bladsjö and Ricardo Muñoz Sánchez. 2025. Introducing MARB — A Dataset for Studying the Social Dimensions of Reporting Bias in Language Models. In Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 59–74, Vienna, Austria. Association for Computational Linguistics.
