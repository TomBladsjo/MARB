# MARB dataset description

### About
The MARB dataset was developed for studying the interaction between social biases and reporting bias in language models.
It was created using naturally occurring written language sequences from the 2021 version of the enTenTen corpus [(Jakubíček et al., 2013)](https://www.sketchengine.eu/ententen-english-corpus/), retrieved using the *concordance* tool at [SketchEngine](https://www.sketchengine.eu/guide/concordance-a-tool-to-search-a-corpus/).
The dataset consists of 30K template sequences – 10K containing each of the phrases "a person", "a woman", "a man" – and their modifications. It covers three categories of sensitive attributes: Disability, Race and Queerness. Each category comes with a list of expressions pertaining to the category (for example, the expressions in the Race category are "native american", "asian", "black", "hispanic", "pacific islander" and "white"). Each of these expressions are inserted as modifiers to each person-word ("a person" -> "an asian person"), resulting in a total of over 1M modified sequences. These can be used to investigate whether a model expects to see certain attributes mentioned more than others.

### Contents

#### Data

The `data` folder contains three CSV files corresponding to the three categories: `disability.csv`, `race.csv` and `queerness.csv`.
The examples in each category were all created from the same set of original sequences, and the order of sequences are the same in all three files.

#### Metadata

The file `metadata.csv` contains information about each original sequence, its original source and its position in the original corpus. 
It contains columns `search_phrase`, `token_number`, `document_number`, `URL`, `website` and `crawl_date`, and the indices correspond to the sequence indices in each of the files in the `data` folder.
The metadata information was retrieved from [SketchEngine](www.sketchengine.eu).

