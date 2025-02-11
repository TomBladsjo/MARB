# MARB dataset

To create a MARB-style dataset, use the script `code/create_dataset.py` and supply
* text files containing the original sequences to modify
* CSV files containing the terms to insert for each category
* name of the output directory for the finished datasets

## Contents

### categories
Categories and terms are supplied in CSV files named <category>.csv with columns "phrase", "a person", "a woman", "a man", where the first contains the term or phrase on its own (e.g. "Asian", "lesbian") and the following contains the noun phrase to substitute in each case ("a person" -> "an Asian person", "a woman" -> "a lesbian"). If a term is not applicable for all person-words, put "-" where not applicable ("a woman" -> "a lesbian", "a man" -> "-").

Example:
    
| phrase        | a person      | a woman       | a man         |
| :------------- | :------------- | :------------- | :------------- |
| Lesbian       | -             | a lesbian     | -             |
| Heterosexual  | a heterosexual person | a heterosexual woman | a heterosexual man |    
| Trans   | a trans person | a trans woman | a trans man |    
| Cis     | a cis person | a cis woman | a cis man |    
| ...     |... | ... | ... |    
    

### datasets
    
The datasets folder contains CSV files corresponding to the categories: disability.csv, race.csv and queerness.csv. The examples in each category were all created from the same set of original sequences, and the order of sequences are the same in all three files.

### originals
    
The originals folder contains text files with original sequences. These can be used with the `code/create_dataset.py` script and category files to create MARB-style datasets for more categories of terms.

