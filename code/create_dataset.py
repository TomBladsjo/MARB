import re
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse


class DatasetMaker():
    '''
    An object for creating, handling and saving MARB-type datasets. 
    
    Categories and terms are supplied in CSV files named <category>.csv with columns "phrase", "a person", "a woman", "a man",
    where the first contains the term or phrase on its own (e.g. "Asian", "lesbian") and the following contains the noun phrase 
    to substitute in each case ("a person" -> "an Asian person", "a woman" -> "a lesbian"). If a term is not applicable for 
    all person-words, put "-" where not applicable ("a woman" -> "a lesbian", "a man" -> "-").
    
    '''
    
    originals = {}
    cats = {}
    datasets = {}
    
    # regex = re.compile(r'(\ba)\s\b(man|woman|person)\b', re.I)  
    
    def __init__(self, originals, **kwargs): 
        """
        Args:
        originals: Original sequences to turn into dataset. Can be a single text file, a list or a directory of files.
    
        Kwargs: 
        categories: A single CSV file or a list or directory of CSV files containing terms to add to sequences. 
        One file per category of terms. The name of the file will be the name of the category in the dataset.
        
        """
        
        if 'regex' in kwargs:
            self.regex = kwargs['regex']
        else:
            self.regex = 'auto'
        
        if type(originals) == list:
            for file in originals:
                self.append_originals(file)
        elif os.path.isdir(originals):
            for file in os.listdir(originals):
                self.append_originals(os.path.join(originals, file))
        elif os.path.isfile(originals):
            self.append_originals(originals)
            
        if kwargs['categories']:
            cats = kwargs['categories']
            if type(cats) == list:
                self.cats_from_list(cats)
            elif os.path.isdir(cats):
                self.cats_from_dir(cats)
            elif os.path.isfile(cats):
                assert cats[-4:] == '.csv', f'Error: "category" file must be CSV file. Got "{cats}".'
                self.mkcat(cats)
                
        
            
            
     
    def append_originals(self, file):
        assert file[-4:] == '.txt', f'Error: "originals" file must be a text file. Got "{file}".'
        originals = []
        with open(file, 'r') as f:
            for line in f:
                originals.append(line.strip())
            if self.regex == 'firstline':
                self.originals[originals[0]] = originals[1:]
            else:
                self.originals[self.regex] = originals
                
               # self.originals.append(line.strip()) 
                
    
    def make_examples(self):
        for cat in self.cats: 
            print(f'Creating examples for category {cat}...')
            personwords = self.cats[cat].columns[1:]  
            for key, sents in self.originals.items():
                if key == 'auto':
                    regex = re.compile(r'\b({})\b'.format('|'.join(personwords)), re.I)
                else:
                    regex = re.compile(key, re.I)
                for sent in tqdm(sents):
                    self.mkex(sent, cat, regex)
        print('Done!')
    
    
    def save_datasets(self, savedir='.'):  
        if not Path(savedir).is_dir():
            print(f'Creating directory {savedir}')
            os.mkdir(savedir)
        print(f'Saving datasets to {savedir}:')
        
        for cat in self.datasets:
            print(f'Saving {cat}.csv...')
            df = pd.DataFrame(self.datasets[cat])
            df.to_csv(os.path.join(savedir, f'{cat}.csv'), index=False)
        print('\nDone!')
    
                        
    def mkcat(self, file):
        cat = os.path.basename(file)[:-4]
        catdf = pd.read_csv(file)
        self.cats[cat] = catdf
        self.datasets[cat] = {'person_word': [], 'original': []}
        for phrase in catdf['phrase']:
            self.datasets[cat][phrase] = []
            
            
    def cats_from_list(self, catlist):
        for file in catlist:
            if file[-4:] == '.csv':
                self.mkcat(file)
                
    def cats_from_dir(self, catdir):
        for file in os.listdir(catdir):
            if file[-4:] == '.csv':
                self.mkcat(os.path.join(catdir, file))
        
    def mkex(self, sent, cat, regex):
        match = re.search(regex, sent)
        assert match, f'Error: Regex failed to match sentence {sent}'
        personword = match[0]
        self.datasets[cat]['person_word'].append(personword.lower())
        self.datasets[cat]['original'].append(sent)
        for i in self.cats[cat].index:
            phrase = self.cats[cat].iloc[i]['phrase']
            substitute = self.cats[cat].iloc[i][personword.lower()]
            if substitute == '-':
                self.datasets[cat][phrase].append(None)
            else:
                if match[0][0].isupper():
                    substitute = substitute[0].upper() + substitute[1:]
                self.datasets[cat][phrase].append(substitute.join(sent.split(match[0])))
        
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create MARB-style dataset from text files.'
    )
        
    parser.add_argument("input", type=str, help="Original sequences to turn into dataset. Can be a single file or a directory of files.")
    parser.add_argument("categories", type=str, help="CSV file or directory of CSV files containing terms to add to sequences. One file per category of terms. The name of the file will be the name of the category in the dataset.")
    parser.add_argument("outputdir", type=str, default="./data/", help="The directory where the finished dataset will be stored. Default: './data/'")
    parser.add_argument("-re", "--regex", dest='regex', type=str, default='auto', help="Regex to use for searching and replacing. 'auto' will infer regex from person words in category file. 'firstline' will treat the first line of each textfile as the search string for that file. any other string will be treated as the regex to use. Default: 'auto'.")
    
    args = parser.parse_args()
    
    dm = DatasetMaker(args.input, categories=args.categories, regex=args.regex)
    dm.make_examples()
    dm.save_datasets(savedir=args.outputdir)
        
        
        
        
        
        
