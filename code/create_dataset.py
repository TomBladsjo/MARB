import re
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse


class DatasetMaker():
    """
    An object class for creating the MARB dataset from natural language sentences.
    """
    def __init__(self):
        self.re = re.compile(r'(\ban?)\s\b(man|woman|person)\b')  # child|kid|boy|girl??

        czernowska_termdir = '/srv/data/gussodato/thesis/generalized-fairness-metrics/terms/identity_terms'
        disabilitydf = pd.read_csv(os.path.join(czernowska_termdir, 'disability.csv')).drop_duplicates('GROUP')
        self.disabilityterms = list(zip(disabilitydf['GROUP'], disabilitydf['TERM'], disabilitydf['POS']))

        self.disability = {}
        self.race = {}
        self.queerness = {}

        
    def categories(self):
        return {
                'disability': [self.disability, self.mkcat_disability, self.mkex_disability], 
                'race': [self.race, self.mkcat_race, self.mkex_race], 
                'queerness': [self.queerness, self.mkcat_queer, self.mkex_queer]
                          }        


    def make_examples(self, originals, categories='all'):
        """
        Takes a list of sentences and creates datasets corresponding to 'categories'.
        Args:
            originals: A list of strings where each string is an English sentence.
            categories: string or list of strings. Available categories are 'disability', 'race' and 'queerness'. If 'all', will 
            create datasets for all three categories. 
        """
        assert type(categories) == str or type(categories) == list, 'categories must be str or list'
        if categories == 'all':
            categories = self.categories()
        elif type(categories) == str:
            categories = [categories]
        
        print(f'Creating examples for categories:')
        for cat in categories:
            self.categories()[cat][1]()
            print(cat)
        for ex in tqdm(originals):
            match = re.search(self.re, ex)
            if match:
                for cat in categories:
                    self.categories()[cat][2](ex, match)
            else:
                print(f'Failed to match sentence: {ex}')
        print('Done!\n')
    
    
    def save_datasets(self, savedir='/srv/data/gussodato/MARB/'):  # , add_or_replace='add' ??
        """
        Saves created datasets as CSV files. Optionally takes the argument savedir (str), which is the path to the directory 
        where the datasets will be stored. Default savedir is '/srv/data/gussodato/MARB/'.
        """
        if not Path(savedir).is_dir():
            print(f'Creating directory {savedir}')
            os.mkdir(savedir)
        print(f'Saving datasets to {savedir}:')
        
        for cat in self.categories():
            if self.categories()[cat][0]:
                print(f'Saving {cat}.csv...')
                df = pd.DataFrame(self.categories()[cat][0])
                df.to_csv(os.path.join(savedir, f'{cat}.csv'), index=False)
        print('\nDone!')
    
        
    def mkcat_disability(self):
        self.disability = {  # add "seeing", "hearing", "neurotypical"??
             'person_word': [],
            
             'hearing': [],
             'sight': [],
             'unspecified': [],
             'mobility': [],
             'cerebral_palsy': [],
             'mental_health': [],
             'epilepsy': [],
             'physical': [],
             'chronic_illness': [],
             'short_stature': [],
             'cognitive': [],
             'down_syndrome': [],
             'without': []
        }
        
    def mkcat_race(self):
        self.race = {  # removed 'american indian', 'hispanic' bc fewer than native american and latinx in google ngrams. 
                       # also removed alaska_native and native_hawaiian (to keep only one per "group")
                       # removed 'african american' bc US specific. 
             'person_word': [],

             'native_american': [],
             'asian': [],
             'black': [],
             'latinx': [],
             'pacific_islander': [],
             'white': []
        }
        
    def mkcat_queer(self):
        self.queerness = {  # remove heterosexual? remove cis or cisgender, change transgender to trans?
             'person_word': [],
            
             'Asexual': [],
             'Bisexual': [],
             'Cis': [],
             'Cisgender': [],
             'Gay': [],
             'Heterosexual': [],
             'LGBTQ': [],
             'Lesbian': [],
             'NB': [],
             'Pansexual': [],
             'Queer': [],
             'Straight': [],
             'Transgender': []
        }                

    def mkex_disability(self, ex, match):
        dataset = self.disability
        dataset['person_word'].append(match[2])
        spl = ex.split(match[0].strip())
        for (group, term, pos) in self.disabilityterms:
            if pos == 'adj':
                dataset[group].append(('a '+term+' '+match[2]).join(spl))
            else:
                dataset[group].append(('a '+match[2]+' '+term).join(spl))

    
    def mkex_race(self, ex, match):
        dataset = self.race
        dataset['person_word'].append(match[2])
        spl = ex.split(match[0].strip())
        dataset['native_american'].append(('a native american '+match[2]).join(spl))
        dataset['asian'].append(('an asian '+match[2]).join(spl))
        dataset['black'].append(('a black '+match[2]).join(spl))
        if match[2] == 'woman':
            dataset['latinx'].append(('a latina '+match[2]).join(spl))
        elif match[2] == 'man':
            dataset['latinx'].append(('a latino '+match[2]).join(spl))
        else:
            dataset['latinx'].append(('a latinx '+match[2]).join(spl))
        dataset['pacific_islander'].append(('a pacific islander '+match[2]).join(spl))  # this sounds unnatural, will get weird results
        dataset['white'].append(('a white '+match[2]).join(spl))
        

    def mkex_queer(self, ex, match):
        dataset = self.queerness
        dataset['person_word'].append(match[2])
        spl = ex.split(match[0].strip())
        dataset['Asexual'].append(('an asexual '+match[2]).join(spl))
        dataset['Bisexual'].append(('a bisexual '+match[2]).join(spl))
        dataset['Cis'].append(('a cis '+match[2]).join(spl))
        dataset['Cisgender'].append(('a cisgender '+match[2]).join(spl))  # remove this and change transgender to trans? or remove cis? 
        if match[2] == 'man':
            dataset['Gay'].append(('a gay '+match[2]).join(spl))
        else:
            dataset['Gay'].append(None)
        dataset['Heterosexual'].append(('a heterosexual '+match[2]).join(spl))
        dataset['LGBTQ'].append(('an LGBTQ '+match[2]).join(spl))
        if match[2] == 'woman':
            dataset['Lesbian'].append(('a lesbian '+match[2]).join(spl))
        else:
            dataset['Lesbian'].append(None)
        if match[2] == 'person':
            dataset['NB'].append(('a nonbinary '+match[2]).join(spl))
        else:
            dataset['NB'].append(None)
        dataset['Pansexual'].append(('a pansexual '+match[2]).join(spl))
        dataset['Queer'].append(('a queer '+match[2]).join(spl))
        dataset['Straight'].append(('a straight '+match[2]).join(spl))
        dataset['Transgender'].append(('a transgender '+match[2]).join(spl))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for comparing reporting bias between groups.')
    parser.add_argument("corpus", type=str, help="Path to .txt file containing sentences (one per line) from which to create the dataset.")
    parser.add_argument("-o", "--outdir", dest='outdir', type=str, default='/srv/data/gussodato/MARB/', help="Directory in which to save the resulting datasets. Default: '/srv/data/gussodato/MARB/'")
    args = parser.parse_args()

    print(f'Reading from file {args.corpus}...')
    with open(args.corpus, 'r') as f:
        sentences = [line.strip().lower() for line in f]

    print('Creating dataset...')
    dataset_maker = DatasetMaker()
    dataset_maker.make_examples(sentences, categories='all')

    dataset_maker.save_datasets(savedir=args.outdir)




