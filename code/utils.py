import pandas as pd
import numpy as np
import torch
import difflib
import sys
from pathlib import Path
import os
from tqdm import tqdm
from IPython import embed
import seaborn
from sklearn.model_selection import train_test_split
import csv



def pppl(log_probs):
    """
    Pseuso-perplexity PPPL (Salazar et al. 2020) for a corpus of sentences given a list of lists of pseudo-log-likelihoods
    (PLLs). Each of the inner lists shoud correspond to one sentence, where the items are the conditional log likelihoods
    for each sentence token. log_probs can be a singleton list.
    """
    N = sum([len(sent) for sent in log_probs])
    if N == 0:
        return None
    PPPL = np.exp2(-(1/N)*sum([sum(sent) for sent in log_probs]))
    return PPPL


def my_get_span(orig, seq):
    """
    (Modified from https://github.com/katyfelkner/winoqueer/blob/main/code/metric.py)
    This function extract spans that are shared between two sequences.
    Use once for every test sentence together with original sentence to 
    get mask template for test sentence.
    """

    orig = [str(x) for x in orig.tolist()]
    seq = [str(x) for x in seq.tolist()]

    matcher = difflib.SequenceMatcher(None, orig, seq)
    template = []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template += [x for x in range(op[3], op[4], 1)]

    return template


def mask_unmodified(sentence, original, lm):
    """
    (Modified from https://github.com/katyfelkner/winoqueer/blob/main/code/metric.py)
    Score each sentence by masking one (unmodified) token at a time.
    The score for a sentence is the sum of log probability of each unmodified token in
    the sentence.
    """
    if type(sentence) != str:
        return []
        
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if uncased:
        sentence = sentence.lower()
        original = original.lower()

    # tokenize
    sent_token_ids = tokenizer.encode(sentence, return_tensors='pt')
    orig_token_ids = tokenizer.encode(original, return_tensors='pt')

    # get spans of non-changing tokens
    template = my_get_span(orig_token_ids[0], sent_token_ids[0])

    N = len(template)  # num. of tokens that can be masked
    mask_id = tokenizer.mask_token_id
    
    sent_log_probs = []

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent_masked_token_ids = sent_token_ids.clone().detach()

        sent_masked_token_ids[0][template[i]] = mask_id

        score = get_log_prob_unigram(sent_masked_token_ids, sent_token_ids, template[i], lm)

        sent_log_probs.append(score.item())

    return sent_log_probs


def mask_all_unigrams(sentence, original, lm):
    """
    Score each sentence by masking one token at a time.
    Returns a list containing the log probability of each token in
    the sentence (both modified and unmodified).
    """
    if type(sentence) != str:
        return []
        
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if uncased:
        sentence = sentence.lower()
        original = original.lower()

    sent_token_ids = tokenizer.encode(sentence, return_tensors='pt')
    N = len(sent_token_ids[0])  # num. of tokens that can be masked
    mask_id = tokenizer.mask_token_id
    
    sent_log_probs = []

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent_masked_token_ids = sent_token_ids.clone().detach()

        sent_masked_token_ids[0][i] = mask_id

        score = get_log_prob_unigram(sent_masked_token_ids, sent_token_ids, i, lm)

        sent_log_probs.append(score.item())

        

    return sent_log_probs    

def batch_mask_all_unigrams(sentence, original, lm):
    """
    Score each sentence by masking one token at a time.
    Returns a list containing the log probability of each token in
    the sentence (both modified and unmodified).
    """
    if type(sentence) != str:
        return []

    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["log_softmax"]
    device = lm['device']
    
    sent_token_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze()
    N = len(sent_token_ids)  # num. of tokens that can be masked
    mask_id = tokenizer.mask_token_id
    
    sent_log_probs = []
    masked_sents = []

    # skipping CLS and SEP tokens, they'll never be masked   
    for i in range(1, N-1):
        sent_masked_token_ids = sent_token_ids.clone().detach()

        sent_masked_token_ids[i] = mask_id

        masked_sents.append(sent_masked_token_ids)

    logits = model(torch.stack(masked_sents).to(device)).logits
    log_probs = softmax(logits)

    # log_probs[i] is the sentence, log_probs[i][i+1] is the masked token, sent_token_ids[i+1] is the id of the masked token
    # (the i+1 and N-2 are for skipping the CLS and SEP tokens)
    sent_log_probs = [log_probs[i][i+1][sent_token_ids[i+1]].item() for i in range(N-2)]        

    return sent_log_probs

def batch_mask_unmodified(sentence, original, lm):
    """
    (Modified from https://github.com/katyfelkner/winoqueer/blob/main/code/metric.py)
    Score each sentence by masking one (unmodified) token at a time.
    Returns a list containing the log probability of each unmodified token in
    the sentence.
    """
    if type(sentence) != str:
        return []
        
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["log_softmax"]
    device = lm['device']

    #if uncased:
    #    sentence = sentence.lower()
    #    original = original.lower()

    # tokenize
    sent_token_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze()
    orig_token_ids = tokenizer.encode(original, return_tensors='pt').squeeze()

    # get spans of non-changing tokens
    template = my_get_span(orig_token_ids, sent_token_ids)

    N = len(template)  # num. of tokens that can be masked
    mask_id = tokenizer.mask_token_id
    
    sent_log_probs = []
    masked_sents = []

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent_masked_token_ids = sent_token_ids.clone().detach()

        sent_masked_token_ids[template[i]] = mask_id

        masked_sents.append(sent_masked_token_ids)

    logits = model(torch.stack(masked_sents).to(device)).logits
    log_probs = softmax(logits)

    # log_probs[i] is the sentence, log_probs[i][i+1] is the masked token, sent_token_ids[i+1] is the id of the masked token
    # (the i+1 and N-2 are for skipping the CLS and SEP tokens)
    sent_log_probs = [log_probs[i][template[i+1]][sent_token_ids[template[i+1]]].item() for i in range(N-2)]    

    return sent_log_probs


def get_log_prob_unigram(masked_token_ids, token_ids, mask_position, lm): # Changed so we can use the same softmax as for batched version
    """
    (Adapted from https://github.com/katyfelkner/winoqueer/blob/main/code/metric.py)
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    device = lm['device']

    # get model hidden states
    logits = model(masked_token_ids.to(device)).logits
    log_probs = log_softmax(logits)
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_position] == mask_id

    target_id = token_ids[0][mask_position]
    log_prob = log_probs[0][mask_position][target_id]

    return log_prob

def try_batch_masking(batched_function, unbatched_function, sent, original, lm):  # sent = data[cat]
    try:
        score = batched_function(sent, original, lm)
        return score
    except RuntimeError as e:  
        if 'cuda out of memory' in str(e).lower():
            print('Input too big, using batch size 1')
            torch.cuda.empty_cache() 
            score = unbatched_function(sent, original, lm)
            return score
        else:
            sys.exit(e)


def evaluate_masked(args):
    """
    (Modified from https://github.com/katyfelkner/winoqueer/blob/main/code/metric.py).
    Evaluate a masked language model using MARB dataset.
    """

    print(f"Evaluating {os.path.basename(args['model'].name_or_path)} on {args['input']}")

    if args['metric'] not in ['PPPL', 'PLL', 'PPPL-corpus']:
        sys.exit(f'Invalid metric {args["metric"]}')

    print('Reading data...')
    if args['n_ex']:
        print('Preparing examples...')
        full_df = pd.read_csv(args['input'])
        # For fully reproducible results:
        df_data = full_df.groupby('person_word', group_keys=False).apply(lambda x: x[:n]).reset_index(drop=True)
        # Alternative with random sampling stratified by person-word:
        #df_data, _ = train_test_split(full_df, train_size=args['n_ex'], stratify=full_df['person_word']) 
        #df_data.reset_index(drop=True)
    else:
        df_data = pd.read_csv(args['input'])
        args['n_ex'] = 'all'
        
    outdir = args['outdir']
    os.makedirs(outdir, exist_ok=True)

    model = args['model']
    tokenizer = args['tokenizer']
    
    outfile = '_'.join([os.path.basename(model.name_or_path), 
                        args['metric'], args['metric_type'], 
                        str(args['n_ex'])+'-ex', 
                        os.path.basename(args['input'])])

    scorefile = os.path.join(outdir, outfile)

    mask_function = mask_unmodified if args['metric'] == 'context-only' else mask_all_unigrams
    batch_mask_function = batch_mask_unmodified if args['metric'] == 'context-only' else batch_mask_all_unigrams 

    print('Evaluating...')
    model.eval()
    
    device = torch.device(args['device'])
    model.to(device)

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=2)
    vocab = tokenizer.get_vocab()

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": True, 
          "device": device
    }

    # score each sentence. score frames has all data columns (original included for comparison, person_word included for later aggregation)
    # (alternative without original and person_word: df_score = pd.DataFrame(columns=list(df_data.columns[2:]))  )
    #scores = {cat: [] for cat in df_data.columns}

    if args['metric'] == 'PPPL-corpus':
        columns = {cat: [] for cat in df_data.columns[1:]}
    else:
        columns = list(df_data.columns)
        with open(scorefile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)


    total = len(df_data.index)
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for index, data in df_data.iterrows():
                original = data['original']
                if args['metric'] == 'PPPL-corpus':
                    for col in columns:
                        score = try_batch_masking(batch_mask_function, mask_function, data[col], original, lm)
                        columns[col].append(score)
                else:
                    scores = [data['person_word']]
                    for col in columns[1:]:
                        score = try_batch_masking(batch_mask_function, mask_function, data[col], original, lm)
                        if args['metric'] == 'PPPL':
                            scores.append(pppl([score]))
                        else:
                            scores.append(sum(score))
                    with open(scorefile, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(scores)
                torch.cuda.empty_cache() 
                pbar.update(1)
    
    if args['metric'] == 'PPPL-corpus':
        scores = {cat: [pppl(columns[cat])] for cat in columns if cat != 'person_word'}
        df_score = pd.DataFrame(scores)
        df_score.to_csv(scorefile, index=False)
    else:
        df_score = pd.read_csv(scorefile)

    print(f"Output written to: {scorefile}")

    return df_score


def evaluate_autoregressive(args):
    """
    Evaluate an autoregressive language model using MARB dataset.
    """

    print(f"Evaluating {os.path.basename(args['model'].name_or_path)} on {args['input']}")

    if args['metric'] not in ['PPL', 'LL', 'PPL-corpus']:
        sys.exit(f'Invalid metric for autoregressive model: {args["metric"]}')

    print('Reading data...')
    if args['n_ex']:
        print('Preparing examples...')
        full_df = pd.read_csv(args['input'])
        # For fully reproducible results:
        df_data = full_df.groupby('person_word', group_keys=False).apply(lambda x: x[:n]).reset_index(drop=True)
        # Alternative with random sampling:
        #df_data, _ = train_test_split(full_df, train_size=args['n_ex'], stratify=full_df['person_word'])
        #df_data.reset_index(drop=True)
    else:
        df_data = pd.read_csv(args['input'])
        args['n_ex'] = 'all'

    outdir = args['outdir']
    os.makedirs(outdir, exist_ok=True)

    model = args['model']
    tokenizer = args['tokenizer']

    print('Evaluating...')
    model.eval()
    
    device = torch.device(args['device'])
    model.to(device)


    # score each sentence. score frames has all data columns (original included for comparison, person_word included for later aggregation)
    # (alternative without original and person_word: df_score = pd.DataFrame(columns=list(df_data.columns[2:]))  )
    scores = {cat: [] for cat in df_data.columns}


    total = len(df_data.index)
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for index, data in df_data.iterrows():
                for cat in scores:        ## if data[cat] == float('nan'): pass??? otherwise its gonna act weird when it gets to queerness
                    if cat == 'person_word':  # save info about person word
                        scores[cat].append(data[cat])
                    elif type(data[cat]) != str:
                        scores[cat].append(float('nan'))
                    else:
                        inputs = tokenizer(data[cat], return_tensors="pt").to(device)
                        nll = float(model(**inputs, labels=inputs["input_ids"]).loss)
                        if args['metric'] == 'PPL':
                            scores[cat].append(np.exp2(nll))
                        elif args['metric'] == 'LL':
                            scores[cat].append(-nll)
                        else:  # (metric is ppl for whole corpus)
                            scores[cat].append(nll)
                torch.cuda.empty_cache() 
                pbar.update(1)
    
    if args['metric'] == 'PPL-corpus':
        scores = {cat: [np.nanmean(scores[cat])] for cat in scores if cat != 'person_word'}

    df_score = pd.DataFrame(scores)
    outfile = '_'.join([os.path.basename(model.name_or_path), 
                        args['metric'],
                        str(args['n_ex'])+'-ex', 
                        os.path.basename(args['input'])])
    df_score.to_csv(os.path.join(outdir, outfile), index=False)

    print(f"Output written to: {os.path.join(outdir, outfile)}")

    return df_score
