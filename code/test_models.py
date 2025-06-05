import argparse
import os
import sys
from utils import *
from transformers import logging


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate model on the MARB dataset.',
    )
    parser.add_argument("model", type=str, help="Model to test. Available models are (MLMs:) 'BERT', 'Roberta', 'Albert', (Generative:) 'GPT2', 'Bloom', 'OPT', 'Mistral'.")
    parser.add_argument("inputdir", type=str, default='../data/', help="Path to dataset dir. Default='../data/'.")
    parser.add_argument("outdir", type=str, default='../results/', help="Path to dir for resulting score file. Default='../results/'.")
    parser.add_argument('-l', '--large', dest='large', action='store_true', default=False, help="If this flag is used, the larger version of the model (if one is available) will be tested.")
    parser.add_argument('-c', '--category', dest='cat', type=lambda s: [item.strip() for item in s.split(',')], default=['all'], help="Categories to evaluate, delimited by ','. If not provided, tries to evaluate on all CSV files available in 'inputdir'.")
    parser.add_argument("-d", "--device", dest='device', type=str, default='cuda', help="Device to use (default='cuda').")
    parser.add_argument("-m", "--metric", dest='metric', type=str, default='auto', help="Metric to use ('PPPL'/'PLL'/'PPPL-corpus' for masked models, 'PPL'/'LL'/'PPL-corpus' for autoregressive models). Default='auto' ('PPPL' for masked models, 'PPL' for autoregressive models.)")
    parser.add_argument("-mt", "--metric_type", dest='metric_type', type=str, default='all-tokens', help="Whether to test all tokens ('all-tokens') or only context tokens ('context-only'). Default='all-tokens'.")
    parser.add_argument("-n", "--n_ex", dest='n_ex', type=int, required=False, help="Size of subset to evaluate on. If not provided, evaluates on full dataset.")
    
    args = parser.parse_args()

    logging.set_verbosity_error()

    if args.cat == ['all']:
        cats = [p for p in os.listdir(args.inputdir) if p[-4:] == '.csv']
    else:
        cats = [cat+'.csv' for cat in args.cat]


    if args.model.lower() == 'bert':
        from transformers import AutoTokenizer, BertForMaskedLM
        if args.large:
            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            model = BertForMaskedLM.from_pretrained("bert-large-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        eval_fn = evaluate_masked
            
    elif args.model.lower() == 'roberta':
        from transformers import AutoTokenizer, RobertaForMaskedLM
        if args.large:
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
            model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-large")
        else:
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
            model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
        eval_fn = evaluate_masked

    elif args.model.lower() == 'albert':
        from transformers import AutoTokenizer, AlbertForMaskedLM
        if args.large:
            tokenizer = AutoTokenizer.from_pretrained("albert/albert-large-v2")
            model = AlbertForMaskedLM.from_pretrained("albert/albert-large-v2")
        else:
            tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
            model = AlbertForMaskedLM.from_pretrained("albert/albert-base-v2")
        eval_fn = evaluate_masked

    elif args.model.lower() == 'gpt2':
        from transformers import AutoTokenizer, GPT2LMHeadModel
        if args.large:
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
            model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-medium")
        else:
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        eval_fn = evaluate_autoregressive

    elif args.model.lower() == 'bloom':
        from transformers import AutoTokenizer, BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        eval_fn = evaluate_autoregressive

    elif args.model.lower() == 'opt':
        from transformers import AutoTokenizer, OPTForCausalLM
        model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        eval_fn = evaluate_autoregressive


    elif args.model.lower() == 'mistral':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map = 'auto')
        eval_fn = evaluate_autoregressive
    
    

    else:
        sys.exit(f'Error: Unsupported model: {args.model}')

    

    print(f'Evaluating {model.name_or_path} on {cats}.')

    if args.metric == 'auto':
        if args.model.lower() in ['bert', 'albert', 'roberta']:
            metric = 'PPPL'
        else:
            metric = 'PPL'
    else:
        metric = args.metric
        

    for cat in cats:
        eval_args = {
            'input': os.path.join(args.inputdir, cat),
            'model': model,
            'tokenizer': tokenizer,
            'outdir': args.outdir,
            'device': args.device,
            'metric': metric,  # 'PPPL'/'PLL'/'PPPL-corpus' // 'PPL'/'LL'/'PPL-corpus'
            'metric_type': args.metric_type,  # 'context-only'/'all-tokens' ((only implemented for masked language models))
            'n_ex': args.n_ex  # to test on smaller subset
        }
        
        eval_fn(eval_args)

    print('Done!')



















