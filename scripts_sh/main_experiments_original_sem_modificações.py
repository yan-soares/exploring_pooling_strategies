import senteval
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel
import torch
import argparse
import pandas as pd
import logging
import os
import functions_code
from nltk.corpus import stopwords
import subprocess
import json
import time

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SentenceEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = None
        self.print_best_layers = None
        
        self.stopwords_set_ids = None
        self.cls_token_id = None
        self.sep_token_id = None

        self.general_embeddings = {}
        self.list_poolings = None
        self.list_layers = None
        self.actual_layer = None

        if model_name == 'bert-base' or  model_name == 'bert-large':
            if model_name == 'bert-base':
                self.name_model = 'google-bert/bert-base-uncased'
                self.qtd_layers = 12
            if model_name == 'bert-large':
                self.name_model = 'google-bert/bert-large-uncased'
                self.qtd_layers = 24
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            try:
                self.model = BertModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = BertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'roberta-base' or  model_name == 'roberta-large':
            if model_name == 'roberta-base':
                self.name_model = 'FacebookAI/roberta-base'
                self.qtd_layers = 12
            if model_name == 'roberta-large':
                self.name_model = 'FacebookAI/roberta-large'
                self.qtd_layers = 24
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            try:
                self.model = RobertaModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = RobertaModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'deberta-base' or model_name == 'deberta-large':
            if model_name == 'deberta-base':
                self.name_model = 'microsoft/deberta-v3-base'
                self.qtd_layers = 12
            if model_name == 'deberta-large':
                self.name_model = 'microsoft/deberta-v3-large'
                self.qtd_layers = 24
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            try:
                self.model = DebertaV2Model.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = DebertaV2Model.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'angle-base' or model_name == 'angle-large':       
            if model_name == 'angle-base':
                self.name_model = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
                self.qtd_layers = 12
            if model_name == 'angle-large':
                self.name_model = 'WhereIsAI/UAE-Large-V1'
                self.qtd_layers = 24            
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            try:
                self.model = AutoModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2", trust_remote_code=True
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True, trust_remote_code=True).to(self.device) 

        if model_name == 'allmpnet':
            self.name_model = 'sentence-transformers/all-mpnet-base-v2'
            self.qtd_layers = 12
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            try:
                self.model = AutoModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        try:
            self.model = torch.compile(self.model)
            print("Modelo compilado com torch.compile() para maior performance.")
        except Exception:
            print("torch.compile() não disponível. Rodando sem compilação.")

        self._prepare_special_token_ids()

    def _prepare_special_token_ids(self):
        # Converte a lista de stopwords (strings) para uma lista de IDs de tokens
        stopwords_list = stopwords.words('english')
        stopword_ids = self.tokenizer.convert_tokens_to_ids(stopwords_list)
        
        # Filtra IDs desconhecidos e cria um tensor na GPU para comparações rápidas
        stopword_ids_filtered = [id for id in stopword_ids if id != self.tokenizer.unk_token_id]
        self.stopwords_set_ids = torch.tensor(stopword_ids_filtered, device=self.device)
        
        # Armazena os IDs dos tokens CLS e SEP para fácil acesso
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def _encode(self, sentences, current_task, batch_size=16384): 
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length = self.model.config.max_position_embeddings
        )

        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}
            with torch.no_grad(), torch.amp.autocast('cuda'):                
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'], batch_tokens['input_ids']) 
                del batch_tokens, output
                #torch.cuda.empty_cache()

                all_embeddings.append(embeddings)           

        self.size_embedding = all_embeddings[0].shape 
        final_embeddings = torch.cat(all_embeddings, dim=0).to('cpu').numpy()
        return final_embeddings
    
    def _create_combined_mask(self, input_ids, attention_mask, exclude_stopwords=False, exclude_cls_sep=False):
        """
        OTIMIZAÇÃO: Cria uma máscara combinada de forma vetorizada na GPU.
        - `attention_mask`: máscara base para ignorar padding.
        - `exclude_stopwords`: se True, zera a posição de stopwords.
        - `exclude_cls_sep`: se True, zera a posição dos tokens [CLS] e [SEP].
        """
        combined_mask = attention_mask.clone()

        if exclude_stopwords:
            # `torch.isin` é uma operação vetorizada e muito rápida na GPU
            stopword_mask = torch.isin(input_ids, self.stopwords_set_ids, invert=True)
            combined_mask = combined_mask * stopword_mask

        if exclude_cls_sep:
            # Cria uma máscara que é False onde os tokens são CLS ou SEP
            special_tokens_mask = (input_ids != self.cls_token_id) & (input_ids != self.sep_token_id)
            combined_mask = combined_mask * special_tokens_mask
        
        return combined_mask
    
    def _mean_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)

        masked_embeddings = output * expanded_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)
        return sum_embeddings / valid_token_counts.clamp(min=1e-9)

        mask = self._create_combined_mask(input_ids, attention_mask, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)

    def _sum_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)

   
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e9)
        return masked_embeddings.max(dim=1)[0]
    
    def _max_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e9)
        return masked_embeddings.max(dim=1)[0]
    
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling, input_ids):

         match name_pooling:
             
            case "CLS":
                return hidden_state[:, 0, :]
            
            case "AVG":
                return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "SUM":
                return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            
            case "MAX":
                return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]
             
            case "AVG-NS":
                return self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            
            case "SUM-NS":
                return self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            
            case "MAX-NS":
                return self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
        
    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg, input_ids):

        name_pooling_split = name_pooling.split('+')
        self.print_best_layers =  "NORMAL"

        match len(name_pooling_split):

            case 1:
                return self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids)
            
            case 2:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids)
                ), 
                dim=1)
            
            case 3:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids)
                ), 
                dim=1)
            
            case 4:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[3], input_ids)
                ), 
                dim=1)

    def get_best_pooling(self, hidden_state, attention_mask, name_pooling, name_agg):

        if name_agg == 'BEST':
            match self.name_model:
                case 'sentence-transformers/all-mpnet-base-v2':
                    SUM_7_12_hidden = torch.stack(hidden_state[7:13], dim=0).sum(dim=0) 
                    SUM_5_12_hidden = torch.stack(hidden_state[5:13], dim=0).sum(dim=0) 
                    AVG_6_12_hidden = torch.stack(hidden_state[6:13], dim=0).mean(dim=0)
                    AVG_5_12_hidden = torch.stack(hidden_state[5:13], dim=0).mean(dim=0)
                    self.print_best_layers =  "cls_SUM-7-12_avg_SUM-5-12_sum_AVG-6-12_avgns_SUM-5-12_sumns_AVG-5-12"

                    cls_result = SUM_7_12_hidden[:, 0, :]
                    avg_result = ((SUM_5_12_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                    sum_result = (AVG_6_12_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                    avg_ns_result = self._mean_pooling_exclude_cls_sep(SUM_5_12_hidden, attention_mask)
                    sum_ns_result = self._sum_pooling_exclude_cls_sep(AVG_5_12_hidden, attention_mask)
                case 'microsoft/deberta-v3-base':
                    SUM_11_12_hidden = torch.stack(hidden_state[11:13], dim=0).sum(dim=0) 
                    SUM_7_10_hidden = torch.stack(hidden_state[7:11], dim=0).sum(dim=0) 
                    SUM_6_10_hidden = torch.stack(hidden_state[6:11], dim=0).sum(dim=0)
                    AVG_7_11_hidden = torch.stack(hidden_state[7:12], dim=0).mean(dim=0)
                    LYR_9 = hidden_state[9]
                    self.print_best_layers =  "cls_SUM-11-12_avg_SUM-7-10_sum_LYR-9_avgns_SUM-6-10_sumns_AVG-7-11"

                    cls_result = SUM_11_12_hidden[:, 0, :]
                    avg_result = ((SUM_7_10_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                    sum_result = (LYR_9 * attention_mask.unsqueeze(-1)).sum(dim=1)
                    avg_ns_result = self._mean_pooling_exclude_cls_sep(SUM_6_10_hidden, attention_mask)
                    sum_ns_result = self._sum_pooling_exclude_cls_sep(AVG_7_11_hidden, attention_mask)

        match name_pooling:
       
            case "CLS+AVG":
                return torch.cat((cls_result, avg_result), dim=1)
            case "CLS+SUM":
                return torch.cat((cls_result, sum_result), dim=1)
            case "CLS+AVG-NS":
                return torch.cat((cls_result, avg_ns_result), dim=1)
            case "CLS+SUM-NS":
                return torch.cat((cls_result, sum_ns_result), dim=1)
            case "AVG+SUM":
                return torch.cat((avg_result, sum_result), dim=1)
            case "AVG+AVG-NS":
                return torch.cat((avg_result, avg_ns_result), dim=1)
            case "AVG+SUM-NS":
                return torch.cat((avg_result, sum_ns_result), dim=1)
            case "SUM+AVG-NS":
                return torch.cat((sum_result, avg_ns_result), dim=1)
            case "SUM+SUM-NS":
                return torch.cat((sum_result, sum_ns_result), dim=1)
            case "AVG-NS+SUM-NS":
                return torch.cat((avg_ns_result, sum_ns_result), dim=1)

            case "CLS+AVG+SUM":
                return torch.cat((cls_result, avg_result, sum_result), dim=1)
            case "CLS+AVG+AVG-NS":
                return torch.cat((cls_result, avg_result, avg_ns_result), dim=1)
            case "CLS+AVG+SUM-NS":
                return torch.cat((cls_result, avg_result, sum_ns_result), dim=1)
            case "CLS+SUM+AVG-NS":
                return torch.cat((cls_result, sum_result, avg_ns_result), dim=1)
            case "CLS+SUM+SUM-NS":
                return torch.cat((cls_result, sum_result, sum_ns_result), dim=1)
            case "CLS+AVG-NS+SUM-NS":
                return torch.cat((cls_result, avg_ns_result, sum_ns_result), dim=1)
            case "AVG+SUM+AVG-NS":
                return torch.cat((avg_result, sum_result, avg_ns_result), dim=1)
            case "AVG+SUM+SUM-NS":
                return torch.cat((avg_result, sum_result, sum_ns_result), dim=1)
            case "AVG+AVG-NS+SUM-NS":
                return torch.cat((avg_result, avg_ns_result, sum_ns_result), dim=1)
            case "SUM+AVG-NS+SUM-NS":
                return torch.cat((sum_result, avg_ns_result, sum_ns_result), dim=1)
            
    def _apply_pooling(self, output, attention_mask, input_ids):  

        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0]
        name_agg = self.pooling_strategy.split("_")[-1]

        if name_agg == 'BEST':
             return self.get_best_pooling(hidden_states, attention_mask, name_pooling, name_agg, input_ids)

        if name_agg.startswith("LYR"):
            layer_idx = int(name_agg.split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling, "LYR", input_ids)        
        else:        
            name_agg_type = name_agg.split("-")[0]
            agg_initial_layer = int(name_agg.split("-")[1])
            agg_final_layer = int(name_agg.split("-")[2])
            
            match name_agg_type:  

                case "SUM":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).sum(dim=0), attention_mask, name_pooling, name_agg, input_ids)
                    
                case "AVG":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).mean(dim=0), attention_mask, name_pooling, name_agg, input_ids)                  
        
    def _strategies_pooling_list (self, args):
        initial_layer_args = args.initial_layer
        final_layer_args = args.final_layer
        poolings_args = args.poolings
        agg_layers_args = args.agg_layers
        
        #POOLING
        pooling_techniques = functions_code.get_pooling_techniques(poolings_args, agg_layers_args)
        
        #LAYERS
        if initial_layer_args is not None:
            initial_layer = initial_layer_args
        else:
            initial_layer = int(self.qtd_layers / 2)

        if final_layer_args is not None:
            final_layer = final_layer_args
        else:
            final_layer = int(self.qtd_layers)

        list_lyrs = functions_code.get_list_layers(final_layer, initial_layer, agg_layers_args)

        #STRATEGIES CONCAT
        pooling_strategies = []
        for l in list_lyrs:
            for p in pooling_techniques:
                pooling_strategies.append(p + "_" + l) 

        #RETURN
        return pooling_strategies, pooling_techniques, list_lyrs

def run_senteval(model_name, tasks, args, type_task):
    results_general = {}

    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")
    
    encoder = SentenceEncoder(model_name, device)
    pooling_strategies, list_poolings, list_layers = encoder._strategies_pooling_list(args)

    #GET ALL EMBEDDINGS
    print("LISTA DE POOLINGS: ", list_poolings)
    print("LISTA DE LAYERS: ", list_layers)
   
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
        if type_task == 'cl':
            senteval_params = {
                'task_path': 'data',
                'usepytorch': False,
                'kfold': args.kfold,
                'classifier': {
                    'nhid': args.nhid,
                    'optim': args.optim,
                    'batch_size': args.batch,
                    'tenacity': 5,
                    'epoch_size': args.epochs
                },
                'encoder': encoder
            }
        else:
             senteval_params = {
                'task_path': 'data',
                'usepytorch': True,
                'kfold': 10,
                'encoder': encoder
            }
        se = senteval.engine.SE(senteval_params, functions_code.batcher)

        # --- NOVO: Medição de tempo ---
        start_time = time.time()
        results_general[pooling] = se.eval(tasks)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        # --- Fim da medição ---

        results_general[pooling]['out_vec_size'] = encoder.size_embedding
        results_general[pooling]['qtd_layers'] = encoder.qtd_layers
        results_general[pooling]['best_layers'] = encoder.print_best_layers
        print(f"Output vector size: {encoder.size_embedding}")
        print(f"BEST LAYERS: {encoder.print_best_layers}")
        print(f"--> Time for this run: {elapsed_time:.2f} minutes") # NOVO: Imprime o tempo no console
                              
    return results_general

def tasks_run(args, main_path, filename_task, tasks_list, type_task):
    path_created = main_path + '/' + filename_task
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_task + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results_data = []

    config_path = os.path.join(path_created, f"config_{args.save_dir}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    for model_name in args.models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, tasks_list, args, type_task)
        for pooling, res in results.items():
            if type_task == 'cl':
                dict_results = [res.get(task, {}) for task in tasks_list]
            elif type_task == 'si':
                dict_results = [res.get(task, {}).get('all', 0) for task in tasks_list[:5]] + [res.get(task, {}) for task in tasks_list[-2:]]
            
            results_data.append({
                "model": model_name,
                "pooling": pooling,
                "out_vec_size": res.get('out_vec_size'),
                "best_layers": res.get('best_layers'),
                "epochs": args.epochs,
                "nhid": args.nhid,
                "qtd_layers": res.get('qtd_layers'),              
                **{task: dict_results[i] for i, task in enumerate(tasks_list)}
            })
        
        final_df1 = pd.DataFrame(results_data)
        final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
                    
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)

def main(args):

    args.models = args.models.split(",")
    args.poolings = args.poolings.split(",")
    args.agg_layers = args.agg_layers.split(",")  

    main_path = '../results_pooling_paper/' + str(args.save_dir)

    initial_layer_args_print = args.initial_layer if args.initial_layer is not None else "default"
    final_layer_args_print = args.final_layer if args.final_layer is not None else "default"

    filename_task = ('_models_' + '&'.join([st for st in args.models]) + 
                     '_epochs_' + str(args.epochs) + 
                     '_batch_' + str(args.batch) +
                     '_nhid_' + str(args.nhid) + 
                     '_inlayer_' + str(initial_layer_args_print) + 
                     '_filayer_' + str(final_layer_args_print) +
                     '_pooling_' + '&'.join([st for st in args.poolings]) + 
                     '_agglayers_' + '&'.join([st for st in args.agg_layers])
                    )
    
    if args.task_type == "classification":      
        filename_cl = "cl" + filename_task
        classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(args, main_path, filename_cl, classification_tasks, 'cl')

    elif args.task_type == "similarity":
        filename_si = "si" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(args, main_path, filename_si, similarity_tasks, 'si')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentEval Experiments")

    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, default=4, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--batch", type=int, default=64, help="Batch Size do classificador")
    parser.add_argument("--kfold", type=int, default=10, help="KFold para validação")
    parser.add_argument("--optim", type=str, default='adam', help="otimizador do classificador")
    parser.add_argument("--nhid", type=int, default=0, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", default=12, type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--final_layer", type=int, default=12, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, required=True, default="all", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, required=True, default="ALL", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks separados por virgula (sem espacos)")
    parser.add_argument("--save_dir", type=str, help="tasks separados por virgula (sem espacos)")

    args = parser.parse_args()
    original_save_dir = args.save_dir

    main(args)

    logging.info("\nTreinamento concluído. Iniciando avaliação automática...")
    
    '''
    try:
        # Constrói o comando para chamar o script evaluate.py
        command = [
            "python", 
            "evaluate.py", 
            "--task_type", 
            classification
            "--dir", 
            original_save_dir
        ]
        
        # Executa o comando
        subprocess.run(command, check=True)
        
    except FileNotFoundError:
        logging.error("Erro: 'evaluate.py' não encontrado no mesmo diretório. A avaliação não pôde ser executada.")
    except subprocess.CalledProcessError as e:
        logging.error(f"O script 'evaluate.py' falhou com o erro: {e}")
    '''