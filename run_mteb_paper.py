# Importações necessárias (sem mudanças)
import mteb
from mteb import MTEB
from transformers import (
    AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer, 
    BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel
)
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import logging
import os
import functions_code # Assumindo que você tem este arquivo
import functions_code_testes # Assumindo que você tem este arquivo
from nltk.corpus import stopwords
import json
import time
import subprocess
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ---
# 1. CLASSE MTEBEncoder (SEM MUDANÇAS)
#    (Sua classe MTEBEncoder, _apply_pooling, _simple_pooling, etc.
#     permanecem exatamente como estão)
# ---

class MTEBEncoder:
    """
    Sua classe MTEBEncoder (SEM MUDANÇAS)
    ... (todo o seu código de __init__, encode, _apply_pooling, etc. vai aqui) ...
    """
    def __init__(self, model_name: str, pooling_strategy: str, device: str):
        self.device = device
        self.pooling_strategy = pooling_strategy
        self.size_embedding = None # Será preenchido após o primeiro encode
        self.print_best_layers = None # Será preenchido
        
        # --- Lógica de Carregamento do Modelo (copiada da sua classe) ---
        if model_name == 'bert-base' or model_name == 'bert-large':
            if model_name == 'bert-base':
                self.name_model = 'google-bert/bert-base-uncased'
                self.qtd_layers = 12
            if model_name == 'bert-large':
                self.name_model = 'google-bert/bert-large-uncased'
                self.qtd_layers = 24
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            ModelClass = BertModel
        
        elif model_name == 'roberta-base' or model_name == 'roberta-large':
            if model_name == 'roberta-base':
                self.name_model = 'FacebookAI/roberta-base'
                self.qtd_layers = 12
            if model_name == 'roberta-large':
                self.name_model = 'FacebookAI/roberta-large'
                self.qtd_layers = 24
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            ModelClass = RobertaModel

        elif model_name == 'deberta-base' or model_name == 'deberta-large':
            if model_name == 'deberta-base':
                self.name_model = 'microsoft/deberta-v3-base'
                self.qtd_layers = 12
            if model_name == 'deberta-large':
                self.name_model = 'microsoft/deberta-v3-large'
                self.qtd_layers = 24
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            ModelClass = DebertaV2Model

        elif model_name == 'angle-base' or model_name == 'angle-large':
            if model_name == 'angle-base':
                self.name_model = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
                self.qtd_layers = 12
            if model_name == 'angle-large':
                self.name_model = 'WhereIsAI/UAE-Large-V1'
                self.qtd_layers = 24
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            ModelClass = AutoModel

        elif model_name == 'allmpnet':
            self.name_model = 'sentence-transformers/all-mpnet-base-v2'
            self.qtd_layers = 12
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            ModelClass = AutoModel
        
        else:
            raise ValueError(f"Nome de modelo desconhecido: {model_name}")

        # Tenta carregar com Flash Attention 2
        try:
            model_args = {'output_hidden_states': True, 'attn_implementation': "flash_attention_2"}
            if "angle" in model_name or "allmpnet" in model_name: # Modelos que precisam de trust_remote_code
                 model_args['trust_remote_code'] = True
                 
            self.model = ModelClass.from_pretrained(self.name_model, **model_args).to(self.device)
            print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
        except (ValueError, ImportError, TypeError) as e:
            print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
            model_args = {'output_hidden_states': True}
            if "angle" in model_name or "allmpnet" in model_name:
                 model_args['trust_remote_code'] = True
            self.model = ModelClass.from_pretrained(self.name_model, **model_args).to(self.device)
        
        self.model.eval() # Modo de avaliação

        # Tenta compilar
        try:
            self.model = torch.compile(self.model)
            print("Modelo compilado com torch.compile() para maior performance.")
        except Exception:
            print("torch.compile() não disponível. Rodando sem compilação.")

        self._prepare_special_token_ids()

    # --- MÉTODO PRINCIPAL DO MTEB ---
    def encode(self, sentences: list[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            tokens = self.tokenizer(
                batch_sentences,
                padding="longest", # Padding dinâmico por lote
                truncation=True,
                return_tensors="pt",
                max_length=512 # Limite razoável
            ).to(self.device)

            with torch.inference_mode(), torch.amp.autocast('cuda'):
                output = self.model(**tokens)
                embeddings = self._apply_pooling(
                    output, 
                    tokens['attention_mask'], 
                    tokens['input_ids']
                )

            all_embeddings.append(embeddings.cpu())

        del tokens, output, embeddings
        torch.cuda.empty_cache()

        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        if self.size_embedding is None:
            self.size_embedding = final_embeddings.shape

        return final_embeddings.numpy()

    # ---
    # TODAS AS SUAS FUNÇÕES DE POOLING (COPIADAS 1-PARA-1)
    # ---
    def _prepare_special_token_ids(self):
        stopwords_list = stopwords.words('english')
        stopword_ids = self.tokenizer.convert_tokens_to_ids(stopwords_list)
        stopword_ids_filtered = [id for id in stopword_ids if id != self.tokenizer.unk_token_id]
        self.stopwords_set_ids = torch.tensor(stopword_ids_filtered, device=self.device)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def _create_combined_mask(self, input_ids, attention_mask, exclude_stopwords=False, exclude_cls_sep=False):
        combined_mask = attention_mask.clone()
        if exclude_stopwords:
            stopword_mask = torch.isin(input_ids, self.stopwords_set_ids, invert=True)
            combined_mask = combined_mask * stopword_mask
        if exclude_cls_sep:
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
    def _sum_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)
    def _max_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e9)
        return masked_embeddings.max(dim=1)[0]
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling, input_ids):
        match name_pooling:
            case "CLS": return hidden_state[:, 0, :]
            case "AVG": return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            case "SUM": return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            case "MAX": return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]
            case "AVG-NS": return self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "SUM-NS": return self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            case "MAX-NS": return self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
    def _pooling_optimized(self, hidden_state, attention_mask, name_pooling, input_ids, optimized_poolings_normal, optimized_poolings_ns):
        if name_pooling in optimized_poolings_normal:
            sum_vector = (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            avg_vector = (sum_vector / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            match name_pooling:
                case 'AVG+SUM': return torch.cat((avg_vector, sum_vector),dim=1)
                case 'CLS+AVG+SUM': return torch.cat((hidden_state[:, 0, :], avg_vector, sum_vector),dim=1)
                case 'AVG+SUM+MAX': return torch.cat((avg_vector, sum_vector, torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]),dim=1)
                case 'AVG+SUM+AVG-NS': return torch.cat((avg_vector, sum_vector, self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
                case 'AVG+SUM+SUM-NS': return torch.cat((avg_vector, sum_vector, self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
                case 'AVG+SUM+MAX-NS': return torch.cat((avg_vector, sum_vector, self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
        elif name_pooling in optimized_poolings_ns:
            mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
            expanded_mask = mask.unsqueeze(-1)
            sum_ns = (hidden_state * expanded_mask).sum(dim=1)
            valid_token_counts = expanded_mask.sum(dim=1)
            avg_ns = sum_ns / valid_token_counts.clamp(min=1e-9)
            match name_pooling:
                case 'AVG-NS+SUM-NS': return torch.cat((avg_ns, sum_ns),dim=1)
                case 'CLS+AVG-NS+SUM-NS': return torch.cat((hidden_state[:, 0, :], avg_ns, sum_ns),dim=1)
                case 'AVG+AVG-NS+SUM-NS': return torch.cat((((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)), avg_ns, sum_ns),dim=1)
                case 'SUM+AVG-NS+SUM-NS': return torch.cat(((hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1), avg_ns, sum_ns),dim=1)
                case 'MAX+AVG-NS+SUM-NS': return torch.cat((torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0], avg_ns, sum_ns),dim=1)
                case 'AVG-NS+SUM-NS+MAX-NS': return torch.cat((avg_ns, sum_ns, self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg, input_ids):
        self.print_best_layers = "NORMAL"
        optimized_poolings_normal = ['AVG+SUM', 'CLS+AVG+SUM', 'AVG+SUM+MAX', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+SUM+MAX-NS']
        optimized_poolings_ns = ['AVG-NS+SUM-NS', 'CLS+AVG-NS+SUM-NS', 'AVG+AVG-NS+SUM-NS', 'SUM+AVG-NS+SUM-NS', 'MAX+AVG-NS+SUM-NS', 'AVG-NS+SUM-NS+MAX-NS']
        total_optimized_poolings = optimized_poolings_normal + optimized_poolings_ns
        if name_pooling in total_optimized_poolings:
            return self._pooling_optimized(hidden_state, attention_mask, name_pooling, input_ids, optimized_poolings_normal, optimized_poolings_ns)
        else:
            name_pooling_split = name_pooling.split('+')
            match len(name_pooling_split):
                case 1: return self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids)
                case 2: return torch.cat((self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids)), dim=1)
                case 3: return torch.cat((self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids)), dim=1)
                case 4: return torch.cat((self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids), self._simple_pooling(hidden_state, attention_mask, name_pooling_split[3], input_ids)), dim=1)
    def get_best_pooling(self, hidden_states, attention_mask, name_pooling):
        match name_pooling:
            case "AVG7+AVG10":
                avg7 = ((hidden_states[7] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); return torch.cat((avg7, avg10), dim=1)
            case "AVG7+AVG10+AVG9":
                avg7 = ((hidden_states[7] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); return torch.cat((avg7, avg10, avg9), dim=1)
            case "AVG9+AVG10":
                avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); return torch.cat((avg9, avg10), dim=1)
            case "AVG9+AVG10+AVG11":
                avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); avg11 = ((hidden_states[11] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)); return torch.cat((avg9, avg10, avg11), dim=1)
            case "AVGdoSUM7e10":
                SUM_7e10 = torch.stack([hidden_states[i] for i in [7,10]], dim=0).sum(dim=0); return ((SUM_7e10 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            case "AVGdoSUM7e10e9":
                SUM_7e10e9 = torch.stack([hidden_states[i] for i in [7,10,9]], dim=0).sum(dim=0); return ((SUM_7e10e9 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            case "AVGdoSUM9e10":
                SUM_9e10 = torch.stack([hidden_states[i] for i in [9,10]], dim=0).sum(dim=0); return ((SUM_9e10 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            case "AVGdoSUM9e10e11":
                SUM_9e10e11 = torch.stack([hidden_states[i] for i in [9,10,11]], dim=0).sum(dim=0); return ((SUM_9e10e11 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
    def _apply_pooling(self, output, attention_mask, input_ids):
        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0]
        name_agg = self.pooling_strategy.split("_")[-1]
        if name_agg == 'BEST':
            self.print_best_layers = "BEST"; return self.get_best_pooling(hidden_states, attention_mask, name_pooling)
        if name_agg.startswith("LYR"):
            layer_idx = int(name_agg.split('-')[-1]); LYR_hidden = hidden_states[layer_idx]; return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling, "LYR", input_ids)
        else:
            name_agg_type = name_agg.split("-")[0]; agg_initial_layer = int(name_agg.split("-")[1]); agg_final_layer = int(name_agg.split("-")[2])
            match name_agg_type:
                case "SUM": agg_hidden = torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).sum(dim=0); return self._get_pooling_result(agg_hidden, attention_mask, name_pooling, name_agg, input_ids)
                case "AVG": agg_hidden = torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).mean(dim=0); return self._get_pooling_result(agg_hidden, attention_mask, name_pooling, name_agg, input_ids)

# ---
# 2. FUNÇÕES DE PARSING DE RESULTADO
# ---

def parse_mteb_results(mteb_results: dict, tasks_list: list) -> dict:
    """
    Parser antigo (para 'cl' e 'si') - (SEM MUDANÇAS)
    """
    flat_results = {}
    for task_name in tasks_list:
        task_result = mteb_results.get(task_name, {})
        if not task_result: flat_results[task_name] = 0; continue
        if 'test' in task_result:
            test_scores = task_result['test']
            lang_key = list(test_scores.keys())[0]
            scores = test_scores[lang_key]
            if 'accuracy' in scores: flat_results[task_name] = scores['accuracy'] * 100
            elif 'cos_sim' in scores and 'spearman' in scores['cos_sim']: flat_results[task_name] = scores['cos_sim']['spearman'] * 100
            elif 'spearman' in scores: flat_results[task_name] = scores['spearman'] * 100
            else:
                try:
                    first_metric = list(scores.values())[0]
                    if isinstance(first_metric, (int, float)): flat_results[task_name] = first_metric * 100
                    else: flat_results[task_name] = 0
                except: flat_results[task_name] = 0
        else: flat_results[task_name] = 0
    return flat_results


# --- NOVO ---
def parse_full_mteb_results(run_results: dict) -> dict:
    """
    O NOVO PARSER DO LEADERBOARD.
    Achata o JSON de resultados em um dicionário de 1 nível com
    TODAS as pontuações de categoria e tarefas individuais.
    """
    flat_scores = {}
    
    # 1. Pega o score médio geral
    if 'average_score' in run_results:
        flat_scores['MTEB_AVG_SCORE'] = run_results['average_score']
    
    # 2. Itera nas categorias (retrieval, classification, etc.)
    for category_name, category_data in run_results.items():
        if not isinstance(category_data, dict):
            continue # Pula chaves simples como 'average_score'

        # 3. Pega o score médio da CATEGORIA
        if 'average_score' in category_data:
            flat_scores[f'CAT_AVG_{category_name.upper()}'] = category_data['average_score']
        
        # 4. Itera nas TAREFAS dentro da categoria
        for task_name, task_data in category_data.items():
            if not isinstance(task_data, dict):
                continue # Pula chaves simples como 'average_score'
            
            # 5. Pega o score principal da TAREFA INDIVIDUAL
            if 'main_score' in task_data:
                # 'main_score' é a métrica oficial (ex: accuracy, ndcg, spearman)
                # O MTEB já calcula em 0-100, então não precisa multiplicar
                flat_scores[task_name] = task_data['main_score']
                
    return flat_scores
# --- FIM DO NOVO ---


# ---
# 3. FUNÇÃO DE EXECUÇÃO PRINCIPAL (MODIFICADA)
# ---

# --- MODIFICADO ---
def run_mteb_evaluation(model_name: str, tasks: list, args: argparse.Namespace, type_task: str):
    """
    Função de avaliação principal, agora usa o parser correto.
    """
    results_general = {}
    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")
    
    try:
        temp_encoder = MTEBEncoder(model_name, 'CLS_LYR-12', device)
        qtd_layers = temp_encoder.qtd_layers
        del temp_encoder
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Erro ao criar encoder temporário para {model_name}: {e}")
        return {}

    if args.agg_layers[0] == 'BEST':
        pooling_strategies = ['AVG7+AVG10_BEST', 'AVG7+AVG10+AVG9_BEST', 'AVG9+AVG10_BEST', 'AVG9+AVG10+AVG11_BEST',
                              'AVGdoSUM7e10_BEST', 'AVGdoSUM7e10e9_BEST', 'AVGdoSUM9e10_BEST', 'AVGdoSUM9e10e11_BEST']
    else:
        pooling_strategies, _, _ = functions_code.strategies_pooling_list(args, qtd_layers)

    print("LISTA DE POOLINGS: ", pooling_strategies)

    for pooling in pooling_strategies:
        print(f"\nRunning: Model={model_name}, Pooling={pooling}")
        
        encoder = MTEBEncoder(model_name, pooling, device)

        if type_task == 'full':
            print("Configurando MTEB para benchmark COMPLETO (leaderboard).")
            evaluation = MTEB(task_langs=["en"]) 
        else:
            print(f"Configurando MTEB para tarefas específicas: {tasks}")
            evaluation = MTEB(tasks=tasks, task_langs=["en"])

        start_time = time.time()
        
        run_results = evaluation.run(
            encoder,
            output_folder=f"mteb_results/{args.save_dir}/{model_name}/{pooling}",
            eval_splits=["test"],
            verbosity=1 
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        # --- AQUI ESTÁ A MUDANÇA ---
        flat_results = {}
        if type_task == 'full':
            # Para o run completo, chama o NOVO PARSER
            if 'average_score' in run_results:
                print(f"##########################################################")
                print(f"+++ SCORE MÉDIO GERAL (LEADERBOARD): {run_results['average_score']:.4f} +++")
                print(f"##########################################################")
                
                # CHAMA O NOVO PARSER DO LEADERBOARD
                flat_results = parse_full_mteb_results(run_results)
            else:
                print("ERRO: 'average_score' não encontrado nos resultados.")
                flat_results = {'MTEB_AVG_SCORE': 0}
        else:
            # Processamento antigo para 'cl' e 'si'
            flat_results = parse_mteb_results(run_results, tasks)
        # --- FIM DA MUDANÇA ---
        
        # Adiciona seus metadados
        flat_results['out_vec_size'] = encoder.size_embedding
        flat_results['qtd_layers'] = encoder.qtd_layers
        flat_results['best_layers'] = encoder.print_best_layers
        
        results_general[pooling] = flat_results # 'flat_results' agora é o dicionário completo
        
        print(f"Output vector size: {encoder.size_embedding}")
        print(f"BEST LAYERS: {encoder.print_best_layers}")
        print(f"--> Time for this run: {elapsed_time:.2f} minutes")
        print("PROGRESS: " + str(pooling_strategies.index(pooling)+1) + '/' + str(len(pooling_strategies)))
        
        del encoder, evaluation, run_results
        torch.cuda.empty_cache()
            
    return results_general
# --- FIM DA MODIFICAÇÃO ---


# ---
# 4. FUNÇÃO DE GERENCIAMENTO DE TAREFAS (MODIFICADA)
# ---

# --- MODIFICADO ---
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
        
        results = run_mteb_evaluation(model_name, tasks_list, args, type_task)
        
        for pooling, res in results.items():
            
            # --- AQUI ESTÁ A MUDANÇA ---
            # 'res' é o dicionário achatado retornado pelo parser
            
            # Pega os metadados antes de desempacotar
            base_data = {
                "model": model_name,
                "pooling": pooling,
                "out_vec_size": res.pop('out_vec_size', None), # .pop para não duplicar
                "best_layers": res.pop('best_layers', None),
                "qtd_layers": res.pop('qtd_layers', None),
            }

            if type_task == 'full':
                # 'res' já contém MTEB_AVG_SCORE, CAT_AVGs, e todas as tarefas
                base_data.update(res) 
            else:
                # Lógica antiga para 'cl' e 'si'
                base_data.update({
                    "epochs": args.epochs, 
                    "nhid": args.nhid, 
                })
                task_scores = {task: res.get(task, 0) for task in tasks_list}
                base_data.update(task_scores)
            
            results_data.append(base_data)
            # --- FIM DA MUDANÇA ---
        
        final_df1 = pd.DataFrame(results_data)
        final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
            
    final_df = pd.DataFrame(results_data)
    
    # --- NOVO: Reordena as colunas para ficar parecido com o Leaderboard ---
    if type_task == 'full' and not final_df.empty:
        cols = final_df.columns.tolist()
        # Colunas principais primeiro
        ordered_cols = ['model', 'pooling', 'MTEB_AVG_SCORE', 'out_vec_size', 'best_layers', 'qtd_layers']
        
        # Colunas de Categoria
        cat_cols = sorted([c for c in cols if c.startswith('CAT_AVG_')])
        
        # Colunas de Tarefas (o resto)
        task_cols = sorted([c for c in cols if c not in ordered_cols and c not in cat_cols])
        
        final_df = final_df[ordered_cols + cat_cols + task_cols]
    # --- Fim do Novo Bloco de Ordenação ---

    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)

    if os.path.exists(path_created + '/' + filename_task + '_intermediate.csv'):
        os.remove(path_created + '/' + filename_task + '_intermediate.csv')

    try:
        if type_task != 'full':
            functions_code_testes.main_evaluate(final_df, type_task, path_created, filename_task, tasks_list)
    except Exception as e:
        print(f"Não foi possível rodar a avaliação final (main_evaluate): {e}")
# --- FIM DA MODIFICAÇÃO ---


# ---
# 5. FUNÇÃO MAIN (SEM MUDANÇAS)
# ---

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
                     '_initiallayer_' + str(initial_layer_args_print) +
                     '_finallayer_' + str(final_layer_args_print) +
                     '_pooling_' + '&'.join([st for st in args.poolings]) +
                     '_agglayers_' + '&'.join([st for st in args.agg_layers])
                    )
    
    if args.run_full_benchmark:
        print("--- INICIANDO EXECUÇÃO DO MTEB COMPLETO (LEADERBOARD) ---")
        print(f"Modelos: {args.models}")
        print(f"Poolings: {args.poolings}")
        print(f"Agg Layers: {args.agg_layers}")
        print("AVISO: Isso pode levar várias horas ou dias, dependendo do modelo.")
        
        filename_full = "FULL_LEADERBOARD" + filename_task
        tasks_run(args, main_path, filename_full, tasks_list=None, type_task='full')

    elif args.task_type == "classification":
        filename_cl = "cl" + filename_task
        mteb_classification_tasks = [
            "SentimentAnalysis", "AmazonReviewsClassification", "Banking77Classification",
            "EmotionClassification", "MassiveIntentClassification", "ToxicConversationsClassification",
            "TweetEvalClassification"
        ]
        classification_tasks = args.tasks.split(",") if args.tasks is not None else mteb_classification_tasks
        print(f"Rodando tarefas de CLASSIFICAÇÃO MTEB: {classification_tasks}")
        tasks_run(args, main_path, filename_cl, classification_tasks, 'cl')

    elif args.task_type == "similarity":
        filename_si = "si" + filename_task
        mteb_similarity_tasks = [
            "STS12", "STS13", "STS14", "STS15", "STS16", 
            "STSBenchmark", "SICKRelatedness"
        ]
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else mteb_similarity_tasks
        print(f"Rodando tarefas de SIMILARIDADE MTEB: {similarity_tasks}")
        tasks_run(args, main_path, filename_si, similarity_tasks, 'si')

# ---
# 6. PONTO DE ENTRADA (SEM MUDANÇAS)
# ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTEB Pooling Experiments (Adaptado de SentEval)")

    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, default=4, help="Número máximo de épocas (apenas para referência)")
    parser.add_argument("--batch", type=int, default=64, help="Batch Size (apenas para referência)")
    parser.add_argument("--kfold", type=int, default=10, help="KFold (apenas para referência)")
    parser.add_argument("--optim", type=str, default='adam', help="otimizador (apenas para referência)")
    parser.add_argument("--nhid", type=int, default=0, help="Numero de camadas ocultas (apenas para referência)")
    parser.add_argument("--initial_layer", default=12, type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_gument("--final_layer", type=int, default=12, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, required=True, default="all", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, required=True, default="ALL", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks MTEB separados por virgula (sem espacos)")
    parser.add_argument("--save_dir", type=str, help="Diretório para salvar os resultados")
    
    parser.add_argument("--run_full_benchmark", action="store_true", 
                        help="Rodar o benchmark MTEB completo (todas as 8 categorias) para a pontuação do leaderboard.")

    args = parser.parse_args()
    original_save_dir = args.save_dir

    main(args)