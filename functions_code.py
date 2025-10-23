import torch
from itertools import combinations
import numpy as np

def get_agg_base():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 13))

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_agg_large():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 25))

    slices = {}
    for size in range(2, 25):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 25):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_pooling_techniques(poolings_args, name_agg):

    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']    

    all_poolings_individuals = simple_poolings + simple_ns_poolings

    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)]
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)]
    four_tokens_poolings = [f"{a}+{b}+{c}+{d}" for a, b, c, d in combinations(all_poolings_individuals, 4)]

    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = all_poolings_individuals + two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'best':
        pooling_prefixs = two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
        #return pooling_prefixs
    if 'simple_all' in poolings_args:
        pooling_prefixs += all_poolings_individuals
        #return pooling_prefixs
    if 'simple-ns' in poolings_args:
        pooling_prefixs += simple_ns_poolings
        #return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
        #return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings
        #return pooling_prefixs  
    if 'four' in poolings_args:
        pooling_prefixs += four_tokens_poolings
        #return pooling_prefixs     
    
    
    if len(pooling_prefixs) > 0:
        return pooling_prefixs
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    if final_layer == 12:
        list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_base()
    if final_layer == 24:
        list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_large()
        
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    lyrs = []
        
    if agg_layers_args[0] == 'ALL':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        lyrs += list_lyrs_agg
        return lyrs
    
    if agg_layers_args[0] == 'SUMAGGLAYERS':
        return list_lyrs_agg_sum
    
    if agg_layers_args[0] == 'AVGAGGLAYERS':
        return list_lyrs_agg_avg
    
    if agg_layers_args[0] == 'LYR':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        return lyrs
    
    if agg_layers_args[0] == 'BEST':
        return ["BEST"]
    
    else:
        return agg_layers_args

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    # batch é uma lista de listas de palavras, ex: [['bom', 'dia'], ['frase', 'maior']]
    
    # --- NOVO: Lógica de ordenação ---
    # Mantém o registro dos índices originais
    original_indices = np.arange(len(batch))
    
    # Calcula o comprimento de cada sentença
    lengths = np.array([len(sent) for sent in batch])
    
    # Obtém os índices que ordenariam o batch por comprimento
    sorted_indices = np.argsort(lengths)
    
    # Ordena o batch e os índices originais
    sorted_batch = [batch[i] for i in sorted_indices]
    original_indices_sorted = [original_indices[i] for i in sorted_indices]
    
    # --- Fim da lógica de ordenação ---

    # Converte para strings e chama o _encode com o batch ordenado
    sentences = [' '.join(sent) for sent in sorted_batch]
    embeddings = params['encoder']._encode(sentences, params.current_task)
    
    # --- NOVO: Reordenar os embeddings para a ordem original ---
    # Cria um array para os embeddings na ordem correta
    restored_order_embeddings = np.zeros_like(embeddings)
    
    # Usa os índices ordenados para colocar cada embedding de volta em sua posição original
    restored_order_embeddings[original_indices_sorted] = embeddings
    
    return restored_order_embeddings

