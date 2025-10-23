#!/bin/bash
python main_experiments.py --task_type classification --models deberta-base --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings CLS+AVG+AVG-NS  --agg_layers SUM-8-11 --save_dir experiments_more_epochs100 &&
python main_experiments.py --task_type classification --models allmpnet --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings AVG  --agg_layers SUM-5-12 --save_dir experiments_more_epochs100 &&
python main_experiments.py --task_type classification --models allmpnet --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings AVG+AVG-NS  --agg_layers SUM-8-12 --save_dir experiments_more_epochs100 &&
python main_experiments.py --task_type classification --models allmpnet --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings CLS+AVG+AVG-NS  --agg_layers SUM-7-12 --save_dir experiments_more_epochs100 &&
python main_experiments.py --task_type classification --models deberta-base --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings AVG  --agg_layers SUM-7-10 --save_dir experiments_more_epochs100 &&
python main_experiments.py --task_type classification --models deberta-base --epochs 100 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 12 --final_layer 12 --poolings AVG+AVG-NS  --agg_layers SUM-6-10 --save_dir experiments_more_epochs100
 

