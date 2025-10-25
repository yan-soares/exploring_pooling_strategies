#!/bin/bash
python main_experiments.py --task_type classification --models allmpnet --initial_layer 12 --final_layer 12 --poolings AVG --agg_layers SUM-5-12 --save_dir best_results_new_code &&
python main_experiments.py --task_type classification --models deberta-base --initial_layer 12 --final_layer 12 --poolings AVG --agg_layers SUM-7-10 --save_dir best_results_new_code &&
python main_experiments.py --task_type classification --models allmpnet --initial_layer 12 --final_layer 12 --poolings AVG+AVG-NS --agg_layers SUM-8-12 --save_dir best_results_new_code &&
python main_experiments.py --task_type classification --models deberta-base --initial_layer 12 --final_layer 12 --poolings AVG+AVG-NS --agg_layers SUM-6-10 --save_dir best_results_new_code &&
python main_experiments.py --task_type classification --models allmpnet --initial_layer 12 --final_layer 12 --poolings CLS+AVG+AVG-NS --agg_layers SUM-7-12 --save_dir best_results_new_code &&
python main_experiments.py --task_type classification --models deberta-base --initial_layer 12 --final_layer 12 --poolings CLS+AVG+AVG-NS --agg_layers SUM-8-11 --save_dir best_results_new_code