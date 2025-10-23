#!/bin/bash
python generate_results_tables.py --task_type classification && sleep 3 && python join_tables.py