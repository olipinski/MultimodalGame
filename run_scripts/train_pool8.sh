#!/bin/bash

source activate MultimodalGame

python model_symmetric.py \
-experiment_name pool8 \
-batch_size 32 \
-batch_size_dev 50 \
-save_interval 1000 \
-save_distinct_interval 50000 \
-m_dim 8 \
-h_dim 100 \
-desc_dim 100 \
-num_classes 10 \
-learning_rate 1e-4 \
-entropy_agent1 0.01 \
-entropy_agent2 0.01 \
-use_binary \
-max_epoch 2000 \
-top_k_dev 1 \
-top_k_train 1 \
-dataset_path ./data/oneshape/oneshape_simple_textselect \
-dataset_indomain_valid_path ./data/oneshape_valid/oneshape_simple_textselect \
-dataset_outdomain_valid_path ./data/oneshape_valid_all_combos/oneshape_simple_textselect \
-dataset_name oneshape_simple_textselect \
-dataset_size_train 5000 \
-dataset_size_dev 1000 \
-wv_dim 100 \
-glove_path ./glove/glove-100d.txt \
-log_path ./logs/ \
-debug_log_level INFO \
-cuda \
-log_interval 1000 \
-log_dev 5000 \
-log_self_com 25000 \
-reward_type "cooperative" \
-randomize_comms \
-random_seed 17 \
-agent_pools \
-num_agents 8 \
-check_accuracy_interval 30000
