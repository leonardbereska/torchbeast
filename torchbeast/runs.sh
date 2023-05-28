#!/bin/bash
. ~/.bashrc
. ~/.bash_aliases
# e.g. run_job test_wandb2 --size 7 -mm lstm -v 2 -wp testy



# test field of view. How does field of view affect performance?
# run_job 7x7_lstm_v1 -s 7 -f 1e8 -mm lstm -v 1 -wp 230527_field_of_view 
# run_job 7x7_lstm_v2 -s 7 -f 1e8 -mm lstm -v 2 -wp 230527_field_of_view 
# run_job 7x7_lstm_v3 -s 7 -f 1e8 -mm lstm -v 3 -wp 230527_field_of_view
# run_job 9x9_lstm_v2 -s 9 -f 1e8 -mm lstm -v 2 -wp 230527_field_of_view
# run_job 9x9_lstm_v3 -s 9 -f 1e8 -mm lstm -v 3 -wp 230527_field_of_view 
# run_job 9x9_lstm_v4 -s 9 -f 1e8 -mm lstm -v 4 -wp 230527_field_of_view

# test 2 vs 3 targets in 7x7 maze and 9x9 maze
# run_job 7x7_v2 -f 10 -s 7 -v 2 -wp impala_tests --num_actors 32 --batch_size 16 
# run_job 9x9_v2 -f 1e10 -s 9 -v 2 -wp impala_tests --num_actors 32 --batch_size 16 
# run_job 9x9_lstm_v2 -f 1e10 -s 9 -v 2 -wp impala_tests --num_actors 32 --batch_size 16 --use_lstm
run_job 9x9_lstm_v2_small_embedding -f 1e10 -s 9 -v 2 -wp impala_tests --num_actors 32 --batch_size 8 --use_lstm

# 9x9 maze with longer training e.g. 1e8
# run_job 9x9_lstm_v2_f8 -wp 9x9 -s 9 -f 1e8 -mm lstm -v 2 --use-cpu
# 9x9 maze with larger field of view e.g. -v 3
# run_job 9x9_lstm_v3 -wp field_of_view -s 9 -f 1e7 -mm lstm -v 3 --use-cpu
# 9x9 maze with larger field of view e.g. -v 4
# run_job 9x9_lstm_v4 -wp field_of_view -s 9 -f 1e7 -mm lstm -v 4 --use-cpu
# 9x9 maze with larger field of view e.g. -v 3
# run_job 9x9_ckconv_v3 -wp field_of_view -s 9 -f 1e7 -mm ckconv -v 3
# 7x7 maze ckconv with more layers
# run_job 7x7_ckconv_v2 -wp 7x7_lstm_ckconv -s 7 -f 1e7 -mm ckconv -v 2
# a2c 7x7 lstm
# run_job 7x7_lstm_v2_a2c -wp a2c -s 7 -f 1e7 -mm lstm -v 2 --algo a2c --use-cpu
