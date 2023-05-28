#!/bin/bash
. ~/.bashrc
. ~/.bash_aliases




# field of view
run_job 7x7_lstm_v1 -f 1e9 -s 7 -v 1 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
run_job 7x7_lstm_v3 -f 1e9 -s 7 -v 3 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
run_job 9x9_lstm_v1 -f 1e9 -s 9 -v 1 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
run_job 9x9_lstm_v3 -f 1e9 -s 9 -v 3 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm

# maze size
# run_job 7x7_lstm_v2 -f 1e9 -s 7 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
# run_job 9x9_lstm_v2 -f 1e9 -s 9 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
# run_job 11x11_lstm_v2 -f 1e10 -s 11 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
# run_job 13x13_lstm_v2 -f 1e10 -s 13 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm
# run_job 15x15_lstm_v2 -f 1e10 -s 15 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 --use_lstm

# no lstm
run_job 9x9_v2 -f 1e9 -s 9 -v 2 -wp impala_maze_size --num_actors 16 --batch_size 8 

# number of targets
# randomization
