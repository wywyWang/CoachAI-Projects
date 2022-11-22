current_time=`date +%Y%m%d-%H%M%S`
model_path="./model/${current_time}_"$1"/"
python train.py --output_folder_name ${model_path} --model_type $1 --encode_length $2 --max_ball_round 35 --shot_dim 32 --area_dim 32 --player_dim 32 --encode_dim 32 --seed_value $3
python evaluate.py ${model_path}