#!/bin/zsh
for ((i = 40; i < 50; i++))
do
    current_time=`date +%Y%m%d-%H%M%S`
    model_path="./model/${current_time}_"$1"_"$i"/"
    python train.py --output_folder_name ${model_path} --model_type $1 --encode_length $2 --sample $3 --seed_value ${i}
    python evaluate.py ${model_path}
done

# current_time=`date +%Y%m%d-%H%M%S`
# model_path="./model/${current_time}_"$1"/"
# python train.py --output_folder_name ${model_path} --model_type $1 --encode_length $2
# python evaluate.py ${model_path}