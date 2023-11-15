#!/bin/bash
declare -a model_list=("LSTM" "ShuttleNet" "Transformer" "rGCN" "DNRI" "GCN" "eGCN" "DyMF")
declare -a seeds=("1")
declare -a seqence_length=("2" "4" "8")
for model in "${model_list[@]}";
do
    for seed in "${seeds[@]}";
    do
        echo "${seed}============================================================================"
        for encode_length in "${seqence_length[@]}";
        do
            current_time=`date +%Y-%m-%d-%H:%M:%S`
            model_name=$model
            model_path="./model/${model_name}_${encode_length}_${current_time}"
            echo "Model Name: ${model_name}	Encode Length: ${encode_length}"
            echo "Training Start:"
            python train.py --model_type ${model_name} --model_folder ${model_path} --encode_length ${encode_length} --seed ${seed}
            echo "Evaluating Start:"
            python evaluate.py ${model_path} 10
            echo "====================="
        done 
    done
done
