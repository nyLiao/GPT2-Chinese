for ((i=0;i<10;i++)); do
    name="model_epoch$i"
    python eval.py --model_config model/pretrain/config.json --tokenizer_path model/pretrain/vocab.txt --tokenized_data_path data/tokenized/ --batch_size 4 --log_step 100 --pretrained_model model/$name/ --num_pieces 500 --output_name $name.txt --device 0
    echo "========================================"
done
