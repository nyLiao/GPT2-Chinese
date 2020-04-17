python train.py --model_config model/prose_pretrain/config.json --tokenizer_path model/prose_pretrain/vocab.txt --tokenized_data_path data/tokenized/ --epochs 10 --lr 5.0e-5 --warmup_steps 2000 --log_step 100 --output_dir model/ --pretrained_model model/prose_pretrain/ --num_pieces 500 --batch_size 2 --device 0