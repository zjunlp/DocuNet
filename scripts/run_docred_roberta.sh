#! /bin/bash
export CUDA_VISIBLE_DEVICES=2

if true; then
  type=context-based
  bs=2
  bls=(3e-5)
  ul=4e-4
  accum=2
  try=4
  for bl in ${bls[@]}
  do
  python -u ../train_balanceloss.py --data_dir ../dataset/docred \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type roberta \
  --model_name_or_path roberta-large \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 4 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06\
  --num_train_epochs 100.0 \
  --seed 111 \
  --num_class 97 \
  --save_path ../checkpoint/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${type}_${try}.pt \
  --log_dir ../logs/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${type}_${try}.log \
  --train_from_saved_model /data/lzb/DocRED/checkpoint/docred/train_roberta-lr3e-5_accum2_unet-lr4e-4_type_context-based_3.pt \
  --load_path /data/lzb/DocRED/checkpoint/docred/train_roberta-lr3e-5_accum2_unet-lr4e-4_type_context-based_3.pt
  done
fi

if false; then
  type=context-based
  bs=2
  bls=(2e-6)
  ul=5e-5
  accum=2
  for bl in ${bls[@]}
  do
  python -u ../train_sample.py --data_dir ../dataset/docred \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type roberta \
  --model_name_or_path roberta-large \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 4 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.00\
  --num_train_epochs 10.0 \
  --seed 111 \
  --num_class 97 \
  --save_path ../checkpoint/samples_epoch10_neg8_continue_best_blr${bl}_ulr${ul}_bs$bs.pt \
  --log_dir ../logs/samples_epoch10_neg8_continue_best_blr${bl}_ulr${ul}_bs$bs.log
  done
fi

