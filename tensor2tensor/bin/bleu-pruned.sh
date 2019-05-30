BEAM_SIZE=4
ALPHA=0.6
CUDA_VISIBLE_DEVICES=""

set -eu


h=transformer_xtreme_dropout_botk_99_100000_keep3
o=$h
python tensor2tensor/tensor2tensor/bin/t2t_prune_save.py --data_dir=gs://for-ai/t2t_data --problem=translate_ende_wmt32k --model=transformer --hparams_set=$h --output_dir=gs://for-ai/runs/a1vn/$o --pruning_params_set=transformer_weight --worker_gpu=0

for i in $(seq 0 1 9)
do
    python ~/tensor2tensor/tensor2tensor/bin/t2t-decoder \
  --data_dir=gs://for-ai/t2t_data \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=$h \
  --worker_gpu=0 --use_tpu=True --cloud_tpu_name=ivan-1  --decode_to_file="gs://for-ai/runs/a1vn/$o/prune_$i/translation.en" --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --output_dir="gs://for-ai/runs/a1vn/$o/prune_$i"

done

for i in $(seq 0 1 9)
do
    sparsity="$((10*$i))"
    echo "Sparsity: $sparsity"
    python ~/tensor2tensor/tensor2tensor/bin/t2t-bleu \
    --translation="gs://for-ai/runs/a1vn/$o/prune_$i/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.decodes" \
    --reference="gs://for-ai/runs/a1vn/$o/prune_$i/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.targets"

done
