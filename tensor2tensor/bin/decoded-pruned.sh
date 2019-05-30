BEAM_SIZE=4
ALPHA=0.6
CUDA_VISIBLE_DEVICES=""

set -eu


h=transformer_adaptive_early_dropout_botk_99_25000
o=$h
python tensor2tensor/tensor2tensor/bin/t2t_prune_save.py --data_dir=gs://for-ai/t2t_data --problem=translate_ende_wmt32k --model=transformer --hparams_set=$h --output_dir=gs://for-ai/runs/a1vn/$o --pruning_params_set=transformer_weight --worker_gpu=0
for k in $(seq 1 1 4)
do
    o=${h}_${k}
    python tensor2tensor/tensor2tensor/bin/t2t_prune_save.py --data_dir=gs://for-ai/t2t_data --problem=translate_ende_wmt32k --model=transformer --hparams_set=$h --output_dir=gs://for-ai/runs/a1vn/$o --pruning_params_set=transformer_weight --worker_gpu=0
done

o=$h
python ~/tensor2tensor/tensor2tensor/bin/t2t-decoder \
--data_dir=gs://for-ai/t2t_data \
--problem=translate_ende_wmt32k \
--model=transformer \
--hparams_set=$h \
--worker_gpu=0 --use_tpu=True --cloud_tpu_name=ivan-1  --decode_to_file="gs://for-ai/runs/a1vn/$o/prune_0/translation.en" --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
--output_dir="gs://for-ai/runs/a1vn/$o/prune_0"
for k in $(seq 1 1 4)
do
    o=${h}_${k}
    python ~/tensor2tensor/tensor2tensor/bin/t2t-decoder \
    --data_dir=gs://for-ai/t2t_data \
    --problem=translate_ende_wmt32k \
    --model=transformer \
    --hparams_set=$h \
    --worker_gpu=0 --use_tpu=True --cloud_tpu_name=ivan-1  --decode_to_file="gs://for-ai/runs/a1vn/$o/prune_0/translation.en" --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
    --output_dir="gs://for-ai/runs/a1vn/$o/prune_0"
done

o=$h
python ~/tensor2tensor/tensor2tensor/bin/t2t-bleu \
--translation="gs://for-ai/runs/a1vn/$o/prune_0/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.decodes" \
--reference="gs://for-ai/runs/a1vn/$o/prune_0/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.targets"
for k in $(seq 1 1 4)
do
    o=${h}_${k}
    echo $k
    python ~/tensor2tensor/tensor2tensor/bin/t2t-bleu \
    --translation="gs://for-ai/runs/a1vn/$o/prune_0/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.decodes" \
    --reference="gs://for-ai/runs/a1vn/$o/prune_0/translation.en.transformer.$h.translate_ende_wmt32k.beam4.alpha0.6.targets"
done
