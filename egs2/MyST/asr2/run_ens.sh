#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="[wavlm_large,hubert_large_ll60k,xls_r_300m]"  # use model_type/layer_index
layer_index="[21,21,21]"

#kmeans_feature="wavlm_large"  # use model_type/layer_index
# layer_index="21"
nclusters=2000

src_lang="wavlm_large_hubert_large_ll60k_xls_r_300m_21_21_21_km2000"
tgt_lang=en

train_set="train_wavlm"
train_dev="eval_wavlm"
test_sets="test_wavlm"

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2_ens.sh \
    --kmeans_opts "--batch_bins 5000 --nj 2" \
    --kmeans_feature "${kmeans_feature}" \
    --layer_index "${layer_index}" \
    --nclusters "${nclusters}" \
    --stage 12 \
    --stop_stage 13 \
    --ngpu 2 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"
