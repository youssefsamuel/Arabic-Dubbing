#!/bin/bash
set -Eeuo pipefail
source `dirname $0`/../config


# calling line: ./sockeye_scripts/evaluation/evaluate-factored.sh processed_datasets/de-text-noisy-durations0.1-en-phones-durations/valid.en models/sockeye/baseline_factored_noised0.1/valid_eval_checkpoint_16/valid.en.output

#ROOT=/content/Arabic-Dubbing
#DATA_HOME=${ROOT}/processed_datasets
#MODELS_HOME=/content/drive/MyDrive/models/sockeye

REF=$1 #processed_datasets/de-text-noisy-durations0.1-en-phones-durations/valid.en
# English phonemes with durations from the dataset not the model output 

HYP=$2 #models/sockeye/trained_baselines/baseline_factored_noised0.1/valid_eval_checkpoint_16/valid.en.output
# English phonemes with durations (the model output) 


EVAL_DIR=`dirname ${HYP}`
P2G_DIR=${ROOT}/models/phoneme_to_grapheme

if [[ ! -s ${DATA_HOME}/ar-text-without-durations-en-text-without-durations/valid.en ]]; then
    echo "Please generate ${DATA_HOME}/ar-text-without-durations-en-text-without-durations first. This is required for the reference text."
    exit 1
else
    # REF_TEXT: english words without bpe (validation dataset)
    # SRC_TEXT: arabic words without bpe (validation dataset)

    REF_TEXT=${DATA_HOME}/ar-text-without-durations-en-text-without-durations/valid.debpe.en 
    SRC_TEXT=${DATA_HOME}/ar-text-without-durations-en-text-without-durations/valid.debpe.ar
    if [[ ! -s ${REF_TEXT} ]]; then
        sed -r 's/(@@ )|(@@ ?$)//g' ${DATA_HOME}/ar-text-without-durations-en-text-without-durations/valid.en > ${REF_TEXT}
        sed -r 's/(@@ )|(@@ ?$)//g' ${DATA_HOME}/ar-text-without-durations-en-text-without-durations/valid.ar > ${SRC_TEXT}
    fi
fi

# This part is used to convert the Englihs phonemes (model output) to words, so that we can compare with the test dataset (REF_TEXT)
if [[ ! -s ${HYP}.words ]]; then
    # Phoneme-to-grapheme conversion
    echo "Converting phonemes to graphemes for translation quality evaluation"

    # Separate phonemes and durations
    python `dirname $0`/separate-hyp-factors.py ${HYP} --shift

    # Preprocess to prepare output for phoneme-to-grapheme conversion
    sed "s/\[pause\]//g" ${HYP}.phonemes > ${HYP}.nopause
    python ${ROOT}/phonemes-eow-to-phoneticwords.py ${HYP}.nopause withoutdurations

    subword-nmt apply-bpe \
        -c ${P2G_DIR}/phoneme_to_grapheme_bpe_10k \
        -i ${HYP}.nopause.phoneticwords \
        | cut -d' ' -f1-1023 \
        > ${HYP}.nopause.phoneticwords.phonebpe

    # Reduce --batch-size if your GPU runs out of memory
    # This is the second model: pretrained model to convert english phonemes (output from model 1) to words
    sockeye-translate \
        --models ${ROOT}/models/phoneme_to_grapheme \
        --checkpoint 48 \
        -b 5 \
        --batch-size 128 \
        --chunk-size 20000 \
        -i ${HYP}.nopause.phoneticwords.phonebpe \
        | sed -r "s/(@@ )|(@@ ?$)//g" \
        > ${HYP}.words
else
    echo "${HYP}.words already exists and will not be re-generated."
fi

# Bleu / Prism --> REF_TEXT vs HYP.words (English words - English words)
# Comet --> SRC_TEXT (arabic words) + REF_TEXT + HYP.words
# Speech overlap --> phonemes + durations (REF + HYP)

echo "Calculating translation quality metrics:"
sacrebleu ${REF_TEXT} -m bleu -f text -lc --tokenize none < ${HYP}.words
echo "Prism:"
# Prism uses PyTorch 1.4.0 which expects CUDA 10.1. Sockeye expects a much more modern CUDA (e.g. 11.7). So run prism on CPU (takes ~8min)
CUDA_VISIBLE_DEVICES=-1 `dirname ${CONDA_PREFIX}`/prism/bin/python ${ROOT}/third_party/prism/prism.py --cand ${HYP}.words --ref ${REF_TEXT} --lang en --model-dir ${ROOT}/third_party/prism/m39v1
echo "COMET:"
comet-score --gpus 1 --quiet --batch_size 128 \
    --model wmt21-comet-da \
    -s ${SRC_TEXT} -t ${HYP}.words -r ${REF_TEXT} \
    | tail -n1

echo -e "\nSpeech overlap metrics:"
python `dirname $0`/format-phonemes-durations.py ${HYP}
python count_durations.py ${REF} ${HYP}.altformat
