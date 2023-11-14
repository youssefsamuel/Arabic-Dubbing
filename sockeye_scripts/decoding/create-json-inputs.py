import argparse
import json
import os

FACTOR_TYPES = [
    'duration',
    'total_duration_remaining',
    'segment_duration_remaining',
    'pauses_remaining'
]

TEXT_PAD_TOKEN = '<shift>'
SRC_SEGMENT_DELIMITER = '<||>'

#$ python3 sockeye_scripts/decoding/create-json-inputs.py -d processed_datasets/de-text-noisy-durations0.1-en-phones-durations --subset test --output-segment-durations -o processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de.json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data-dir", "-d", required=True,
                    help="Directory containing test.de/valid.de and the multi_factored[_eow] directory")
parser.add_argument("--subset", required=True, choices=["valid", "test"],
                    help="Dataset to prepare (valid or test)")
parser.add_argument("--output-segment-durations", action='store_true',
                    help="Include segment durations for forcing at inference time")
parser.add_argument("--reinsert-eow", action='store_true',
                    help="Include prefix for EOW factor")
parser.add_argument("--ignore-factors", default=[], nargs='+', choices=FACTOR_TYPES,
                    help="Remove specified factors from the JSON input. This is meant to be used for the factor ablation study.")
parser.add_argument("--remove-src-segments", action='store_true',
                    help="Remove segment duration information from the source")
parser.add_argument("--output", "-o",
                    help="Path to write JSON file")

args = parser.parse_args()
if args.output is None:
    # Set default output path
    args.output = os.path.join(args.data_dir, args.subset + '.json_input') #in our case we pass the output path when calling the function

# Now getting segments directly so this function is unnecessary
# def get_segment_durations(segment_durations_remaining_line: List[str]) -> List[int]:
#     segment_durations = [int(segment_durations_remaining_line[0])]
#     for pos, dur in enumerate(map(int, segment_durations_remaining_line[1:]), start=1):
#         if dur > int(segment_durations_remaining_line[pos - 1]):
#             segment_durations.append(dur)

#     return segment_durations

for factor in args.ignore_factors:
    FACTOR_TYPES.remove(factor)

if args.reinsert_eow:
    factor_dir = 'multi_factored_eow'
    FACTOR_TYPES.append('eow')
else: #Our case
    factor_dir = 'multi_factored'


#data_dir: processed_datasets/de-text-noisy-durations0.1-en-phones-durations

#f_src:  processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de
#f_trg:  processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multifactored/test.en.text
#f_segs: processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.segments
#f_out:  processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de.json
with open(os.path.join(args.data_dir, args.subset+'.de')) as f_src, \
     open(os.path.join(args.data_dir, 'multi_factored', args.subset+'.en.text')) as f_trg, \
     open(os.path.join(args.data_dir, args.subset+'.segments')) as f_segs, \
     open(args.output, 'w') as f_out:
    f_factors = []
    for f in FACTOR_TYPES:
        #append to f_factors a file handler to each of the files in multlifactored directory
        f_factors.append(open(os.path.join(args.data_dir, factor_dir, args.subset+'.en.' + f))) 


    """
        we will create a dictionary for each line.
        keys:
            1. text: test.de (source file)
            2. target_prefix: <shift> (first token given to the model)
            3. target_prefix_factors: we get the first column of each multifactored file.
                'duration'
                'total_duration_remaining'
                'segment_duration_remaining'
                'pauses_remaining'
            4. target_segment_durations: list of durations of each segment 
            5. use_target_prefix_all_chunks: False
    """
    for src_line, trg_text, segments, *trg_factors in zip(f_src, f_trg, f_segs, *f_factors):
        line_dict = dict() #dictionary for each line
        line_dict['text'] = src_line.strip() #german text
        if args.remove_src_segments:
            line_dict['text'] = line_dict['text'].split(SRC_SEGMENT_DELIMITER)[0].rstrip()
            # during training all the target factors are needed, 
            # however, for testing and validation we only need the first column
        line_dict['target_prefix'] = TEXT_PAD_TOKEN
        line_dict['target_prefix_factors'] = [l.strip().split()[0] for l in trg_factors]
        if args.output_segment_durations:
            line_dict['target_segment_durations'] = list(map(int, segments.strip().split()))
            # line_dict['target_segment_durations'] = get_segment_durations(trg_factors[2].strip().split())
        line_dict['use_target_prefix_all_chunks'] = 'false'
        f_out.write(json.dumps(line_dict, ensure_ascii=False).replace('"false"', 'false') + '\n')

    for fh in f_factors:
        fh.close()
