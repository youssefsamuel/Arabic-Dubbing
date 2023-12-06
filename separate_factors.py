"""
Generate separate text and factor files from the combined data
"""

import os
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


SRC_LANG = 'ar'
TGT_LANG = 'en'
SEGMENTS_SUFFIX = 'segments'

SPLITS = ['train', 'valid', 'test']

PAUSE_TOKEN = '[pause]'
EOW_TOKEN = '<eow>'
NOT_EOW_TOKEN = '<!eow>'
EOW_FACTOR_SPECIAL_TOKEN = '<special>'
DUMMY_DURATION = 0  # Duration value for eow/pause
SRC_DURATION_SEP = '<||>'

FACTOR_TYPES = [
    'DURATION',
    'PAUSES_REMAINING',
    'TOTAL_DURATION_REMAINING',
    'SEGMENT_DURATION_REMAINING',
    'EOW'
]

class FactorFileManager:
    """
    Context Manager to neatly manage all the factor output file handles
    """
    def __init__(self, prefix, allow_overwrite=False):
        self.handles = dict()
        self.prefix = prefix #multifactored + split + language
        self.overwrite = allow_overwrite

    def __enter__(self):
        for f in FACTOR_TYPES:
            output_path = self.prefix + f.lower()  # example: multifactored/train.en.duration
            if os.path.exists(output_path) and not self.overwrite:
                raise FileExistsError(f"Factor file {output_path} already exists. Rerun with --allow-overwrite if you're sure you want to overwrite.")
            self.handles[f] = open(output_path, 'w') #each key in the dictionary map to a file handler
            logging.info(f"Writing {f} factors to {output_path}")

        text_output_path = self.prefix + 'text'  # example: multifactored/train.en.text
        if os.path.exists(text_output_path) and not self.overwrite:
            raise FileExistsError(f"Factor file {text_output_path} already exists. Rerun with --allow-overwrite if you're sure you want to overwrite.")
        self.handles['text'] = open(text_output_path, 'w')
        logging.info(f"Writing text to {text_output_path}")
        return self.handles

    def __exit__(self, exc_type, exc_value, exc_tb):
        for fh in self.handles.values():
            fh.close()

def calculate_factors(line, npause, segments, pad_token, no_shift=False, eow_factor=False):
    # For a single line: we will get its factors.
   
    factors = dict() 
    for f in FACTOR_TYPES:
        factors[f] = []
    # factors dictionary: 
        #keys are the factortypes, the values are lists.
    text = [] #f_main

    if not no_shift: # If there is shift
        factors['PAUSES_REMAINING'].append(npause) #total number of pauses in that sentence
        # Dummy initial tokens when we're accounting for the extra shift
        # to accommodate calculated factors at step 0
        text.append(pad_token) #<shift>
        factors['DURATION'].append(0) #f_dur: at first = 0 because no phonemes yet.
        if eow_factor:
            factors['EOW'].append(EOW_FACTOR_SPECIAL_TOKEN) #<special>

    # Separate durations and calculate pauses remaining
    token_pos = 0 # token index
    # loop on each token
    while token_pos < len(line):
        # in this loop we get fmain (=text), fdur = DURATION, fpause
        # so we still need to compute total durations and segment durations.
        if line[token_pos] == PAUSE_TOKEN:
            text.append(line[token_pos])
            # Duration 0 for pause tokens
            factors['DURATION'].append(DUMMY_DURATION) # = 0
            if eow_factor: # this flag is not used in our case, it is used to predict eow as a factor.
                factors['EOW'].append(EOW_FACTOR_SPECIAL_TOKEN)
            npause -= 1 
            token_pos += 1
            factors['PAUSES_REMAINING'].append(npause)
        elif line[token_pos] == EOW_TOKEN:
            if eow_factor:
                factors['EOW'][-1] = EOW_TOKEN
            else:
                text.append(line[token_pos]) # <eow>
                # Duration 0 for eow tokens
                factors['DURATION'].append(DUMMY_DURATION) # 0
                factors['PAUSES_REMAINING'].append(npause) 
            token_pos += 1
        else:
            # Phoneme
            text.append(line[token_pos])
            factors['DURATION'].append(int(line[token_pos + 1])) 
            """
             because our train.ar file consists of AH0 12 EH1 2, 
             so if a phoneme is found, we get the phoneme, add it to text, 
             then we get its duration and add it
             finally we skip two tokens.
            """
            if eow_factor:
                factors['EOW'].append(NOT_EOW_TOKEN)
            token_pos += 2
            factors['PAUSES_REMAINING'].append(npause)

    if not no_shift:
        # Calculate factors for total and segment durations remaining        
        segment_idx = 0  #segments is a list of integers, each integer is the number of frames in that segment.
        curr_segment = segments[0] # segment duration
        curr_total = sum(segments) # total duration of the sentence
        factors['SEGMENT_DURATION_REMAINING'] = []
        factors['TOTAL_DURATION_REMAINING'] = []
        for pos, dur in enumerate(factors['DURATION']): # pos = 0 --> dur = x (token)
            if text[pos] == PAUSE_TOKEN: 
                segment_idx += 1 # go to the next segment
                curr_segment = segments[segment_idx] 
            else:
                curr_segment -= dur # segment duration - phoneme duration
            curr_total -= dur # remaining sentence duration is always deducted.
            factors['SEGMENT_DURATION_REMAINING'].append(curr_segment)
            factors['TOTAL_DURATION_REMAINING'].append(curr_total)

    """
        we constructed the table in the paper.
    """
    return text, factors 

def process_files(args: argparse.Namespace):
    for split in SPLITS: # split = train or test or dev
        logging.info(f"Processing {split}")
        # open the three files for that split
        with open(os.path.join(args.input_dir, f"{split}.{SRC_LANG}")) as src_in, \
             open(os.path.join(args.input_dir, f"{split}.{TGT_LANG}")) as tgt_in, \
             open(os.path.join(args.input_dir, f"{split}.{SEGMENTS_SUFFIX}")) as src_segments, \
             FactorFileManager(os.path.join(args.output_dir, f"{split}.{TGT_LANG}."), args.allow_overwrite) as factor_handles:
            for tgt_line in tgt_in: # english sentences [phonemes + [pause] + durations + <eow>]
                tgt_line = tgt_line.strip().split()
                tgt_pauses = tgt_line.count(PAUSE_TOKEN) # [pause]
                if not args.no_src_durations: # if there are source durations
                    src_pauses = len(src_in.readline().strip().split(SRC_DURATION_SEP)[1].strip().split(' ')) - 1 
                    # count the pauses in the source 
                    # the source consists of phonemes + <||> + bins, so we get the bins, count them, subtract 1 to get pauses.
                    assert src_pauses == tgt_pauses, "Mismatched number of pauses in source and target"

                segments = list(map(int, src_segments.readline().strip().split()))
                # convert the line in the segments: "245 532" into list of int [245, 532]

                """
                    input:  tgt_line: english sentence
                            npauses: pauses in english sentence
                            segments: list of integers, number of frames per segment
                """
                text, factors = calculate_factors(line=tgt_line, 
                                                  npause=tgt_pauses,
                                                  segments=segments,
                                                  pad_token=args.pad_token,
                                                  no_shift=args.no_shift,
                                                  eow_factor=args.eow_as_factor)
                
                # Write to files
                factor_handles['text'].write(' '.join(text) + '\n')
                for f in factors.keys():
                    factor_handles[f].write(' '.join(map(str, factors[f])) + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_arg = parser.add_argument("--input-dir", "-i", required=True,
                                    help="Directory containing train, dev, and test sets")
    # when calling the function: in our case: processed_datasets/ar-text-noisy-durations0.1-en-phones-durations

    parser.add_argument("--output-dir", "-o", default='multi_factored',
                        help=f"Subdirectory under {input_arg.metavar} for factors to be written to")
    parser.add_argument("--no-shift", action='store_true',
                        help="If this is True, text and durations will not be prepended with a dummy token. "
                             "This shift is needed to match Sockeye's output factor shift, so use with care.")
    # no shift: we don't use because it is needed for the Sockeye model. 
    parser.add_argument("--pad-token", default="<shift>",
                        help="Dummy token to insert to account for factor position shifts. "
                             "Sockeye moves target factors right by 1 to condition factor generation on output tokens.")
    parser.add_argument("--no-src-durations", action='store_true',
                        help="Indicate that there are no duration bins in the source. Just skips a check, doesn't affect output.")
    parser.add_argument("--eow-as-factor", action='store_true',
                        help="Remove <eow> from source and use a factor to denote EOW or not EOW")
    parser.add_argument("--allow-overwrite", action='store_true',
                        help="Allow overwriting if the factor files already exist.")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # processed_datasets/ar-text-noisy-durations0.1-en-phones-durations/multi_factored

    if args.no_shift:
        logging.warning("Shifting tokens is disabled. Make sure this matches what you want with Sockeye output factors. "
                        "This setting disables some factor outputs since they won't align without the shift.")
        FACTOR_TYPES.remove('TOTAL_DURATION_REMAINING')
        FACTOR_TYPES.remove('SEGMENT_DURATION_REMAINING')

    if not args.eow_as_factor:
        logging.info("EOW factor file will not be generated. Use --eow-as-factor to include it.")
        FACTOR_TYPES.remove('EOW')

    process_files(args)