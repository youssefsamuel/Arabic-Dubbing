import codecs
import json
import os
import pickle
import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from preprocessing_scripts import load_tsv, add_noise_to_durations, get_speech_durations, Bin
from subword_nmt.apply_bpe import BPE


AR_OUTPUT_CHOICES_WITH_DURATIONS = {
    'ar-text-clean-durations',
    'ar-text-noisy-durations',
    'ar-text-dummy-durations'
}
AR_OUTPUT_CHOICES = AR_OUTPUT_CHOICES_WITH_DURATIONS.add('ar-text-without-durations')

EN_OUTPUT_CHOICES = {
    'en-text-without-durations',
    'en-phones-without-durations',
    'en-phones-durations'
}


def build_datasets(data_path,
                   duration_freq,
                   ar_output_type,
                   en_output_type,
                   output_dir,
                   bpe_ar,
                   bpe_en,
                   tsvs,
                   num_bins=100,
                   upsampling=None,
                   sd=None,
                   write_segments_to_file=False):
    if num_bins > 0:
        bin_instance = Bin(duration_freq, n=num_bins)
    counter = 0
    train_tsv, dev_tsv, test_tsv = tsvs
    train_ar, dev_ar, test_ar = [], [], [] 
    train_en, dev_en, test_en = [], [], []
    train_segments, dev_segments, test_segments = [], [], []
    return_durations = False
    return_text = False

    all_included_keys = set().union(train_tsv.keys(), dev_tsv.keys(), test_tsv.keys())

    for file in os.listdir(data_path): # covost mfa
        # We want only JSON files.
        name = file.split(".")[0]
        if os.path.isfile(os.path.join(data_path, name + ".json")):
            data = json.load(open(os.path.join(data_path, name + ".json")))
        else:
            logging.debug(f"{file} ignored")
            continue

        # Data that is not in the covost_tsv TSV files is not used
        if name not in all_included_keys:
            continue

        counter += 1 # number of files in mfa

        if en_output_type == 'en-phones-durations':
            return_durations = True
        if en_output_type == 'en-text-without-durations':
            return_text = True

        phones, duration_freq, _, durations, _, text = get_speech_durations(data,
                                                                            duration_freq,
                                                                            return_durations=return_durations,
                                                                            return_text=return_text)
        pauses_count = phones.count('[pause]')

        if return_durations:
            assert len(durations) >= 1

        if ar_output_type in AR_OUTPUT_CHOICES_WITH_DURATIONS:
            if num_bins > 0:
                bins = bin_instance.find_bin(speech_durations=durations) # number of frames in each segment in that file 

            # noisy or dummy durations for Ar
            """
            up sampling is used for data augmentation.
            we will get the durations (number of frames per segment for that file)
            we will add noise to these durations to get "noisy_durations"
            suppose the file has 3 segments, upsampling 10.

            Goal: to introduce flexibility so that the output is not mandatory the same length of the input
            so this will improve the translation quality.

            noisy_durations: 

            suppose durations = [5 6 7]
            after adding noise:
            noisy_durations = [[5.2 5.1 5.2 .....], [6.4 6.2 .......], [7.3, 7.4, .....]] each list 10 elements
            noisy_durations is a list of 3 (segments in file) lists: each sublist is of size 10 (up sampling)


            noisy bins = [[5.2, 6.4, 7.3], [], [], ...] 
            noisy bins is a list of 10 (up sampling) lists: each sublist is of size 3 (segments in file)

            noisy_durations_rearrange_int: 10 lists, each sublist 3 (same as noisy bins) but rounded.
            """
            if ar_output_type == 'ar-text-noisy-durations':
                noisy_durations = add_noise_to_durations(durations, sd, upsampling)
                if num_bins > 0:
                    noisy_bins = [[] for _ in range(upsampling)]
                    for dur in noisy_durations:
                        noisy_bins_temp = bin_instance.find_bin(speech_durations=dur)
                        for i in range(upsampling):
                            noisy_bins[i].append(noisy_bins_temp[i])
                noisy_durations_rearrange_int = [[] for _ in range(upsampling)] 
                for dur in noisy_durations:
                    for i in range(upsampling):
                        noisy_durations_rearrange_int[i].append(round(dur[i]))
            elif ar_output_type == 'ar-text-dummy-durations':
                temp = []
                for _ in range(len(bins)):
                    temp.append(' <X>')

        if en_output_type == 'en-phones-durations':
            if ar_output_type in AR_OUTPUT_CHOICES_WITH_DURATIONS:
                assert pauses_count == len(durations) - 1

        if name in train_tsv.keys():
            # Source side (Arabic)
            """
                if name in train:
                    if ar.clean:
                        sentence = ar sentece tokenized by bpe + <||> + bins
                    
                    if noisy:
                        put the sentence 10 times with corresponding bins.
                """
            sentence_segments = []
            if ar_output_type == 'ar-text-clean-durations':
                if num_bins > 0:
                    sentence = [bpe_ar.process_line(train_tsv[name][1]) + " <||> " + " ".join(bins)]
                else: # case of no bins (rarely occurs)
                    sentence = [bpe_ar.process_line(train_tsv[name][1]) + " <||> " + " ".join(map(str, durations))]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]
            elif ar_output_type == 'ar-text-noisy-durations':
                sentence = []
                for i in range(upsampling):
                    if num_bins > 0:
                        sentence.append(bpe_ar.process_line(train_tsv[name][1]) + " <||> " + " ".join(noisy_bins[i]))
                    else:
                        sentence.append(bpe_ar.process_line(train_tsv[name][1]) + " <||> " + " ".join(map(str, noisy_durations_rearrange_int[i])))
                    if return_durations and write_segments_to_file:
                        sentence_segments.append(" ".join(map(str, noisy_durations_rearrange_int[i])))
            elif ar_output_type == 'ar-text-dummy-durations':
                sentence = [bpe_ar.process_line(train_tsv[name][1]) + " <||> " + " ".join(temp)]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]
            elif ar_output_type == 'ar-text-without-durations':
                sentence = [bpe_ar.process_line(train_tsv[name][1])]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]

            train_ar.extend(sentence)
            train_segments.extend(sentence_segments)

            # Target side (English)
            if en_output_type == 'en-text-without-durations':
                train_en.append(bpe_en.process_line(text))
            elif en_output_type.startswith('en-phones'):
                if ar_output_type != 'ar-text-noisy-durations':
                    train_en.append(" ".join(phones))
                else:
                    for _ in range(upsampling):
                        train_en.append(" ".join(phones))

        elif name in dev_tsv.keys() or name in test_tsv.keys():
            if name in dev_tsv.keys():
                curr_tsv = dev_tsv
                curr_ar = dev_ar
                curr_en = dev_en
                curr_segments = dev_segments
            else:
                curr_tsv = test_tsv
                curr_ar = test_ar
                curr_en = test_en
                curr_segments = test_segments

            # Source side (German)
            if ar_output_type == 'ar-text-noisy-durations' or ar_output_type == 'ar-text-clean-durations':
                if num_bins > 0:
                    sentence = bpe_ar.process_line(curr_tsv[name][1]) + " <||> " + " ".join(bins)
                else:
                    sentence = bpe_ar.process_line(curr_tsv[name][1]) + " <||> " + " ".join(map(str, durations))
            elif ar_output_type == 'ar-text-dummy-durations':
                sentence = bpe_ar.process_line(curr_tsv[name][1]) + " <||> " + " ".join(temp)
            elif ar_output_type == 'ar-text-without-durations':
                sentence = bpe_ar.process_line(curr_tsv[name][1])
            if return_durations and write_segments_to_file:
                curr_segments.append(" ".join(map(str, durations)))

            curr_ar.append(sentence)

            # Target side (English)
            if en_output_type == 'en-text-without-durations':
                curr_en.append(bpe_en.process_line(text))
            elif en_output_type.startswith('en-phones'):
                curr_en.append(" ".join(phones))
            
        if counter % 20000 == 0:
            logging.info(f"{counter} files processed")


    """
       in case of ar text without + eng text without durations:
            sentences after bpe (no bins, no durations)
        

       in case of ar text noisy + english phones durations:
        train.ar:
            sentence after bpe + <||> + bins
            each sentence 10 times.
        valid.ar / test.ar:
            sentence after bpe + <||> + bins
            each sentence one time because we don't add noise to test or dev.
        
        train.en:
            phonemes of each sentence with the duration of each phoneme
            each sentence repeated 10 to match the sentences of the source.
        
        valid.en / test.en:
            phonemes of each sentence with the duration of each phoneme
            no repetition

        train.segments / val.segments / dev.segments:
            durations of each segment in that file


    """


    write_to_file(train_ar, os.path.join(output_dir, 'train.ar'))
    write_to_file(dev_ar, os.path.join(output_dir, 'valid.ar'))
    write_to_file(test_ar, os.path.join(output_dir, 'test.ar'))
    write_to_file(train_en, os.path.join(output_dir, 'train.en'))
    write_to_file(dev_en, os.path.join(output_dir, 'valid.en'))
    write_to_file(test_en, os.path.join(output_dir, 'test.en'))
    if train_segments != [] and write_segments_to_file:
        write_to_file(train_segments, os.path.join(output_dir, 'train.segments'))
        write_to_file(dev_segments, os.path.join(output_dir, 'valid.segments'))
        write_to_file(test_segments, os.path.join(output_dir, 'test.segments'))

    logging.info("Wrote new dataset to {}".format(output_dir))

def write_to_file(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write('{}\n'.format(line))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("--ar-output-type", "--ar", required=True,
                        choices=AR_OUTPUT_CHOICES)
    parser.add_argument("--en-output-type", "--en", required=True,
                        choices=EN_OUTPUT_CHOICES)

    # Paths
    parser.add_argument("-i", "--input-mfa-dir", default='covost_mfa/data',
                        help="Directory containing MFA JSON files")
    parser.add_argument("-o", "--processed-output-dir", default='processed_datasets',
                        help="Parent directory for output data")
    parser.add_argument("--covost-dir", default='./covost_tsv',
                        help="Directory containing covost TSV files")
    parser.add_argument("--durations-path", default='durations_freq_all.pkl',
                        help="Pickle file containing dictionary of durations"
                        " and corresponding frequencies")
    parser.add_argument("--bpe-ar", default='data/training/ar_codes_10k',
                        help="BPE codes for ar side")
    parser.add_argument("--bpe-en", default='data/training/en_codes_10k_mfa',
                        help="BPE codes for en side")
    parser.add_argument("--force-redo", "-f", action='store_true', # if force redo is False and path already exists: don't create datasets again, else make a new directory 
                        help="Redo datasets even if the output directory already exists")
    # For use with factored models, make sure you use the --write-segments-to-file option,
    #  since that will generate some files required for generating the factored data.
    parser.add_argument("--write-segments-to-file", action='store_true', 
                        help="Write unnoise and unbinned segment durations to a separate file")

    # Other arguments
    parser.add_argument("--upsampling", type=int, default=1,
                        help="Upsample examples by this factor (for noisy outputs)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Standard deviation for noise added to durations")
    parser.add_argument("--num-bins", type=int, default=100,
                        help="Number of bins. 0 means no binning.")

    args = parser.parse_args()

    # Read data
    train_tsv, dev_tsv, test_tsv = load_tsv(args.covost_dir)
    codes_ar = codecs.open(args.bpe_ar, encoding='utf-8')
    # codes_ar: the 10k codes files in arabic that will be sent to the BPE
    bpe_ar = BPE(codes_ar) # bpe model that can be used to tokenize german sentences
    codes_en = codecs.open(args.bpe_en, encoding='utf-8')
    bpe_en = BPE(codes_en)

    # durations_path: the pickle file 
    assert os.path.exists(args.durations_path), \
        "Run get_durations_frequencies.py first to get the dictionary of durations" \
        " and how many times each is observed in our data!"
    with open(args.durations_path, 'rb') as f:
        logging.info("Loading durations' frequencies")
        durations_pkl = pickle.load(f)
    """
    Pickle is a module in Python that provides a convenient way 
    to serialize and deserialize objects. 
    Serialization is the process of converting a Python object into a byte stream, 
    and deserialization is the process of reconstructing the original object from the byte stream. 
    """
    if not os.path.exists(args.processed_output_dir): # the created datasets.
        os.makedirs(args.processed_output_dir)

    output_path = os.path.join(args.processed_output_dir, args.ar_output_type)
    if args.num_bins == 0:
        logging.warning("Binning of source segment durations is turned off. "
                        "This is not expected for any of the default models. "
                        "Run with --num-bins > 0 if this was not intentional.")
        output_path += '-unbinned'
    if args.ar_output_type == 'ar-text-noisy-durations':
        if args.noise_std == 0.0:
            logging.error(f"You probably want non-zero noise with {args.ar_output_type}")
            sys.exit(1)
        output_path += str(args.noise_std)
        logging.info(f"Will add noise to speech durations in Ar and upsample by {args.upsampling}.")
    output_path += '-' + args.en_output_type
    logging.info(f"Setting output directory to {output_path}")

    if not args.force_redo and os.path.exists(output_path):
        logging.error(f"Path {output_path} already exists. Run with --force-redo/-f to force overwrite.")
        sys.exit(1)
    else:
        os.makedirs(output_path, exist_ok=True)

    logging.info("Building datasets")
    build_datasets(data_path=args.input_mfa_dir, # mfa file
                   duration_freq=durations_pkl, 
                   ar_output_type=args.ar_output_type,
                   en_output_type=args.en_output_type,
                   output_dir=output_path, # processed_datasets
                   bpe_ar=bpe_ar,
                   bpe_en=bpe_en,
                   tsvs=[train_tsv, dev_tsv, test_tsv], # tsv files (each tsv file key: name, value (english + german sentences))
                   num_bins=args.num_bins, #100
                   upsampling=args.upsampling, #10
                   sd=args.noise_std, #0.1
                   write_segments_to_file=args.write_segments_to_file)
