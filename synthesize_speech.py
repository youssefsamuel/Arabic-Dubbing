import os
import argparse
import pickle
import json
import re
import subprocess
from subprocess import Popen, PIPE, DEVNULL, check_call
import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from tqdm import tqdm
import numpy as np
import torch
from subword_nmt.apply_bpe import BPE
from pydub import AudioSegment

from preprocessing_scripts import Bin
from arabert import ArabertPreprocessor

"""
Summary of the whole file:

1- Voice Activity Detection (VAD): Detects speech segments and pauses in the input audio using the Silero VAD model.
2- Text Processing and Translation: Processes the source text, applies Byte Pair Encoding (BPE), and translates the text using the Sockeye model.
3- FastSpeech2 Synthesis: Uses the FastSpeech2 model to synthesize phoneme and duration outputs.
4- Audio and Video Generation: Combines the synthesized audio segments with the original video, adding pauses and creating final dubbed videos.
5- Cleanup: Removes some files.
"""

SEGMENT_DURATION_SEPARATOR = ' <||> ' 
FACTOR_DELIMITER = '|' #used to separate different factors or components in a string, likely in the context of the Sockeye translation factors.
EOW = '<eow>' 
PAUSE = '[pause]'
SHIFT= '<shift>'
SAMPLING_RATE = 22050
HOP_LENGTH = 256
arabert_preprocessor = ArabertPreprocessor(
    model_name= "aubmindlab/bert-base-arabertv2",
    keep_emojis = False,
    remove_html_markup = True,
    replace_urls_emails_mentions = True,
    strip_tashkeel = True,
    strip_tatweel = True,
    insert_white_spaces = False,
    remove_non_digit_repetition = True,
    replace_slash_with_dash = True,
    map_hindi_numbers_to_arabic = True,
    apply_farasa_segmentation = True
)

def remove_punc(sentence):
    cleaned_sentence = re.sub(r'[،؛:<>()«»\-…]', '', sentence)
    return cleaned_sentence

def get_sorted_audio_files(data_dir):
    """
    Get all the wav files in the directory named `*.Y.wav` and return them sorted numerically by `Y`

    input example:
    common_voice_en_511071.wav
    common_voice_en_67895.wav
    common_voice_en_217222.wav
    common_voice_en_19191100.wav
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    files = sorted(files, key=lambda f: int(f.split('.')[-2]))
    return [os.path.join(args.data_dir, "subset" + args.subset, f) for f in files]


class SileroVad:
    """
    VAD is a technique used in speech processing to identify regions of an audio signal that contain speech and distinguish them from non-speech regions (such as silence or background noise)
    """
    def __init__(self):
        self.sampling_rate = 16000
        """
        self.model: Loads the Silero VAD model using PyTorch's torch.hub.load function. The model is retrieved from the specified repository (snakers4/silero-vad).
        """
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           onnx=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        """
        self.get_speech_timestamps: References a function for obtaining start and end timestamps of speech segments.
        self.read_audio: References a function for reading audio files.
        """

    def get_timestamps(self, wav_file):
        """
        Get list of start and end timestamps of speech segments and lengths of pauses
        ____________________________________________________________________________________________________
        It takes a WAV file as input and returns a list of start and end timestamps of speech segments and lengths of pauses.
        + Uses the Silero VAD model to process the audio and identify speech segments.
        + Optionally visualizes probabilities and sets a threshold for speech detection.
        """
        wav = self.read_audio(wav_file, sampling_rate=self.sampling_rate)
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate,
                                                       min_silence_duration_ms=300, visualize_probs=False,
                                                       threshold=0.3, return_seconds=True)
        """
        "speech_timestamps": an aobject that will contain a list of dictionaries where each dictionary represents a detected speech segment. Each dictionary typically includes information about the start and end times of the speech segment. Here's an explanation of the key information:
        """
        pauses = []
        if len(speech_timestamps) > 1:
            for i, pair in enumerate(speech_timestamps): #enumerate get both the index i and the corresponding pair.
                if i == 0:
                    previous_start, previous_end = pair["start"], pair["end"]
                    #No pause is calculated or appended in the first iteration.
                else:
                    current_start, current_end = pair["start"], pair["end"]
                    pause = current_start - previous_end
                    pauses.append(round(pause, 3)) #round to 3 decimal places
                    previous_start, previous_end = pair["start"], pair["end"]
                    #previous_start and previous_end are updated for the next iteration.
        return speech_timestamps, pauses


class SockeyeTranslator:
    """
    Wrapper around sockeye-translate command line to translate lines one at a time without reloading model
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specified Sockeye model checkpoint {model_path} does not exist")
        sockeye_command = ['python', '-u', '-m', 'sockeye.translate',
                           '--models', os.path.dirname(model_path),
                           '--checkpoints', os.path.basename(model_path).split('.')[-1],
                           '-b', '5',
                           '--batch-size', '1',
                           '--output-type', 'translation_with_factors',
                           '--max-output-length', '768',
                           '--force-factors-stepwise', 'frames', 'total_remaining', 'segment_remaining', 'pauses_remaining',
                           '--json-input'
                          ]
        """
        This sockeye_command is later used in the script with subprocess.Popen to execute the Sockeye translation command. The command will use the specified Sockeye model to perform translation with certain settings and output the results with additional factors related to frames, total remaining, segment remaining, and pauses remaining.
        """
        logging.info(f"Running Sockeye command: {' '.join(sockeye_command)}")
        self.sockeye_process = Popen(sockeye_command, stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=os.environ,
                                     text=True, encoding='utf-8', universal_newlines=True, bufsize=1)

    def translate_line(self, line, segments):
        """
        Send one line to sockeye-translate and get back the translation
        """
        json_line = self.make_json_input(line, segments)
        logging.debug(f"Sending input to sockeye-translate: {json_line}")
        self.sockeye_process.stdin.write(json_line + '\n')
        self.sockeye_process.stdin.flush()
        return self.sockeye_process.stdout.readline()
        """
        Parameters:
        line: The input text (presumably a sentence) to be translated.
        segments: The segments or durations associated with the input text.

        Actions:
        Calls the make_json_input method to create a JSON-formatted input based on the input text and segments.
        Logs the input being sent to Sockeye-translate for debugging purposes.
        Writes the JSON-formatted input to the standard input of the Sockeye process.
        Flushes the standard input buffer.
        Returns the translation result read from the standard output of the Sockeye process.
        """
    def make_json_input(self, line, segment_durations):
        """
        Create the JSON-formatted input for target factor prefixes etc.
        """
        input_dict = {
            'text': line,
            'target_prefix': SHIFT,
            'target_prefix_factors': ['0',
                                      str(sum(segment_durations)),
                                      str(segment_durations[0]),
                                      str(len(segment_durations) - 1)
                                     ],
            'target_segment_durations': segment_durations,
            'use_target_prefix_all_chunks': 'false'
        }
        return json.dumps(input_dict, ensure_ascii=False).replace('"false"', 'false')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-dir", type=str,
                        default=os.path.join("/content", "Arabic-Dubbing", "data", "test"),
                        help="Directory containing the audio files. Inside this directory, the files should be in subsetX/*.Y.wav, "
                             "where sorting numerically by the Y field will give us the files in the same order as the transcript file. "
                             "This is already true for the test set subsets.")
    parser.add_argument("--source-text", type=str,
                        help="File containing the source Arabic text. Defaults to using subsetX.ar for the test set subsets.")
    parser.add_argument("--subset", choices=['1', '2'], type=str, required=True,
                        help="Which test set subset to generate dubs for.")
    parser.add_argument("--sockeye-model", type=str,
                        default=os.path.join("/content", "Arabic-Dubbing", "models", "sockeye", "trained_baselines", "baseline_factored_noised0.1", "model", "params.00065"),
                        help="Path to a Sockeye model checkpoint.")
    parser.add_argument("--fastspeech-dir", type=str,
                        default=os.path.join("/content", "Arabic-Dubbing", "third_party", "FastSpeech2"),
                        help="Path to the FastSpeech2 directory.")
    parser.add_argument("--bpe-ar", type=str,
                        default=os.path.join("/content", "Arabic-Dubbing", "data", "training", "ar_codes_20k"),
                        help="BPE codes for German source text.")
    parser.add_argument("--durations-freq", type=str,
                        default=os.path.join("/content", "Arabic-Dubbing", "durations_freq_all.pkl"),
                        help="Path to durations_freq_all.pkl")
    parser.add_argument("--output-video-dir", type=str,
                        help="Directory to write final dubbed videos.")
    parser.add_argument("--genderfile", type=str,
                        help="File to containing the gender of each speaker")
    parser.add_argument("--join-mode", type=str, choices=['match_pause', 'match_start'], default='match_start',
                        help="When joining segments together to create final clip:\n"
                             "match_pause: Pause lengths match source.\n"
                             "match_start: Try to match segment start times. May not match exactly if segments are too long.\n")

    args = parser.parse_args()
    gender_file_path = args.genderfile
    gender_list = []
    #Reading the genders
    with open(gender_file_path, "r") as f:
        # Read each line in the file
        for line in f:
            string = line[:-1]
            print(string)
            # Check if the line contains "male"
            if string == "male":
                # If it does, add 0 to the list
                gender_list.append(0)
            else:
                # If it doesn't, add 1 to the list
                gender_list.append(1)

    # Default source text is `subsetX.ar`
    if args.source_text is None:
        args.source_text = os.path.join(args.data_dir, "subset" + args.subset + '.ar')

    # Do not change: These directories are fixed for FastSpeech2 trained on LibriTTS data
    output_dir = os.path.join(args.fastspeech_dir, 'output', 'result', 'LibriTTS')
    durations_dir = os.path.join(args.fastspeech_dir, 'preprocessed_data', 'LibriTTS', 'duration')

    # Default directory is a subdirectory of the input audio directory called `dubbed`
    if args.output_video_dir is None:
        args.output_video_dir = os.path.join(args.data_dir, "subset" + args.subset, 'dubbed')
    os.makedirs(args.output_video_dir, exist_ok=True)

    # Get audio files and lines of text - aligned with each other
    audio_files = get_sorted_audio_files(os.path.join(args.data_dir, "subset" + args.subset))
    with open(args.source_text) as f_src:
        src_text = f_src.readlines()
    assert len(audio_files) == len(src_text), "Number of audio files and number of lines in source text did not match."
    """
    Processing Audio Files and Source Text:
    Obtains a sorted list of audio files (audio_files) from the specified directory.
    Reads lines of source text (src_text) from the provided source text file.
    Asserts that the number of audio files and lines in the source text match.

    BPE Processing and Duration Binning:
    Creates a BPE processor (bpe_ar) using the provided BPE codes.
    Loads duration frequencies for binning using a Bin instance.
    Initializes a SileroVad instance for voice activity detection.
    Initializes a SockeyeTranslator instance for translation.

    Translation and Processing:
    Iterates over audio files, extracting speech timestamps and pauses using Silero VAD.
    Applies binning to segment durations and performs BPE encoding on source text segments.
    Translates the BPE-encoded text using Sockeye, writing the output to files.
    Processes the translation output, removing certain tokens and splitting on [pause].

    FastSpeech2 Synthesis and Audio/Video Generation:
    Runs FastSpeech2 on the generated phoneme and duration outputs.
    Reconstructs final audio segments, considering pauses and adjusting if required.
    Exports the final audio segments to WAV files.
    Embeds the audio onto video if corresponding video files exist.
    """
    # Create BPE processor
    bpe_ar = BPE(open(args.bpe_ar))

    # Load duration frequencies for binning
    with open(args.durations_freq, 'rb') as f:
        durations_freq = pickle.load(f)
    bin_instance = Bin(durations_freq, n=100)

    silero_vad = SileroVad()

    sockeye_translator = SockeyeTranslator(args.sockeye_model)

    speech_timestamps = []
    pauses = []
    hyp_segments = []
    gender_index = 0
    logging.info(f"Generating translated phoneme and duration outputs")
    with open(os.path.join(output_dir, 'subset' + args.subset + '.en.output'), 'w') as f_out, \
         open(os.path.join(output_dir, 'subset' + args.subset + '.en.fs2_inp'), 'w') as f_fs2_inp:
        for idx, audio_file in tqdm(enumerate(audio_files)):
            if gender_list[gender_index] == 0:
                speaker_id = "6115"
            else:
                speaker_id = "3816"
            gender_index += 1
            duration_frames = []
            vad = silero_vad.get_timestamps(audio_file)
            speech_timestamps.append(vad[0])
            pauses.append(vad[1])
            for timestamp in speech_timestamps[idx]:
                duration_frames.append(int(np.round(timestamp["end"] * SAMPLING_RATE / HOP_LENGTH) - np.round(timestamp["start"] * SAMPLING_RATE / HOP_LENGTH)))

            # BPE each segment and append segment durations bins
            bins = bin_instance.find_bin(speech_durations=duration_frames)
            sentence_segments = src_text[idx].split('[pause]')
            sentence_bpe = [remove_punc(bpe_ar.process_line(arabert_preprocessor.preprocess(sentence_seg.strip())).replace(' +', '').replace('+ ', '')) for sentence_seg in sentence_segments]
            sentence_bped_str = " ".join(sentence_bpe) + SEGMENT_DURATION_SEPARATOR + " ".join(bins)

            # Get translation from Sockeye
            hyp = sockeye_translator.translate_line(sentence_bped_str, duration_frames)
        

            f_out.write(hyp)
            # Remove `<eow>` and `<shift>` tokens
            hyp = " ".join([t for t in hyp.split() if t.split(FACTOR_DELIMITER)[0] not in [EOW, SHIFT]])
            # Split upon `[pause]`
            hyp_segments.append(re.split(r"\s*" + re.escape(PAUSE) + r"\|[^\s]+\s*", hyp))

            # Process each segment separately. Will later be joined with pauses again
            for seg_idx, hyp_segment in enumerate(hyp_segments[idx]):
                seg_fs2_id = f"subset{args.subset}-{idx+1}-{seg_idx+1}"
                # Write input in FastSpeech2 format

                f_fs2_inp.write(seg_fs2_id + '|' + str(speaker_id) + '|{')
                f_fs2_inp.write(' '.join([t.split(FACTOR_DELIMITER)[0] for t in hyp_segment.split()]))
                f_fs2_inp.write('}|\n')
                # Save durations to file for FastSpeech2 to read
                np.save(os.path.join(durations_dir, "LibriTTS-duration-" + seg_fs2_id + '.npy'),
                        np.array([int(t.split(FACTOR_DELIMITER)[1]) for t in hyp_segment.split()]))

    
    # FastSpeech2 doesn't work unless you're in the right directory due to relative paths in their configs.
    os.chdir(args.fastspeech_dir)
    logging.info("Running FastSpeech2 on phoneme and duration outputs")


    check_call(f"`dirname ${{CONDA_PREFIX}}`/fastspeech2/bin/python {os.path.join(args.fastspeech_dir, 'synthesize.py')}  "
               f"--source {os.path.join(output_dir, 'subset' + args.subset + '.en.fs2_inp')}  --restore_step 800000 --mode batch "
               f"-p {os.path.join(args.fastspeech_dir, 'config/LibriTTS/preprocess.yaml')} "
               f"-m {os.path.join(args.fastspeech_dir, 'config/LibriTTS/model.yaml')} "
               f"-t {os.path.join(args.fastspeech_dir, 'config/LibriTTS/train.yaml')} >/dev/null",
               shell=True)

    logging.info("Reconstructing final audio segments")
    # Re-construct audio from the pieces and add pauses
    for idx, audio_file in tqdm(enumerate(audio_files)):
        # Counting pauses for re-insertion
        num_pauses_hyp = len(hyp_segments[idx]) - 1

        # Add silence in the beginning (if VAD detected speech after 0.0s in the beginning of the video)
        if speech_timestamps[idx][0]['start'] > 0.0:
            pauses_start = speech_timestamps[idx][0]['start']
        else:
            pauses_start = 0.0
        audio = [AudioSegment.silent(duration=pauses_start * 1000)]

        for seg_idx, hyp_segment in enumerate(hyp_segments[idx]):
            # Join audio segments, adding pauses if needed
            seg_fs2_id = f"subset{args.subset}-{idx+1}-{seg_idx+1}"
            audio.append(AudioSegment.from_file(os.path.join(output_dir, seg_fs2_id + '.wav'), format="wav"))
            if seg_idx < num_pauses_hyp and seg_idx < len(pauses[idx]):
                pause_mseconds = pauses[idx][seg_idx] * 1000
                if args.join_mode == 'match_start':
                    # Adjust the pause by the difference between original and generated audio (without going below zero)
                    orig_seg_mseconds = (speech_timestamps[idx][seg_idx]['end'] - speech_timestamps[idx][seg_idx]['start']) * 1000
                    pause_mseconds -= len(audio[-1]) - orig_seg_mseconds
                    pause_mseconds = max(0, pause_mseconds)
                audio.append(AudioSegment.silent(duration=pause_mseconds))

        # Concatenate all audio segments together
        audio_final = sum(audio)
        audio_path = os.path.join(args.output_video_dir, os.path.basename(audio_file).replace('.wav', '.en.wav'))
        audio_final.export(audio_path, format="wav")

        # Embed wav onto video
        video_path = audio_path.replace('.wav', '.mp4')
        if os.path.exists(audio_file.replace('.wav', '.mp4')):
            check_call(f"ffmpeg -i {audio_file.replace('.wav', '.mp4')} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy {video_path} -hide_banner -loglevel error -y", shell=True)
        elif os.path.exists(audio_file.replace('.wav', '.mov')):
            check_call(f"ffmpeg -i {audio_file.replace('.wav', '.mov')} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy {video_path} -hide_banner -loglevel error -y", shell=True)
        else:
            logging.error(f"Could not find video at {audio_file.replace('.wav', '.{mp4,mov}')}")

    logging.info("Cleaning up intermediate files")
    # Remove intermediate files
    check_call(f"rm -f {output_dir}/*.wav", shell=True)
    check_call(f"rm -f {output_dir}/*.png", shell=True)
    check_call(f"rm -f {args.output_video_dir}/*.wav", shell=True)
    check_call(f"rm -f {durations_dir}/*", shell=True)

    logging.info(f"Dub generation complete. Output videos can be found in {args.output_video_dir}")
