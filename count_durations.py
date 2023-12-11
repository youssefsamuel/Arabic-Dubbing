
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
import sys
from statistics import mean
from preprocessing_scripts import Bin

ref = sys.argv[1]
hyp = sys.argv[2]

path = "./"
correct_dur_pred, incorrect_dur_pred = 0, 0
durations_hyp = []
durations_ref = []

durations_path = './durations_freq_all.pkl'
if os.path.exists(durations_path):
    with open(durations_path, 'rb') as f:
        duration_freq = pickle.load(f)
        print("loaded durations' freq!")
        bin_instance = Bin(duration_freq, n=100)

with open(os.path.join(path, ref)) as f:
    lines = f.readlines()
    sentences = []
    for line in lines: #Each line would represent a sentence in the ref file (ex: /processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.en)
        # Sample line: B 2 AH1 8 T 3 <eow> IH1 5 F 10 <eow> Y 4 UW1 7 <eow> HH 1 AE1 11 V 7 <eow> T 7 AH0 4 <eow> G 8 OW1 14 <eow>
        # B 9 IH0 4 F 11 AO1 20 R 3 <eow> DH 4 EH1 8 N 16 <eow> [pause] G 6 OW1 12 <eow> AA1 9 N 3 <eow> IH1 3 N 13 <eow> sp 16 P 2 ER0 13 S 10 UW1 10 T 9 
        # <eow> AH0 4 V 5 <eow> sp 5 Y 4 UH1 5 R 4 <eow> D 12 R 2 IY1 5 M 16 <eow> sp 19
        line = " ".join(line.split("<eow>")) # Replaces the <eow> with spaces
        #Processed line (after replacing <eow>): B 2 AH1 8 T 3   IH1 5 F 10   Y 4 UW1 7   HH 1 AE1 11 V 7   T 7 AH0 4   G 8 OW1 14   B 9 IH0 4 F 11 AO1 20 R 3
        #    DH 4 EH1 8 N 16   [pause] G 6 OW1 12   AA1 9 N 3   IH1 3 N 13   sp 16 P 2 ER0 13 S 10 UW1 10 T 9   AH0 4 V 5   sp 5 Y 4 UH1 5 R 4   D 12 R 2 IY1 5 M 16   sp 19
        line_segments = line.split('[pause]') 
        # Segments after splitting by [pause]: ['B 2 AH1 8 T 3   IH1 5 F 10   Y 4 UW1 7   HH 1 AE1 11 V 7   T 7 AH0 4   G 8 OW1 14   B 9 IH0 4 F 11 AO1 20 R 3   DH 4 EH1 8 N 16   ',
        #  ' G 6 OW1 12   AA1 9 N 3   IH1 3 N 13   sp 16 P 2 ER0 13 S 10 UW1 10 T 9   AH0 4 V 5   sp 5 Y 4 UH1 5 R 4   D 12 R 2 IY1 5 M 16   sp 19']
        counter_line = []

        for segment in line_segments:
            counter_segment = 0
            durations = segment.split()[1:][::2]
            # Durations extracted from segment: ['2', '8', '3', '5', '10', '4', '7', '1', '11', '7', '7',
            # '4', '8', '14', '9', '4', '11', '20', '3', '4', '8', '16']
            for dur in durations:
                counter_segment += int(dur)
            counter_line.append(counter_segment)
            # Counter segment after processing: 166 (Sum of the durations in that segment)
        durations_ref.append(counter_line)
        #Final durations_ref: [[166, 187]] for the 2 segments  

with open(os.path.join(path, hyp)) as f:
    lines = f.readlines()
    sentences = []
    for i, line in enumerate(lines):
        line = " ".join(line.split("<eow>"))
        line_segments = line.split('[pause]')
        counter_line = []
        for segment in line_segments:
            counter_segment = 0
            segments = segment.split()
            num = 0
            for seg in segments:
                if num % 2 != 0:  # odd
                    try:
                        counter_segment += int(seg)
                        # pdb.set_trace()
                    except:
                        print(segments)
                        print("Duration is {} for line {}".format(seg, i))
                        num += 1
                num += 1

            counter_line.append(counter_segment)

        durations_hyp.append(counter_line)

errors = []
scores = []
count = 0
count_right = 0
one_pause_or_more = 0
differences = []
sampling_rate = 22050
hop_length = 256
large_dif = 0
small_dif = 0
threshold_in_frames = int(0.3 * sampling_rate/hop_length)
# 0.3s corresponds to 25 frames

for i in range(len(durations_ref)):
    # if len(durations_ref[i]) > 0:
        one_pause_or_more +=1
        if len(durations_ref[i]) == len(durations_hyp[i]):
            for j in range(len(durations_ref[i])):
                if durations_ref[i][j] == 0:
                    temp = 1
                else:
                    temp = durations_ref[i][j]
                # abs_diff: Difference between segments length in hypothesis and reference
                abs_diff = abs(durations_hyp[i][j] - temp)  
                # error: relative difference with respect to the reference
                errors.append(abs_diff/temp)
                # score: value less than 1, that is the ratio of the reference to the hypothesis segment lengths
                if durations_hyp[i][j] < temp:
                    scores.append(durations_hyp[i][j]/temp)
                else:
                    scores.append(temp/durations_hyp[i][j])
                differences.append(abs_diff)
                if abs_diff >= threshold_in_frames:
                    # If difference greater than or equal to 25 frame
                    large_dif += 1
                else:
                    small_dif += 1
            # Counter for the number of sentences having same number of segments between hypothesis and reference
            count_right += 1
        else:
            # Number of segments in reference and hypothesis are not equal
            # Counter for the number of sentences having different number of segments between hypothesis and reference
            count += 1
            for j in range(len(durations_ref[i])):
                if durations_ref[i][j] == 0:
                    temp = 1
                else:
                    temp = durations_ref[i][j]
                # abs_diff: Maximum of (Segments length in reference, 1)
                abs_diff = abs(0 - temp)
                # error: Always equal to 1
                errors.append(abs_diff/temp)
                scores.append(0.0)
                # Always append 35 frames to the differences
                differences.append(threshold_in_frames + 10)
                large_dif += 1

# print(count)
# print(count_right)
print("Metric 1 is {}".format(1 - mean(errors))) # Accuracy (Ceiled by the value 1, the higher the better)
print("Metric 2 is {}".format(mean(scores))) # Metric estimating how close the segments length are to each other (Ceiled by the value 1, the higher the better)
print("Metric 3 is {}".format(mean(differences))) # Average of the differences between the segments length (Have no bounds)

print("Segments with diff > 0.3s: {}".format(large_dif/(small_dif + large_dif))) # Ratio of the large differences with respect to all the differences 
print("How many sentences have 1 or more pauses: {}".format(one_pause_or_more)) # one_pause_or_more counts the number of sentences in the file

print("Predicted wrong number of pauses in {} out of {} sentences".format(count, count + count_right)) # Stating the number of sentences that had problems in the number of segments
