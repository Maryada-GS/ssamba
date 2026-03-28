# -*- coding: utf-8 -*-
# @Time    : 8/13/21 5:31 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : mix_mask_tr_data.py

# combine audioset and librispeech, count how many samples of each of them.

import json
import random


def combine_json(file_list, name="audioset_librispeech960"):
    wav_list = []
    for file in file_list:
        with open(file, "r") as f:
            cur_json = json.load(f)
        cur_data = cur_json["data"]
        print(len(cur_data))
        random.shuffle(cur_data)
        for entry in cur_data:
            entry["labels"] = "/m/09x0r"

        wav_list = wav_list + cur_data
    with open(name + ".json", "w") as f:
        print(len(wav_list))
        json.dump({"data": wav_list}, f, indent=1)


if __name__ == "__main__":
    import os
    DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset")
    audioset_data = os.path.join(DATASET_ROOT, "audioset", "datafiles", "unbal_train_data.json")
    librispeech_data = os.path.join(DATASET_ROOT, "librispeech", "librispeech_tr960_cut.json")
    output_path = os.path.join(DATASET_ROOT, "pretraining", "audioset_librispeech")
    os.makedirs(os.path.join(DATASET_ROOT, "pretraining"), exist_ok=True)
    combine_json([audioset_data, librispeech_data], name=output_path)
