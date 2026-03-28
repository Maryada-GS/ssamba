import numpy as np
import json
import os
import zipfile
import wget
import torchaudio

#


def get_immediate_subdirectories(a_dir):
    return [
        name
        for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


def get_immediate_files(a_dir):
    return [
        name
        for name in os.listdir(a_dir)
        if os.path.isfile(os.path.join(a_dir, name))
    ]


# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "esc50")
os.makedirs(DATASET_ROOT, exist_ok=True)

base_dir = DATASET_ROOT
output_dir = os.path.join(DATASET_ROOT, "audio_16k")
datafiles_dir = os.path.join(DATASET_ROOT, "datafiles")
os.makedirs(datafiles_dir, exist_ok=True)

file_count = 0
if not os.path.exists(os.path.join(DATASET_ROOT, "audio")):
    print(f"ESC-50 not found at {DATASET_ROOT}, downloading...")
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(DATASET_ROOT, "ESC-50-master.zip")
    wget.download(esc50_url, out=zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_ROOT)
    os.remove(zip_path)
    # Move contents up from ESC-50-master subfolder
    extracted = os.path.join(DATASET_ROOT, "ESC-50-master")
    for item in os.listdir(extracted):
        os.rename(os.path.join(extracted, item), os.path.join(DATASET_ROOT, item))
    os.rmdir(extracted)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_list = get_immediate_files(os.path.join(DATASET_ROOT, "audio"))
    for audio in audio_list:
        input_path = os.path.join(DATASET_ROOT, "audio", audio)
        output_path = os.path.join(output_dir, audio)
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
        torchaudio.save(output_path, waveform, 16000)
        file_count += 1
        if file_count % 2000 == 0:
            print(f"Processed {file_count} files.")

label_set = np.loadtxt(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc_class_labels_indices.csv"),
    delimiter=",",
    dtype="str",
)
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

for fold in [1, 2, 3, 4, 5]:
    meta = np.loadtxt(
        os.path.join(DATASET_ROOT, "meta", "esc50.csv"),
        delimiter=",",
        dtype="str",
        skiprows=1,
    )
    train_wav_list = []
    eval_wav_list = []
    for entry in meta:
        cur_label = label_map[entry[3]]
        cur_path = entry[0]
        cur_fold = int(entry[1])
        cur_dict = {
            "wav": os.path.join(output_dir, cur_path),
            "labels": "/m/07rwj" + cur_label.zfill(2),
        }
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)
    print(
        f"fold {fold}: {len(train_wav_list)} training samples, {len(eval_wav_list)} test samples"
    )

    train_file = os.path.join(datafiles_dir, f"esc_train_data_{fold}.json")
    eval_file = os.path.join(datafiles_dir, f"esc_eval_data_{fold}.json")
    with open(train_file, "w") as f:
        json.dump({"data": train_wav_list}, f, indent=1)
    with open(eval_file, "w") as f:
        json.dump({"data": eval_wav_list}, f, indent=1)

print("Finished ESC-50 Preparation")
