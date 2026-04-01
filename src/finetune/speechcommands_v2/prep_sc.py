import json
import os

import numpy as np
import wget

DATA_DIR = "./data"
SC_DIR = f"{DATA_DIR}/speech_commands_v0.02"
SC_URL = (
    "https://storage.googleapis.com/download.tensorflow.org"
    "/data/speech_commands_v0.02.tar.gz"
)
DATAFILES_DIR = f"{DATA_DIR}/datafiles"
SPLIT_TO_JSON = {
    "train": f"{DATAFILES_DIR}/speechcommand_train_data.json",
    "validation": f"{DATAFILES_DIR}/speechcommand_valid_data.json",
    "testing": f"{DATAFILES_DIR}/speechcommand_eval_data.json",
}


def download_dataset():
    print("Downloading Speech Commands v0.02...")
    os.makedirs(DATA_DIR, exist_ok=True)
    tarball = f"{DATA_DIR}/speech_commands_v0.02.tar.gz"
    wget.download(SC_URL, out=DATA_DIR)
    os.makedirs(SC_DIR, exist_ok=True)
    os.system(f"tar -xzf {tarball} -C {SC_DIR}")
    n_files = sum(len(files) for _, _, files in os.walk(SC_DIR))
    print(f"\n{n_files} files extracted.")


def build_train_list():
    def subdirs(d):
        return [n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))]

    def files(d):
        return [n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))]

    with open(f"{SC_DIR}/validation_list.txt") as f:
        val_list = set(f.readlines())
    with open(f"{SC_DIR}/testing_list.txt") as f:
        test_list = set(f.readlines())

    excluded = val_list | test_list
    all_samples = [
        f"{cmd}/{sample}\n"
        for cmd in subdirs(SC_DIR)
        if cmd != "_background_noise_"
        for sample in files(f"{SC_DIR}/{cmd}")
    ]
    train_list = [x for x in all_samples if x not in excluded]

    with open(f"{SC_DIR}/train_list.txt", "w") as f:
        f.writelines(train_list)


def build_label_map():
    label_set = np.loadtxt(
        f"{DATA_DIR}/speechcommands_class_labels_indices.csv",
        delimiter=",",
        dtype="str",
    )
    return {eval(row[2]): row[0] for row in label_set[1:]}


def build_json_files(label_map):
    os.makedirs(DATAFILES_DIR, exist_ok=True)
    abs_sc_dir = os.path.join(os.path.abspath(os.getcwd()), "data/speech_commands_v0.02")

    for split, out_path in SPLIT_TO_JSON.items():
        with open(f"{SC_DIR}/{split}_list.txt") as f:
            filelist = f.readlines()

        wav_list = [
            {
                "wav": f"{abs_sc_dir}/{f.strip()}",
                "labels": "/m/spcmd" + label_map[f.split("/")[0]].zfill(2),
            }
            for f in filelist
        ]

        with open(out_path, "w") as f:
            json.dump({"data": wav_list}, f, indent=1)

        print(f"{split}: {len(wav_list)} samples -> {out_path}")


if __name__ == "__main__":
    if not os.path.exists(SC_DIR):
        download_dataset()

    if not os.path.exists(f"{SC_DIR}/train_list.txt"):
        build_train_list()

    label_map = build_label_map()

    if not os.path.exists(DATAFILES_DIR):
        build_json_files(label_map)
        print("Done.")
