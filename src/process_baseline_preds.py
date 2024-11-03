"""Converts predictions from any of the shared task submissions to the format used by our eval script"""

from eval import strip_gloss_punctuation
import datasets
import pandas as pd
import os
from typing import cast
import argparse

dataset = cast(datasets.DatasetDict, datasets.load_dataset('lecslab/glosslm-split'))

def read_st_file(path: str):
    """Reads a pred file (probably `lang-test-track#-covered.txt`)"""
    all_glosses = []

    with open(path, 'r') as infile:
        for line in infile:
            line_prefix = line[:2]
            if line_prefix == '\\g':
                preds = line[3:].strip()
                preds = strip_gloss_punctuation(preds)
                all_glosses.append(preds)
    return all_glosses

iso_to_glotto = {
    'arp': 'arap1274',
    "git": "gitx1241",
    "lez": "lezg1247",
    "ntu": "natu1246",
    "nyb": "nyan1302",
    "ddo": "dido1241",
    "usp": "uspa1245",
}

def create_df(preds, isocode, segmented):
    assert isocode in iso_to_glotto
    glottocode = iso_to_glotto[isocode]
    id_or_ood = "ID" if glottocode in ["arap1274", "dido1241", "uspa1245"] else "OOD"
    gold = dataset[f'test_{id_or_ood}'].filter(lambda row: row['glottocode'] == glottocode and row['is_segmented'] == segmented)

    assert len(gold) == len(preds), f"Length mismatch: {len(gold)} expected, {len(preds)} observed"

    gold_glosses = [strip_gloss_punctuation(g) for g in gold["glosses"]]
    return pd.DataFrame({
        "id": gold["id"],
        "glottocode": gold["glottocode"],
        "is_segmented": gold["is_segmented"],
        "pred": preds,
        "gold": gold_glosses,
    }), id_or_ood

def process_folder(dirpath):
    data = {}

    for file in os.listdir(dirpath):
        if not file.endswith('txt'):
            continue
        filename = os.fsdecode(file)
        path = os.path.join(dirpath, filename)
        comps = filename.split("-")
        preds = read_st_file(path)
        df, id_or_ood = create_df(preds, comps[0], 'yes' if comps[2] == 'track2' else 'no')
        if comps[0] in data:
            data[comps[0]] = pd.concat([data[comps[0]], df])
        else:
            data[comps[0]] = df

    id = pd.concat([data['arp'], data['ddo'], data['usp']])
    id.to_csv(os.path.join(dirpath, 'test_ID-preds.csv'))
    ood = pd.concat([data['git'], data['lez'], data['ntu'], data['nyb']])
    ood.to_csv(os.path.join(dirpath, 'test_OOD-preds.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    args = parser.parse_args()
    process_folder(args.file)