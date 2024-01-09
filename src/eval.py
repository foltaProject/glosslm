"""
Contains the evaluation scripts for comparing predicted and gold IGT
Adapted from SIGMORPHON 2023 Shared Task code
"""

import datasets
import json
import os
import pandas as pd
import re
from typing import List

import click
from torchtext.data.metrics import bleu_score


def eval_accuracy(pred: List[List[str]], gold: List[List[str]]) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be
    in the correct position in the list."""
    total_correct_predictions = 0
    total_tokens = 0
    summed_accuracies = 0

    for entry_pred, entry_gold, i in zip(pred, gold, range(len(gold))):
        entry_correct_predictions = 0

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if (
                token_index < len(entry_pred)
                and entry_pred[token_index] == entry_gold[token_index]
                and entry_pred[token_index] != "[UNK]"
            ):
                entry_correct_predictions += 1

        entry_accuracy = entry_correct_predictions / len(entry_gold)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_tokens += len(entry_gold)

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    return {"average_accuracy": average_accuracy, "accuracy": overall_accuracy}


def eval_stems_grams(pred: List[List[str]], gold: List[List[str]]) -> dict:
    perf = {
        "stem": {"correct": 0, "pred": 0, "gold": 0},
        "gram": {"correct": 0, "pred": 0, "gold": 0},
    }

    for entry_pred, entry_gold in zip(pred, gold):
        for token_index in range(len(entry_gold)):
            # We can determine if a token is a stem or gram by checking if it is
            # all uppercase
            token_type = "gram" if entry_gold[token_index].isupper() else "stem"
            perf[token_type]["gold"] += 1

            if token_index < len(entry_pred):
                pred_token_type = (
                    "gram" if entry_pred[token_index].isupper() else "stem"
                )
                perf[pred_token_type]["pred"] += 1

                if entry_pred[token_index] == entry_gold[token_index]:
                    # Correct prediction
                    perf[token_type]["correct"] += 1

    stem_perf = {
        "prec": 0
        if perf["stem"]["pred"] == 0
        else perf["stem"]["correct"] / perf["stem"]["pred"],
        "rec": perf["stem"]["correct"] / perf["stem"]["gold"],
    }
    if (stem_perf["prec"] + stem_perf["rec"]) == 0:
        stem_perf["f1"] = 0
    else:
        stem_perf["f1"] = (
            2
            * (stem_perf["prec"] * stem_perf["rec"])
            / (stem_perf["prec"] + stem_perf["rec"])
        )

    gram_perf = {
        "prec": 0
        if perf["gram"]["pred"] == 0
        else perf["gram"]["correct"] / perf["gram"]["pred"],
        "rec": perf["gram"]["correct"] / perf["gram"]["gold"],
    }
    if (gram_perf["prec"] + gram_perf["rec"]) == 0:
        gram_perf["f1"] = 0
    else:
        gram_perf["f1"] = (
            2
            * (gram_perf["prec"] * gram_perf["rec"])
            / (gram_perf["prec"] + gram_perf["rec"])
        )
    return {"stem": stem_perf, "gram": gram_perf}


def eval_morpheme_glosses(
    pred_morphemes: List[List[str]], gold_morphemes: List[List[str]]
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(pred_morphemes, gold_morphemes)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])
    return {"morpheme_level": morpheme_eval, "classes": class_eval, "bleu": bleu}


def eval_word_glosses(pred_words: List[List[str]], gold_words: List[List[str]]):
    """Evaluates the performance at the morpheme level"""
    word_eval = eval_accuracy(pred_words, gold_words)
    bleu = bleu_score(pred_words, [[line] for line in gold_words])
    return {"word_level": word_eval, "bleu": bleu}


def evaluate_igt(
    pred_path: str,
    test_split: str,
    segmented: bool = True,
    verbose=True
):
    """Performs evaluation of a predicted IGT file"""

    def _eval(preds: List[str], gold: List[str]):
        pred_words = [str(pred).split() for pred in preds]
        gold_words = [gloss.split() for gloss in gold]
        word_eval = eval_accuracy(pred_words, gold_words)

        pred_morphemes = [re.split(r"\s|-", str(pred)) for pred in preds]
        gold_morphemes = [re.split(r"\s|-", gloss) for gloss in gold]

        eval_dict = {
            "word_level": word_eval,
            **eval_morpheme_glosses(
                pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes
            ),
        }
        return eval_dict

    all_eval = {}
    pred_df = pd.read_csv(pred_path)
    if segmented:
        pred_df = pred_df[pred_df["is_segmented"] == "yes"]
    else:
        pred_df = pred_df[pred_df["is_segmented"] != "yes"]
    preds = pred_df["pred"]

    assert test_split in ["test_ID", "test_OOD"]
    dataset = datasets.load_dataset('lecslab/glosslm-split', split=test_split)
    if segmented:
        dataset = dataset.filter(lambda x: x["is_segmented"] == "yes")
    else:
        dataset = dataset.filter(lambda x: x["is_segmented"] != "no")
    assert pred_df["ID"].tolist() == dataset["ID"]
    gold = dataset["glosses"]
    all_eval["all"] = _eval(preds, gold)

    for lang in pred_df["glottocode"].unique():
        lang_dataset = dataset.filter(lambda x: x["glottocode"] == lang)
        lang_preds = pred_df[pred_df["glottocode"] == lang]
        assert lang_preds["ID"].tolist() == lang_dataset["ID"]
        preds = lang_preds["pred"]
        gold = lang_dataset["glosses"]
        all_eval[lang] = _eval(preds, gold)

    results_dir = os.path.dirname(pred_path)
    if segmented:
        results_path = f"{results_dir}/{test_split}-segmented.json"
    else:
        results_path = f"{results_dir}/{test_split}-unsegmented.json"
    with open(results_path, 'x') as outfile:
        json.dump(all_eval, outfile, sort_keys=True, indent=4)

    if verbose:
        print(test_split)
        print(json.dumps(all_eval, sort_keys=True, indent=4))

    return all_eval


@click.command()
@click.option(
    "--pred",
    help="File containing predicted IGT",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--test_split",
    help="Test split to be evaluated",
    type=str,
    required=True,
)
def main(pred: str, test_split: str):
    evaluate_igt(pred, test_split, segmented=True)
    evaluate_igt(pred, test_split, segmented=False)


if __name__ == "__main__":
    main()
