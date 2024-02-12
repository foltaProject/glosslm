"""
Contains the evaluation scripts for comparing predicted and gold IGT
Adapted from SIGMORPHON 2023 Shared Task code
"""

import datasets
import json
import os
import pandas as pd
import re
from jiwer import wer
from typing import List
import evaluate

import click
from torchtext.data.metrics import bleu_score


def strip_gloss_punctuation(glosses: str):
    """Strips any punctuation from gloss string (assuming it is surrounded by spaces)"""
    return re.sub(r"(\s|^)[^\w\s](\s|$)", " ", glosses).strip()


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


def eval_error_rate(pred: List[str], gold: List[str]) -> float:
    prediction = ' '.join(pred)
    reference = ' '.join(gold)
    return wer(reference, prediction)


def eval_avg_error_rate(pred: List[List[str]], gold: List[List[str]]) -> float:
    total_error_rate = 0
    for (sample_pred, sample_gold) in zip(pred, gold):
        total_error_rate += eval_error_rate(sample_pred, sample_gold)
    avg_error_rate = total_error_rate / len(pred)
    return avg_error_rate


def eval_morpheme_glosses(
    pred_morphemes: List[List[str]], gold_morphemes: List[List[str]]
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(pred_morphemes, gold_morphemes)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])
    mer = eval_avg_error_rate(pred_morphemes, gold_morphemes)
    return {"morpheme_level": morpheme_eval, "classes": class_eval, "bleu": bleu, "MER": mer}


def eval_word_glosses(pred_words: List[List[str]], gold_words: List[List[str]]):
    """Evaluates the performance at the morpheme level"""
    word_eval = eval_accuracy(pred_words, gold_words)
    bleu = bleu_score(pred_words, [[line] for line in gold_words])
    wer = eval_avg_error_rate(pred_words, gold_words)
    return {"word_level": word_eval, "bleu": bleu, "WER": wer}


def evaluate_igt(
    pred_path: str,
    test_split: str,
    ft_glottocode: str = None,
    segmented: bool = True,
    verbose=True
):
    """Performs evaluation of a predicted IGT file"""

    def _eval(preds: List[str], gold: List[str]):
        preds = [strip_gloss_punctuation(pred) for pred in preds]
        gold = [strip_gloss_punctuation(g) for g in gold]
        pred_words = [str(pred).split() for pred in preds]
        gold_words = [gloss.split() for gloss in gold]
        # word_eval = eval_accuracy(pred_words, gold_words)

        pred_morphemes = [re.split(r"\s|-", str(pred)) for pred in preds]
        gold_morphemes = [re.split(r"\s|-", gloss) for gloss in gold]

        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(
            predictions=preds, references=gold, word_order=2
        )

        eval_dict = {
            **eval_word_glosses(
                pred_words=pred_words, gold_words=gold_words
            ),
            **eval_morpheme_glosses(
                pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes
            ),
            'chrf': chrf_score['score']
        }
        return eval_dict

    all_eval = {}
    pred_df = pd.read_csv(pred_path).fillna('')
    pred_df = pred_df[pred_df["is_segmented"] == ("yes" if segmented else "no") ]

    if len(pred_df) == 0:
        return {}

    assert test_split in ["test_ID", "test_OOD"]
    # dataset = datasets.load_dataset('lecslab/glosslm-split', split=test_split)
    # if segmented:
    #     dataset = dataset.filter(lambda x: x["is_segmented"] == "yes")
    # else:
    #     dataset = dataset.filter(lambda x: x["is_segmented"] != "no")

    if ft_glottocode is None:
        # assert pred_df["id"].tolist() == dataset["id"]
        # gold = dataset["glosses"]
        all_eval["all"] = _eval(pred_df["pred"], pred_df["gold"])

    for lang in pred_df["glottocode"].unique():
        # lang_dataset = dataset.filter(lambda x: x["glottocode"] == lang)
        lang_preds = pred_df[pred_df["glottocode"] == lang]
        # assert lang_preds["id"].tolist() == lang_dataset["id"]
        # preds = lang_preds["pred"]
        # gold = lang_dataset["glosses"]
        all_eval[lang] = _eval(lang_preds["pred"], lang_preds["gold"])

    results_dir = os.path.dirname(pred_path)
    results_path = f"{results_dir}/{test_split}-{'segmented' if segmented else 'unsegmented'}.json"
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
@click.option(
    "--ft_glottocode",
    help="Glottode of finetuning language",
    type=str,
    required=False,
)
def main(pred: str, test_split: str, ft_glottocode):
    evaluate_igt(pred, test_split, segmented=True, ft_glottocode=ft_glottocode)
    evaluate_igt(pred, test_split, segmented=False, ft_glottocode=ft_glottocode)


if __name__ == "__main__":
    main()
