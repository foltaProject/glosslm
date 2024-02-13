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

import click
from torchtext.data.metrics import bleu_score


def get_lang(glottocode):
    lang_map = {
        "arap1274": "arp",
        "gitx1241": "git",
        "dido1241": "ddo",
        "uspa1245": "usp",
        "nyan1302": "nyb",
        "natu1246": "ntu",
        "lezg1247": "lez",
    }
    return lang_map[glottocode]


def strip_gloss_punctuation(glosses: str):
    """Strips any punctuation from gloss string (assuming it is surrounded by spaces)"""
    return re.sub(r"(\s|^)[^\w\s](\s|$)", " ", glosses).strip()


def eval_accuracy(segs: List[List[str]], pred: List[List[str]], gold: List[List[str]], vocab: dict, morph: bool = True) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be
    in the correct position in the list."""
    total_correct_predictions = 0
    total_correct_iv_preds = 0
    total_correct_oov_preds = 0
    total_tokens = 0
    total_iv_tokens = 0
    summed_accuracies = 0

    if segs is None:
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

    for entry_segs, entry_pred, entry_gold, i in zip(segs, pred, gold, range(len(gold))):
        if not morph:
            entry_segs = re.sub(r'([.,…«»\?!])', '', " ".join(entry_segs)).split()

        entry_correct_predictions = 0
        iv_entry_correct_predictions = 0
        oov_entry_correct_predictions = 0

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if (
                token_index < len(entry_pred)
                and entry_pred[token_index] == entry_gold[token_index]
                # and entry_pred[token_index] != "[UNK]"
            ):
                entry_correct_predictions += 1
                # try:
                #     seg = entry_segs[token_index]
                # except:
                #     print(entry_segs)
                if (
                    entry_segs[token_index] in vocab.keys()
                    and entry_gold[token_index] in vocab[entry_segs[token_index]]
                ):
                    total_iv_tokens += 1
                    iv_entry_correct_predictions += 1
                else:
                    oov_entry_correct_predictions += 1
            else:
                if (
                    entry_segs[token_index] in vocab.keys()
                    and entry_gold[token_index] in vocab[entry_segs[token_index]]
                ):
                    total_iv_tokens += 1

        entry_accuracy = entry_correct_predictions / len(entry_gold)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_correct_iv_preds += iv_entry_correct_predictions
        total_correct_oov_preds += oov_entry_correct_predictions

        total_tokens += len(entry_gold)
        total_oov_tokens = total_tokens - total_iv_tokens

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    overall_in_vocab = total_correct_iv_preds / (
        total_iv_tokens + 0.0000001
    )
    overall_oov = total_correct_oov_preds / (total_oov_tokens + 0.0000001)
    oov_rate = total_oov_tokens / total_tokens

    return {
        "average_accuracy": average_accuracy,
        "accuracy": overall_accuracy,
        "in_vocab_accuracy": overall_in_vocab,
        "oov_accuracy": overall_oov,
        "oov_rate": oov_rate
    }


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
    seg_morphemes: List[List[str]], pred_morphemes: List[List[str]], gold_morphemes: List[List[str]], morph_dict: dict,
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(seg_morphemes, pred_morphemes, gold_morphemes, morph_dict)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])
    mer = eval_avg_error_rate(pred_morphemes, gold_morphemes)
    return {"morpheme_level": morpheme_eval, "classes": class_eval, "bleu": bleu, "MER": mer}


def eval_word_glosses(
    seg_words: List[List[str]], pred_words: List[List[str]], gold_words: List[List[str]], word_dict: dict,
):
    """Evaluates the performance at the word level"""
    word_eval = eval_accuracy(seg_words, pred_words, gold_words, word_dict, morph=False)
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

    def _eval(segs: List[str], preds: List[str], gold: List[str], morph_dict: dict, word_dict: dict, segmented: bool):
        preds = [strip_gloss_punctuation(pred) for pred in preds]
        gold = [strip_gloss_punctuation(g) for g in gold]
        seg_words = [str(seg).split() for seg in segs]
        pred_words = [str(pred).split() for pred in preds]
        gold_words = [gloss.split() for gloss in gold]

        if segmented:
            seg_morphemes = [re.split(r"\s|-", str(seg)) for seg in segs]
        else:
            seg_morphemes = None
        pred_morphemes = [re.split(r"\s|-", str(pred)) for pred in preds]
        gold_morphemes = [re.split(r"\s|-", gloss) for gloss in gold]

        eval_dict = {
            **eval_word_glosses(
                seg_words=seg_words, pred_words=pred_words, gold_words=gold_words, word_dict=word_dict
            ),
            **eval_morpheme_glosses(
                seg_morphemes=seg_morphemes, pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes, morph_dict=morph_dict
            ),
        }
        return eval_dict

    all_eval = {}
    pred_df = pd.read_csv(pred_path).fillna('')
    pred_df = pred_df[pred_df["is_segmented"] == ("yes" if segmented else "no") ]

    if len(pred_df) == 0:
        return {}

    assert test_split in ["test_ID", "test_OOD"]

    if ft_glottocode is None:
        all_eval["all"] = _eval(pred_df["pred"], pred_df["gold"])

    for lang in pred_df["glottocode"].unique():
        lang_preds = pred_df[pred_df["glottocode"] == lang]
        with open(f"../error_analysis/vocabs/morphemes/{get_lang(lang)}.json") as morph_json:
            morph_dict = json.load(morph_json)
        if segmented:
            with open(f"../error_analysis/vocabs/words/{get_lang(lang)}-seg.json") as word_json:
                word_dict = json.load(word_json)
        else:
            with open(f"../error_analysis/vocabs/words/{get_lang(lang)}-unseg.json") as word_json:
                word_dict = json.load(word_json)
        all_eval[lang] = _eval(lang_preds["transcription"], lang_preds["pred"], lang_preds["gold"], morph_dict, word_dict, segmented)

    results_dir = os.path.dirname(pred_path)
    results_path = f"{results_dir}/{test_split}-{'segmented' if segmented else 'unsegmented'}.json"
    with open(results_path, 'w') as outfile:
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
@click.option(
    "--not_segmented",
    help="Evalaute only unsegmented (True), otherwise evaluate both",
    type=bool,
    required=False,
    default=False,
)
def main(pred: str, test_split: str, ft_glottocode, not_segmented):
    if not_segmented:
        evaluate_igt(pred, test_split, segmented=False, ft_glottocode=ft_glottocode)
    else:
        evaluate_igt(pred, test_split, segmented=True, ft_glottocode=ft_glottocode)
        evaluate_igt(pred, test_split, segmented=False, ft_glottocode=ft_glottocode)


if __name__ == "__main__":
    main()
