import pandas as pd
from eval import strip_gloss_punctuation, eval_word_glosses, eval_morpheme_glosses
from typing import List
import re
import argparse

def clean_preds(preds):
    corrected_preds = preds.replace('\.$', '', regex=True)
    corrected_preds = corrected_preds.replace('\,', '', regex=True)
    corrected_preds = corrected_preds.replace('»', '', regex=True)
    corrected_preds = corrected_preds.replace('«', '', regex=True)
    corrected_preds = corrected_preds.replace('\"', '', regex=True)
    corrected_preds = corrected_preds.replace('\. ', ' ', regex=True)
    corrected_preds = corrected_preds.replace('\.\.+', '', regex=True)
    corrected_preds = corrected_preds.replace('\ +', ' ', regex=True)
    return corrected_preds

def _eval(preds: List[str], gold: List[str]):
    preds = [strip_gloss_punctuation(pred) for pred in preds]
    gold = [strip_gloss_punctuation(g) for g in gold]
    pred_words = [str(pred).split() for pred in preds]
    gold_words = [gloss.split() for gloss in gold]
    # word_eval = eval_accuracy(pred_words, gold_words)

    pred_morphemes = [re.split(r"\s|-", str(pred)) for pred in preds]
    gold_morphemes = [re.split(r"\s|-", gloss) for gloss in gold]

    eval_dict = {
        **eval_word_glosses(
            pred_words=pred_words, gold_words=gold_words
        ),
        **eval_morpheme_glosses(
            pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes
        ),
    }
    return eval_dict

def postprocess(path, segmented: bool):
    pred_df = pd.read_csv(path).fillna('')
    pred_df = pred_df[pred_df['is_segmented'] == ("yes" if segmented else "no")]
    pred_df['pred'] = clean_preds(pred_df['pred'])
    pred_df['gold'] = clean_preds(pred_df['gold'])
    pred_df.to_csv(path[:-4] + '.postprocessed.csv')
    all_eval = {}
    all_eval['all'] = _eval(pred_df['pred'], pred_df['gold'])

    for lang in pred_df["glottocode"].unique():
        lang_df = pred_df[pred_df["glottocode"] == lang]
        all_eval[lang] = _eval(lang_df['pred'], lang_df['gold'])
    return all_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument('--segmented', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    postprocess(args.file, args.segmented)