import click
import pandas as pd
from datasets import load_dataset


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


@click.command()
@click.option(
    "--pred_dir",
    help="File containing predicted IGT",
    type=click.Path(exists=True),
    required=True,
)
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
def main(pred_dir: str, pred: str, test_split: str, ft_glottocode: str):
    dataset = load_dataset('lecslab/glosslm-split', split=test_split)
    dataset = dataset.filter(lambda x: x["glottocode"] == ft_glottocode)
    transcriptions = dataset["transcription"]
    # print(transcriptions)
    preds_df = pd.read_csv(pred)
    preds_df['pred'] = clean_preds(preds_df['pred'])
    preds_df['gold'] = clean_preds(preds_df['gold'])
    preds_df.to_csv(pred[:-4] + '.postprocessed.csv', index=False)
    preds_df["transcription"] = transcriptions
    print(preds_df.head())
    preds_df.to_csv(f"{pred_dir}/{test_split}-preds-postprocessed.csv")


if __name__ == "__main__":
    main()
