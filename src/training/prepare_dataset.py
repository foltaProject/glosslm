from typing import Dict, cast

import datasets
import transformers

from training.experiment_config import ExperimentConfig


def tokenize(tokenizer: transformers.ByT5Tokenizer, max_length: int):
    def _tokenize(batch):
        nonlocal tokenizer, max_length

        if "glosses" in batch:
            targets = batch["glosses"]
        else:
            targets = None

        model_inputs = tokenizer(
            batch["prompt"],
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        return model_inputs

    return _tokenize


def create_prompt(row: Dict, use_translation: bool = True):
    """Creates an input prompt from the fields in the row.

    Args:
        row (Dict): The row (likely from a dataset) to process
        use_translation (bool): If True, include the translation in the input prompt
    """
    transcription = " ".join((row["transcription"]).split())
    glosses = " ".join((row["glosses"]).split())
    lang = "an unknown language" if row["language"] == "" else row["language"]
    is_segmented = "unknown" if row["is_segmented"] == "" else row["is_segmented"]
    prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
"""
    if row["translation"] is not None and use_translation:
        if len(row["translation"].strip()) > 0:
            translation = " ".join((row["translation"]).split())
            prompt += f"Translation in {row['metalang']}: {translation}\n"

    prompt += "Glosses: "

    row["prompt"] = prompt
    row["glosses"] = glosses
    return row


def prepare_dataset(tokenizer: transformers.ByT5Tokenizer, config: ExperimentConfig):
    """Creates the dataset for training/finetuning

    Args:
        tokenizer (transformers.ByT5Tokenizer): The pretrained tokenizer
        mode ("pretrain" | "finetune" | "predict"): The training mode
        ft_glottocode (str | None): If provided, filter to only a given language (for finetuning/prediction)
        use_unimorph (bool): If True, uses the UniMorph-normalized version of the data
        exclude_st_seg (bool): If True, excludes segmented data from the training split for the evaluation languages
        use_translation (bool): If True, translations will be included in the prompt
    """
    dataset = datasets.load_dataset(
        f"lecslab/glosslm-corpus-split{'-unimorph' if config.use_unimorph else ''}"
    )
    dataset = cast(datasets.DatasetDict, dataset)
    dataset = dataset.filter(
        lambda x: x["transcription"] is not None and x["glosses"] is not None
    )

    # For fair evaluation, we might need to filter out
    # the segmented training examples for our evaluation languages
    if config.exclude_st_seg:
        print("Excluding segmented shared task data...")
        dataset = dataset.filter(
            lambda row: row["source"] != "sigmorphon_st" or row["is_segmented"] == "no"
        )

    # Select the appropriate splits (ID or OOD)
    if config.ft_glottocode is not None:
        dataset = dataset.filter(lambda row: row["glottocode"] == config.ft_glottocode)

        splits = ["train", "eval", "test"]
        if all((dataset[f"{split}_ID"].num_rows != 0 for split in splits)):
            dataset["train"] = dataset["train_ID"]
            dataset["eval"] = dataset["eval_ID"]
            dataset["test"] = dataset["test_ID"]
        elif all((dataset[f"{split}_OOD"].num_rows != 0 for split in splits)):
            dataset["train"] = dataset["train_OOD"]
            dataset["eval"] = dataset["eval_OOD"]
            dataset["test"] = dataset["test_OOD"]
        else:
            raise ValueError("Neither ID nor OOD splits had your glottocode!")
    else:
        # We must be pretraining
        dataset["eval"] = dataset["eval_ID"]

    # Create prompts and tokenize
    dataset = dataset.map(
        create_prompt, fn_kwargs={"use_translation": config.use_translation}
    )
    dataset = dataset.map(tokenize(tokenizer, max_length=1024), batched=True)

    dataset["train"] = dataset["train"].shuffle()
    return dataset
