import pathlib

import datasets
import evaluate
import fire
import numpy as np
import os
import pandas as pd
import torch
import transformers
import wandb
import random
from tqdm import tqdm

DEBUG = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# torch.backends.cuda.matmul.allow_tf32 = True


def create_prompt(row):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    transcription = ' '.join((row['transcription']).split())
    glosses = ' '.join((row['glosses']).split())
    lang = 'an unknown language' if row['language'] == '' else row['language']
    is_segmented = 'unknown' if row['is_segmented'] == '' else row['is_segmented']
    prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
"""
    if row['translation'] is not None:
        if len(row['translation'].strip()) > 0:
            translation = ' '.join((row['translation']).split())
            prompt += f"Translation in {row['metalang']}: {translation}\n"

    prompt += 'Glosses: '

    row['prompt'] = prompt
    row['glosses'] = glosses
    return row


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


def create_trainer(
    model,
    dataset,
    tokenizer,
    batch_size,
    lr,
    max_epochs,
):
    print("Creating trainer...")
    metric = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.argmax(-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, word_order=2
        )
        result = {"chrf++": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        lr=None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

    args = transformers.TrainingArguments(
        output_dir="training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=10,
        gradient_accumulation_steps=64,
        # gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        # load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
        # tf32=True,
    )

    return transformers.Trainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        ),
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["eval_ID"],
        compute_metrics=compute_metrics,
    )


def main(
    mode: str,
    model_path: str,
    test_split: str = None,
):
    pretrained_model = "google/byt5-base"

    random.seed(0)
    if mode == "train" and not DEBUG:
        wandb.init(project="glossLM", entity="wav2gloss", config={
            "model": pretrained_model,
            "segmentation_mode": "all", # "all", "segmented", "unsegmented"
            "translations": "yes", # "yes", "no", "yes_bert"
            "typological_info": "none", # "none", "glottofamily", "glottotree", "grambank", "lang2vec"
            "synthetic_data": "none", # "none", "chatgpt", "treebank", "galactic_deps"
            "curriculum": "none",
        })

    MODEL_INPUT_LENGTH = 1024

    if mode == "train":
        tokenizer = transformers.ByT5Tokenizer.from_pretrained(
            pretrained_model, use_fast=False
        )

        dataset = datasets.load_dataset('lecslab/glosslm-split')
        # filter out samples with empty transcription or gloss fields
        # TODO: fix this in dataset source
        dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)

        dataset = dataset.map(create_prompt)
        dataset = dataset.map(
            tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
        )

        dataset["train"] = dataset["train"].shuffle()

        print(f"Loading model from {pretrained_model}")
        model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model)
        trainer = create_trainer(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=2,
            lr=5e-5,
            max_epochs=10,
        )

        print("Training...")
        trainer.train()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        print(f"Model saved at {model_path}")
    elif mode == "predict":
        tokenizer = transformers.ByT5Tokenizer.from_pretrained(
            pretrained_model, use_fast=False
        )
        dataset = datasets.load_dataset('lecslab/glosslm-split')
        dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)
        assert test_split in ['id', 'ood']
        if test_split == 'id':
            predict_dataset = dataset["test_ID"]
        else:
            predict_dataset = dataset["test_OOD"]
        predict_dataset = predict_dataset.map(create_prompt)
        predict_dataset = predict_dataset.map(
            tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
        )
        print(predict_dataset)
        preds = []
        ids = predict_dataset["ID"]
        is_segmented = predict_dataset["is_segmented"]
        glottocode = predict_dataset["glottocode"]
        model = transformers.AutoModelForPreTraining.from_pretrained(
                model_path).to(device)
        for ex in tqdm(predict_dataset["input_ids"]):
            preds.append(
                tokenizer.decode(
                    model.generate(
                        torch.tensor([ex]).to(device),
                        max_length=MODEL_INPUT_LENGTH,
                    )[0],
                    skip_special_tokens=True,
                )
            )
        preds_df = pd.DataFrame({
            "ID": ids,
            "glottocode": glottocode,
            "is_segmented": is_segmented,
            "pred": preds,
        })
        preds_df.to_csv(f"{test_split}-preds.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
