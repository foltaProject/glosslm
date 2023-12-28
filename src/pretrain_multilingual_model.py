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
from compute_metrics import compute_metrics

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

    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        lr=None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

    args = transformers.Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        # load_best_model_at_end=True,
        logging_steps=100,
        generation_max_length=1024,
        generation_num_beams=3,
        report_to="wandb",
        # tf32=True,
    )

    return transformers.Seq2SeqTrainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        ),
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["eval_ID"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
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

    tokenizer = transformers.ByT5Tokenizer.from_pretrained(
        pretrained_model, use_fast=False
    )
    dataset = datasets.load_dataset('lecslab/glosslm-split')
    dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)
    dataset = dataset.map(create_prompt)
    dataset = dataset.map(
        tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
    )

    dataset["train"] = dataset["train"].shuffle()

    print(f"Loading model from {pretrained_model}")
    model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model if mode == 'train' else model_path)
    model.generation_config.max_new_tokens = MODEL_INPUT_LENGTH
    trainer = create_trainer(
        model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        lr=5e-5,
        max_epochs=10,
    )

    if mode == "train":
        print("Training...")
        trainer.train()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        print(f"Model saved at {model_path}")

    elif mode == "predict":
        print("Creating predictions...")

        assert test_split in ['id', 'ood']
        test_split = "test_" + test_split.upper()

        preds = trainer.predict(dataset[test_split])
        labels = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds_df = pd.DataFrame({
            "ID": dataset[test_split]["ID"],
            "glottocode": dataset[test_split]["glottocode"],
            "is_segmented": dataset[test_split]["is_segmented"],
            "pred": preds,
        })
        preds_df.to_csv(f"{test_split}-preds.csv", index=False)
        print(f"Predictions for {test_split} data saved to {test_split}-preds.csv")


if __name__ == "__main__":
    fire.Fire(main)
