import pathlib

import datasets
import evaluate
import fire
import numpy as np
import torch
import transformers
import wandb
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def create_prompt(row):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    lang = 'an unknown language' if row['language'] == '' else row['language']
    is_segmented = 'unknown' if row['is_segmented'] == '' else row['is_segmented']
    prompt = f"""Provide the glosses for the following line of Interlinear Glossed Text for a sentence in {lang}.

Transcription in {lang}: {row['transcription']}
Transcription segmented into morphemes: {is_segmented}
"""
    if row['translation'] != '':
        prompt += f"Translation in {row['metalang']}: {row['translation']}\n"

    prompt += 'Glosses: '

    row['prompt'] = prompt
    row['glosses'] = row['glosses'] if row['glosses'] != '' else None
    return row


def tokenize(tokenizer: transformers.ByT5Tokenizer, max_length: int):
    def _tokenize(batch):
        nonlocal tokenizer, max_length

        if "gloss" in batch:
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
        gradient_accumulation_steps=64,
        # gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        # load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
        tf32=True,
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
):
    random.seed(0)
    if mode == "train":
        wandb.init(project="glossLM", entity="wav2gloss")

    MODEL_INPUT_LENGTH = 1024

    pretrained_model = "google/byt5-base"

    if mode == "train":
        tokenizer = transformers.ByT5Tokenizer.from_pretrained(
            pretrained_model, use_fast=False
        )

        dataset = datasets.load_dataset('lecslab/glosslm-split')
        dataset = dataset.map(create_prompt)
        dataset = dataset.map(
            tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
        )

        dataset["train"] = dataset["train"].shuffle()

        model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model)
        trainer = create_trainer(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=16,
            lr=5e-5,
            max_epochs=10,
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        print(f"Model saved at {model_path}")
    elif mode == "predict":
        pass


if __name__ == "__main__":
    fire.Fire(main)
