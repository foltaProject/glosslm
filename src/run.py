import argparse
import os
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

from config_to_dataclass import config_to_dataclass
from training.compute_metrics import compute_metrics
from training.eval import strip_gloss_punctuation
from training.experiment_config import ExperimentConfig
from training.prepare_dataset import prepare_dataset
from training.utils import DelayedEarlyStoppingCallback, postprocess

DEBUG = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def _make_if_needed(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run(config: ExperimentConfig):
    random.seed(0)

    if config.mode == "finetune":
        config.max_epochs = min(config.max_epochs, 100)

    # Initialize WandB experiment
    if (config.mode == "train" or config.mode == "finetune") and not DEBUG:
        run_name = config.exp_name
        if config.mode == "finetune":
            run_name += f"-ft-{config.ft_glottocode}"

        wandb.init(
            project="glossLM",
            entity="wav2gloss",
            name=run_name,
            config=asdict(config),
        )

    # Prepare model, dataset, tokenizer
    tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-base", use_fast=False)
    dataset = prepare_dataset(tokenizer=tokenizer, config=config)
    model = T5ForConditionalGeneration.from_pretrained(config.pretrained_model)

    # Create trainer
    print("Creating trainer...")
    output_dir = _make_if_needed(os.path.join(config.checkpoint_dir, config.exp_name))
    print(f"checkpoints saving to {output_dir}")
    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        lr=None,
        warmup_init=True,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)
    args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=10,
        gradient_accumulation_steps=64,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=10 if config.use_early_stopping else 3,
        num_train_epochs=config.max_epochs,
        predict_with_generate=True,
        load_best_model_at_end=config.use_early_stopping,
        logging_steps=100,
        generation_max_length=1024,
        generation_num_beams=3,
        report_to="wandb",
        metric_for_best_model="chrf++",
        fp16=True,
        dataloader_num_workers=4,
    )
    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        optimizers=(optimizer, lr_scheduler),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        ),
        train_dataset=dataset["train"],  # type:ignore
        eval_dataset=dataset["eval"],  # type:ignore
        compute_metrics=compute_metrics(tokenizer),
        tokenizer=tokenizer,
        callbacks=[
            DelayedEarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ]
        if config.use_early_stopping
        else [],
    )

    if config.mode == "pretrain" or config.mode == "finetune":
        print("Training...")
        if config.checkpoint_path is not None:
            print(f"Continuing training from {config.checkpoint_path}")
        trainer.train(config.checkpoint_path)

        if config.output_model_path is None:
            raise ValueError("Must have an output path when training")
        print(f"Saving model to {config.output_model_path}")
        trainer.save_model(_make_if_needed(config.output_model_path))
        print("Model saved!")

    elif config.mode == "predict":
        print("Creating predictions...")
        preds_dir = _make_if_needed(
            "../preds/{config.exp_name}/{force_unwrap(config.ft_isocode)}/"
        )
        preds_path = os.path.join(preds_dir, "test-preds.csv")
        preds = trainer.predict(dataset["test"])  # type:ignore
        labels = np.where(
            preds.predictions != -100, preds.predictions, tokenizer.pad_token_id
        )
        preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds = [strip_gloss_punctuation(pred) for pred in preds]
        gold = [strip_gloss_punctuation(g) for g in dataset["test"]["glosses"]]
        preds_df = pd.DataFrame(
            {
                "id": dataset["test"]["id"],
                "glottocode": dataset["test"]["glottocode"],
                "is_segmented": dataset["test"]["is_segmented"],
                "pred": preds,
                "gold": gold,
            }
        )
        preds_df.to_csv(preds_path, index=False)
        preds_df["pred"] = postprocess(preds_df["pred"])
        preds_df["gold"] = postprocess(preds_df["gold"])
        preds_df.to_csv(preds_path[:-4] + ".postprocessed.csv", index=False)
        print(f"Predictions for {config.ft_glottocode} data saved to {preds_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "-o",
        "--overrides",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or "",
        dataclass_type=ExperimentConfig,
    )
    breakpoint()
    run(config)
