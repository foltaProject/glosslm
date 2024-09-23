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
from eval import strip_gloss_punctuation

DEBUG = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# torch.backends.cuda.matmul.allow_tf32 = True


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


def create_prompt(
    row,
    use_translation: bool = True,
):
    """Processing function for rows in the dataset, creates an input prompt from the fields in the row."""
    transcription = ' '.join((row['transcription']).split())
    glosses = ' '.join((row['glosses']).split())
    lang = 'an unknown language' if row['language'] == '' else row['language']
    is_segmented = 'unknown' if row['is_segmented'] == '' else row['is_segmented']
    prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
"""
    if row['translation'] is not None and use_translation:
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


class DelayedEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        # Only start applying early stopping logic after start_epoch
        if state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


def create_trainer(
    model,
    output_dir,
    checkpoint_path,
    dataset,
    tokenizer,
    batch_size,
    lr,
    max_epochs,
    use_early_stopping,
    early_stopping_patience,
    id_or_ood,
):
    assert id_or_ood in ["ID", "OOD"]
    print("Creating trainer...")

    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        lr=None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

    print(f"checkpoints saving to {output_dir}")
    args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=10,
        gradient_accumulation_steps=64,
        # gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=10 if use_early_stopping else 3,
        num_train_epochs=max_epochs,
        predict_with_generate=True,
        load_best_model_at_end=use_early_stopping,
        logging_steps=100,
        generation_max_length=1024,
        generation_num_beams=3,
        report_to="wandb",
        metric_for_best_model="chrf++",
        # tf32=True,
        fp16=True,
        dataloader_num_workers=4,
    )

    return transformers.Seq2SeqTrainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        ),
        train_dataset=dataset["train" if id_or_ood == "ID" else "train_OOD"] if dataset else None,
        eval_dataset=dataset[f"eval_{id_or_ood}"],
        compute_metrics=compute_metrics(tokenizer),
        tokenizer=tokenizer,
        callbacks=[DelayedEarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if use_early_stopping else [],
    )


def main(
    mode: str,
    exp_name: str,
    output_model_path: str = None,
    checkpoint_path: str = None,
    pretrained_model: str = None,
    test_split: str = None,
    ft_glottocode: str = None,
    max_epochs: int = 13,
    early_stopping_patience: int = 3,
    exclude_st_seg: bool = False,
    use_translation: bool = True,
    use_unimorph: bool = True,
    checkpoint_save_dir: str = "training_checkpoints/",
):
    assert mode in ["train", "predict", "finetune"]
    assert (output_model_path is not None) if mode == "train" else True
    assert (test_split is not None and pretrained_model is not None) if mode == "predict" else True
    assert (ft_glottocode is not None and pretrained_model is not None and output_model_path is not None) if mode == "finetune" else True

    random.seed(0)

    if (mode == "train" or mode == "finetune") and not DEBUG:
        run_name = exp_name
        if mode == "finetune":
            run_name += "-ft-" + ft_glottocode

        wandb.init(project="glossLM", entity="wav2gloss", name=run_name, config={
            "model": pretrained_model,
            "segmentation_mode": "all", # "all", "segmented", "unsegmented"
            "use translations": use_translation, # "yes", "no", "yes_bert"
            "typological_info": "none", # "none", "glottofamily", "glottotree", "grambank", "lang2vec"
            "synthetic_data": "none", # "none", "chatgpt", "treebank", "galactic_deps"
            "curriculum": "none",
            "finetuning_language": ft_glottocode
        })

    MODEL_INPUT_LENGTH = 1024

    tokenizer = transformers.ByT5Tokenizer.from_pretrained(
        "google/byt5-base", use_fast=False
    )

    if use_unimorph:
        dataset = datasets.load_dataset('lecslab/glosslm-split-unimorph')
    else:
        dataset = datasets.load_dataset('lecslab/glosslm-split')
        nyan_train = dataset['train_OOD'].filter(lambda row: row['glottocode'] == 'nyan1302').shuffle().select(range(1000))
        natu_train = dataset['train_OOD'].filter(lambda row: row['glottocode'] == 'natu1246').shuffle().select(range(500))

        dataset['train'] = datasets.concatenate_datasets([dataset['train'], nyan_train, natu_train])
    dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)

    # filtering out the shared task segmented data for comparison
    if mode == "train" and exclude_st_seg:
        print("excluding segmented shared task data")
        dataset_st = dataset.filter(lambda x: x["source"] == "sigmorphon_st")
        dataset_st_unseg = dataset_st.filter(lambda x: x["is_segmented"] == "no")
        dataset_no_st = dataset.filter(lambda x: x["source"] != "sigmorphon_st")
        dataset["train"] = datasets.concatenate_datasets([dataset_no_st["train"], dataset_st_unseg["train"]])

    # If finetuning, we may need to switch to using the OOD data splits
    id_or_ood = "ID"
    if mode == "finetune":
        dataset = dataset.filter(lambda row: row["glottocode"] == ft_glottocode)
        if exclude_st_seg:
            print("excluding segmented shared task data")
            dataset = dataset.filter(lambda row: row["is_segmented"] == "no")
        if dataset['eval_ID'].num_rows == 0 and dataset['eval_OOD'].num_rows != 0:
            id_or_ood = "OOD"
        # at minimum, finetune for 100 epochs
        if max_epochs < 100:
            max_epochs = 100

    if not use_translation:
        print("excluding translations")
    dataset = dataset.map(create_prompt, fn_kwargs={"use_translation": use_translation})
    dataset = dataset.map(
        tokenize(tokenizer, max_length=MODEL_INPUT_LENGTH), batched=True
    )

    print(dataset["train"])
    dataset["train"] = dataset["train"].shuffle()
    dataset["train_OOD"] = dataset["train_OOD"].shuffle()

    if mode == "train":
        pretrained_model = "google/byt5-base"
    print(f"Loading model from {pretrained_model}")
    model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model)
    output_dir = os.path.join(checkpoint_save_dir, exp_name)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    trainer = create_trainer(
        model,
        output_dir=output_dir,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        lr=5e-5,
        max_epochs=max_epochs,
        use_early_stopping=(mode == "finetune"),
        early_stopping_patience=early_stopping_patience,
        id_or_ood=id_or_ood,
        checkpoint_path=checkpoint_path,
    )

    if mode == "train" or mode == "finetune":
        print("Training...")
        if checkpoint_path is not None:
            print(f"Continuing training from {checkpoint_path}")
        trainer.train(checkpoint_path)
        # if not os.path.exists(output_model_path):
        #     os.makedirs(output_model_path)
        print(f"Saving model to {output_model_path}")
        trainer.save_model(output_model_path)
        print(f"Model saved at {output_model_path}")

    elif mode == "predict":
        print("Creating predictions...")

        assert test_split in ['ID', 'OOD']
        test_split = "test_" + test_split.upper()

        # only generate preds for finetuning lang if specified
        if ft_glottocode is not None:
            dataset = dataset.filter(lambda row: row["glottocode"] == ft_glottocode)
            lang = get_lang(ft_glottocode)
            if not os.path.exists(f"../preds/{lang}-{exp_name}"):
                os.makedirs(f"../preds/{lang}-{exp_name}")
            pred_path = f"../preds/{lang}-{exp_name}/{test_split}-preds.csv"

        else:
            if not os.path.exists(f"../preds/{exp_name}"):
                os.makedirs(f"../preds/{exp_name}")
            pred_path = f"../preds/{exp_name}/{test_split}-preds.csv"

        preds = trainer.predict(dataset[test_split])
        labels = np.where(preds.predictions != -100, preds.predictions, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds = [strip_gloss_punctuation(pred) for pred in preds]

        gold = [strip_gloss_punctuation(g) for g in dataset[test_split]["glosses"]]

        preds_df = pd.DataFrame({
            "id": dataset[test_split]["id"],
            "glottocode": dataset[test_split]["glottocode"],
            "is_segmented": dataset[test_split]["is_segmented"],
            "pred": preds,
            "gold": gold,
        })

        preds_df.to_csv(pred_path, index=False)

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

        preds_df['pred'] = clean_preds(preds_df['pred'])
        preds_df['gold'] = clean_preds(preds_df['gold'])
        preds_df.to_csv(pred_path[:-4] + '.postprocessed.csv', index=False)

        print(f"Predictions for {test_split} data saved to {pred_path}")


if __name__ == "__main__":
    fire.Fire(main)
