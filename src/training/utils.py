from typing import Optional, TypeVar

import transformers

T = TypeVar("T")


def force_unwrap(v: Optional[T]) -> T:
    if v is None:
        raise AssertionError("Force unwrap failed because value is None")
    return v


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

    def on_evaluate(
        self,
        args,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        # Only start applying early stopping logic after start_epoch
        if state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


def postprocess(preds):
    corrected_preds = preds.replace("\.$", "", regex=True)
    corrected_preds = corrected_preds.replace("\,", "", regex=True)
    corrected_preds = corrected_preds.replace("»", "", regex=True)
    corrected_preds = corrected_preds.replace("«", "", regex=True)
    corrected_preds = corrected_preds.replace('"', "", regex=True)
    corrected_preds = corrected_preds.replace("\. ", " ", regex=True)
    corrected_preds = corrected_preds.replace("\.\.+", "", regex=True)
    corrected_preds = corrected_preds.replace("\ +", " ", regex=True)
    return corrected_preds
