from dataclasses import dataclass
from typing import Literal

TRAIN_MODE = Literal["pretrain", "predict", "finetune"]

_glotto_to_iso = {
    "arap1274": "arp",
    "gitx1241": "git",
    "dido1241": "ddo",
    "uspa1245": "usp",
    "nyan1302": "nyb",
    "natu1246": "ntu",
    "lezg1247": "lez",
}


@dataclass
class ExperimentConfig:
    """
    Args:
        mode ("pretrain", "finetune", "predict"): The mode to run in
        exp_name (str): A string used to label the experiment in logging
        pretrained_model (str): The name of the pretrained model to train or predict with
        ft_glottocode (str, optional): The language to use for finetuning/prediction
        max_epochs (int): Maximum number of training epochs
        early_stopping_patience (int): Number of epochs with no improvement after which training is stopped
        exclude_st_seg (bool): If True, excludes the segmented training data for the evaluation languages
        use_translation (bool): If True, include the translation in the prompt
        use_unimorph (bool): If True, use the UniMorph-normalized version of the dataset
        output_model_path (str): The path to output the model to
        checkpoint_path (str, optional): The path to the checkpoint file when continuing training
        checkpoint_save_dir (str): Directory where checkpoints will be saved
    """

    # General
    mode: TRAIN_MODE
    exp_name: str
    pretrained_model: str = "google/byt5-base"

    # Dataset
    ft_glottocode: str | None = None
    exclude_st_seg: bool = False
    use_translation: bool = True
    use_unimorph: bool = True

    # Training
    max_epochs: int = 13
    early_stopping_patience: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 2

    # Files
    output_model_path: str | None = None
    checkpoint_path: str | None = None
    checkpoint_dir: str = "training_checkpoints/"

    # Computed properties
    @property
    def ft_isocode(self):
        if self.ft_glottocode is not None:
            return _glotto_to_iso[self.ft_glottocode]
        else:
            return None

    @property
    def use_early_stopping(self):
        return self.mode == "finetune"

    def __post_init__(self):
        """Validates sanity checks on the parameters"""
        if self.ft_glottocode is not None:
            if self.mode == "pretrain":
                raise ValueError("Pretraining should not have a specified glottocode!")
        else:
            if self.mode != "pretrain":
                raise ValueError("Finetuning/prediction must have a glottocode!")
