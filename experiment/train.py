import os
import logging
import json
from pathlib import Path

from datasets import load_dataset, DatasetDict
import Model

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def load_data(path):

    data_dir = Path(path)
    folder_list = os.listdir(data_dir)

    dataset_dict = dict()

    for folder in folder_list:

        target_dir = data_dir.joinpath(folder)
        file_list = [str(p) for p in target_dir.glob(pattern="*/*.csv")]
        data = load_dataset("csv", data_files=file_list)
        dataset_dict[folder] = data["train"]
    
    return DatasetDict(dataset_dict)


def train():

    # load model and tokenizer
    # model = Model.Graformer()
    # encoder_tokenizer = AutoTokenizer.from_pretrained()
    # decoder_tokenizer = AutoTokenizer.from_pretrained()

    # load dataset
    data = load_data("/opt/ml/final-project-level3-nlp-01/data/ko-ja")

    # preprocessing and tokenize

    # make dataloader

    # train loop



    pass



if __name__ == "__main__":
    
    train()

