import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Any
from datasets import load_dataset
from transformers import BatchEncoding


def read_yaml(cfg):
    try:
        with open(cfg) as f:
            model_args, data_args, training_args = yaml.load_all(f, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print("Config File Not Found:", e)
    return model_args, data_args, training_args


def preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, switch_language_pair=False, need_prefix=True):
    def preprocess_function(examples, max_length, max_target_length):

        src_sentence = examples["text"] if not switch_language_pair else examples["target"]
        tgt_sentence = examples["target"] if not switch_language_pair else examples["text"]
        
        if need_prefix:
            tgt_sentence = [decoder_tokenizer.bos_token + ex for ex in tgt_sentence]

        model_inputs = encoder_tokenizer(src_sentence, max_length=max_length, padding=False, truncation=True)
        decoder_inputs = decoder_tokenizer(tgt_sentence, max_length=max_target_length, padding=False, truncation=True)

        model_inputs['decoder_input_ids'] = decoder_inputs['input_ids']
        model_inputs['decoder_attention_mask'] = decoder_inputs['attention_mask']
        model_inputs['labels'] = [ex[1:]+[decoder_tokenizer.pad_token_id] for ex in decoder_inputs['input_ids']]

        return model_inputs
        
    return preprocess_function


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = decoder_tokenizer.batch_decode(preds, skip_special_tokens = True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions = decoded_preds, references = decoded_labels)
    result = {"bleu": result["score"]}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}

    return result


def load_data(path):
    _train, _valid = load_dataset(path, split=["train[1%:]", "train[:1%]"], use_auth_token="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")
    return _train, _valid


@dataclass
class CustomDataCollator:
    """ variant of DataCollatorForSeq2Seq """
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        """
        features (input): List[Dict]  :  Dict keys -> input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels
        features (return): Dict[]
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        max_encoder_input_length = max(len(l) for l in input_ids)
        for i in range(len(input_ids)):
            remainder = [self.label_pad_token_id] * (max_encoder_input_length - len(input_ids[i]))
            input_ids[i] = input_ids[i] + remainder
            attention_mask[i] = attention_mask[i] + remainder

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        decoder_input_ids = [feature["decoder_input_ids"] for feature in features]
        decoder_attention_mask = [feature["decoder_attention_mask"] for feature in features]
        max_label_length = max(len(l) for l in labels)
        for i in range(len(labels)):
            remainder = [self.label_pad_token_id] * (max_label_length - len(labels[i]))
            labels[i] = labels[i] + remainder
            decoder_input_ids[i] = decoder_input_ids[i] + remainder
            decoder_attention_mask[i] = decoder_attention_mask[i] + remainder

        features = BatchEncoding({
            "input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask, "labels": labels,
        }, tensor_type=return_tensors)

        return features