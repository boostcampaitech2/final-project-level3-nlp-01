import argparser
from model import GraformerModel
from arguments import ModelArguments, DataTrainingArguments

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    default_data_collator
)
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader
import torch

import yaml
import tqdm
import logging

logger = logging.getLogger('Training')
formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
file_handler = logging.FileHandler(file_name='train.log')
logger.addHandler(file_handler)

def read_yaml(cfg):
    try:
        with open(cfg) as f:
            model_args, data_args, training_args = yaml.load_all(f, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print("Config File Not Found:", e)
    return model_args, data_args, training_args

def main(args):
    # Config
    logger.info("Load Arguments")
    model_args, data_args, training_args = read_yaml(args.config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    set_seed(training_args.seed)

    ENCODER_NAME = model_args.encoder_name_or_path
    DECODER_NAME = model_args.decoder_name_or_path
    
    encoder_config = AutoConfig.from_pretrained(ENCODER_NAME)
    decoder_config = AutoConfig.from_pretrained(DECODER_NAME)
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME) # source lang
    decoder_tokenizer = AutoTokenizer.from_pretarined(DECODER_NAME) # target lang
    
    model = GraformerModel(ENCODER_NAME, DECODER_NAME)
    
    # TODO Freeze Model
    for name, param in model.decoder.named_parameters():
        param.requires_grad = False
    
    # TODO Dataset Loader
    datasets = load_from_disk(data_args.dataset_path)
    train_set = datasets["train"]
    valid_set = datasets["validation"]
    def preprocess_function(examples, max_length, max_target_length):
        model_inputs = encoder_tokenizer(examples['korean'], max_length=max_length, padding="max_length", truncation=True)
    
        target_sentences = [decoder_tokenizer.bos_token + ex for ex in examples['japanese']]
        decoder_inputs = decoder_tokenizer(target_sentences, max_length=max_target_length, padding="max_length", truncation=True)

        model_inputs['decoder_input_ids'] = decoder_inputs['input_ids']
        model_inputs['decoder_attention_mask'] = decoder_inputs['attention_mask']
        model_inputs['labels'] = [ex[1:] for ex in decoder_inputs['input_ids']]

        return model_inputs
    
    fn_kwargs = {"max_length": 512, "max_target_length": 1024}
    tokenized_train = train_set.map(preprocess_function, num_proc=5, batched=True, remove_columns=train_set.column_names, fn_kwargs=fn_kwargs)
    tokenized_valid = valid_set.map(func, num_proc=5, batched=True, remove_columns=valid_set.column_names, fn_kwargs=fn_kwargs)
    
    train_datalaoder = DataLoader(tokenized_train, batch_size=32, pin_memory=True, shuffle=True, 
                                  drop_last=True, num_workers=5, collate_fn=default_data_collator)
    eval_dataloader = DataLoader(tokenized_valid, batch_size=32, pin_memory=True, shuffle=False, 
                                 drop_last=False, num_workers=5, collate_fn=default_data_collator)
    # eval_dataloader = DataLoader(eval_data, batch_size=BATCH_SIZE,
    #                    shuffle=True, collate_fn=generate_batch)
    
    
    # Get the language codes for input/target.
    # source_lang = data_args.source_lang.split("_")[0]
    # target_lang = data_args.target_lang.split("_")[0]

    # # Temporarily set max_target_length for training.
    # max_target_length = data_args.max_target_length
    # padding = "max_length" if data_args.pad_to_max_length else False
    optimizer = torch.optim.AdamW(model.parmeters(), lr=training_args.learning_rate)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        trainig_args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)

    metric = load_metric("sacrebleu")

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

    model.to(device)
    for epoch in range(training_args.num_train_epochs):
        # TODO: Train
        model.train()
        for step, batch in enumerate(train_dataloader):
            # TODO: batch?
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["decoder_attention_mask"])
               
            loss = 
            
            if (step + 1) % training_args.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimzer.zero_grad()
            
            
        
        # TODO: Do eval
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            pass
            
        # TODO: Save Model?
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        type=str,
        help="config file",
    )
    args = parser.parse_args()
    main(args)