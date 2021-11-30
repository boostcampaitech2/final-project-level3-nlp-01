import yaml
import tqdm
import math
import logging
import argparse

import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, set_seed, default_data_collator, get_cosine_schedule_with_warmup

from model import GrafomerModel
from utils import preprocess_function_with_setting, load_data

# logger = logging.getLogger('Training')
# formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
# file_handler = logging.FileHandler(filename='train.log')
# logger.addHandler(file_handler)


@hydra.main(config_path='./configs', config_name="config.yaml")
def main(cfg):
    # Config
    # logger.info("Load Arguments")
    
    # set_seed(training_args.seed)
    print(OmegaConf.to_yaml(cfg))
    
    enc_name = cfg.encoder.name
    dec_name = cfg.decoder.name
    
    encoder_tokenizer = getattr(__import__("transformers"), cfg.encoder.tokenizer).from_pretrained(enc_name) # source lang
    decoder_tokenizer = getattr(__import__("transformers"), cfg.decoder.tokenizer).from_pretrained(dec_name) # target lang
    
    model = GrafomerModel(enc_name, dec_name, cfg)
    
    
    # TODO Data Loader
    train_set, valid_set = load_data(cfg.data.ko_ja)
    
    fn_kwargs = cfg.data.fn_kwargs
    tokenized_train = train_set.map(preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, cfg.data.switch), 
                                    num_proc=8, batched=True, remove_columns=train_set.column_names, fn_kwargs=fn_kwargs)
    tokenized_valid = valid_set.map(preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, cfg.data.switch), 
                                    num_proc=8, batched=True, remove_columns=valid_set.column_names, fn_kwargs=fn_kwargs)
    
    train_dataloader = DataLoader(tokenized_train, batch_size=cfg.train_config.batch_size, pin_memory=True,
                                  shuffle=True, drop_last=True, num_workers=5, collate_fn=default_data_collator)
    valid_dataloader = DataLoader(tokenized_valid, batch_size=cfg.train_config.batch_size, pin_memory=True, 
                                  shuffle=False, drop_last=False, num_workers=5, collate_fn=default_data_collator)


    # TODO Decoder Model Freeze
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # TODO: Train
    metric = load_metric("sacrebleu")

    total_steps = (len(train_dataloader) * cfg.train_config.num_train_epochs) // cfg.train_config.gradient_accumulation_steps 
    warmup_steps = int(total_steps ** 0.05)
    
    no_decay=['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': cfg.train_config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.train_config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    loss_function = nn.CrossEntropyLoss(ignore_index=decoder_tokenizer.pad_token_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    step = 0
    completed_steps = 0
    model.to(device)
    for epoch in range(cfg.train_config.num_train_epochs):
        
        model.train()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for step, batch in progress_bar:

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(): # 16 bit training 적용
            
                logits = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["decoder_attention_mask"])
                logits = logits.view(-1, len(decoder_tokenizer))
                
                labels = batch["labels"].view(-1)
                loss = loss_function(logits, labels)

                scaler.scale(loss / cfg.train_config.gradient_accumulation_steps).backward()

                step += 1
            
                if step % cfg.train_config.gradient_accumulation_steps == 0:
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    completed_steps += 1
                    progress_bar.update()
                    progress_bar.set_description(
                        f"Train: [{epoch + 1:03d}] "
                        f"Loss: {loss:.3f}, "
                        f"lr: {optimizer.param_groups[0]['lr']:.7f}"
                    )
            
        
        # TODO: Do eval
        
            
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
    main()