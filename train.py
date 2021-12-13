import random
import math
import logging
import argparse
from tqdm import tqdm

import yaml
import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer
from transformers import set_seed, get_cosine_schedule_with_warmup, AdamW

from model import GrafomerModel
from utils import preprocess_function_with_setting, load_data, postprocess_text, CustomDataCollator

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
file_handler = logging.FileHandler(filename='train.log')
logger.addHandler(file_handler)


@hydra.main(config_path='./configs', config_name="config.yaml")
def main(cfg):
    
    # Config
    print("\n====== Using Hydra Configurations ======")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    if cfg.train_config.seed:
        logger.info(f"set the seed in ``random``, ``numpy``, ``torch`` at {cfg.train_config.seed}")
        set_seed(cfg.train_config.seed)
    
    enc_name = cfg.encoder.name
    dec_name = cfg.decoder.name
    
    encoder_tokenizer = getattr(__import__("transformers"), cfg.encoder.tokenizer).from_pretrained(enc_name) # source lang
    decoder_tokenizer = getattr(__import__("transformers"), cfg.decoder.tokenizer).from_pretrained(dec_name) # target lang
    
    model = GrafomerModel(enc_name, dec_name, cfg)
    print(f"number of model parameters: {model.num_parameters()}")
    
    # Temp: 만약 decoder tokenizer에 bos token 추가가 필요하다면 주석 해제
    # special_tokens_dict = {"additional_special_tokens": [cfg.decoder.bos_token]}
    # decoder_tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    # model.decoder.resize_token_embeddings(len(decoder_tokenizer))
    # decoder_tokenizer_tokenizer.bos_token = cfg.decoder.bos_token
    
    # Temp: 따로 모델에 bos token의 embedding을 늘려줄 필요가 없을 때는 위 주석 코드 말고 여기만 사용
    # e.g. 중국어 gpt는 bert tokenizer를 사용해서 모델 임베딩은 바꿔줄 필요없이 bos token으로 cls 토큰을 설정해주시만 하면 됨.
    if decoder_tokenizer.bos_token is None:
        decoder_tokenizer.bos_token = cfg.decoder.bos_token

    
    # TODO Data Loader
    # train_set, valid_set = load_data(cfg.data.ko_ja)
    train_set, valid_set = load_data(cfg.train_config.data_path)
    
    fn_kwargs = cfg.data.fn_kwargs
    tokenized_train = train_set.map(preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, cfg.data.switch, cfg.decoder.need_prefix), 
                                    num_proc=8, batched=True, remove_columns=train_set.column_names, fn_kwargs=fn_kwargs)
    tokenized_valid = valid_set.map(preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, cfg.data.switch),
                                    num_proc=8, batched=True, remove_columns=valid_set.column_names, fn_kwargs=fn_kwargs)

    data_collator = CustomDataCollator(encoder_pad_token_id=encoder_tokenizer.pad_token_id, decoder_pad_token_id=decoder_tokenizer.pad_token_id)
    train_dataloader = DataLoader(tokenized_train, batch_size=cfg.train_config.batch_size, pin_memory=True,
                                  shuffle=True, drop_last=True, num_workers=8, collate_fn=data_collator)
    valid_dataloader = DataLoader(tokenized_valid, batch_size=cfg.train_config.batch_size, pin_memory=True, 
                                  shuffle=False, drop_last=False, num_workers=8, collate_fn=data_collator)

    print("전처리 결과 한번 확인\n", decoder_tokenizer.batch_decode(next(iter(train_dataloader))["decoder_input_ids"]))


    # TODO Decoder Model Freeze
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # TODO: Train
    sacre_bleu = load_metric("sacrebleu")
    bert_score = load_metric("bertscore")

    total_steps = len(train_dataloader)  // cfg.train_config.gradient_accumulation_steps * cfg.train_config.num_train_epochs
    steps_per_epoch = len(train_dataloader) // cfg.train_config.gradient_accumulation_steps
    warmup_steps = int(total_steps ** 0.05)
    eval_steps = cfg.train_config.eval_steps
    
    logger.info(
        f"num train dataloader: {len(train_dataloader)} | gradient accumulaion steps: {cfg.train_config.gradient_accumulation_steps} | num_train_epochs: {cfg.train_config.num_train_epochs} | "
        f"total steps: {total_steps} | steps per epoch: {steps_per_epoch}"
    )

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

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.train_config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    loss_function = nn.CrossEntropyLoss(ignore_index=decoder_tokenizer.pad_token_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    step = 0
    completed_steps = 0
    model.to(device)
    for epoch in range(cfg.train_config.num_train_epochs):
        
        model.train()
        progress_bar = tqdm(range(steps_per_epoch), ncols=100)

        for batch in train_dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(): # 16 bit training 적용
            
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["decoder_attention_mask"])
                logits = outputs.logits.view(-1, len(decoder_tokenizer))
                
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
            if step % (eval_steps * cfg.train_config.gradient_accumulation_steps) == 0 :
                
                model.eval()
                eval_progress_bar = tqdm(range(len(valid_dataloader)), ncols=100)
                eval_loss = 0
                preds, gt = [], []
                sample_indices = random.choices(range(500), k=50)

                for eval_step, eval_batch in enumerate(valid_dataloader):

                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}

                    """
                    output = my_model.generate(input_ids, attention_mask=attention_mask , max_length=1025,
                    pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                    num_beams=5, temperature=0.9, top_k=50, top_p=1.0, 
                    repetition_penalty=1.0, use_cache=True)
                    """
                    with torch.no_grad():
                        
                        # decoder_input_ids = torch.ones((cfg.train_config.batch_size, 1), dtype=torch.long, device=device) * decoder_tokenizer.bos_token_id
                        generated_tokens = model.generate(eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], max_length=int(eval_batch["input_ids"].shape[1] * 1.3),
                                                        # decoder_input_ids=eval_batch["decoder_input_ids"], decoder_attention_mask=eval_batch["decoder_attention_mask"],
                                                        pad_token_id=decoder_tokenizer.pad_token_id, eos_token_id=decoder_tokenizer.eos_token_id, bos_token_id=decoder_tokenizer.bos_token_id,
                                                        do_sample=True, top_k=50, top_p=0.95)
                        labels = eval_batch["labels"]

                        decoded_preds = decoder_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                        sacre_bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
                        bert_score.add_batch(predictions=decoded_preds, references=decoded_labels)
                        preds.extend(decoded_preds); gt.extend(decoded_labels)

                    
                    eval_progress_bar.update()
                    if eval_step == 1000: 
                        break
                
                sacre_bleu_results = sacre_bleu.compute()
                bert_score_results = bert_score.compute()

                logger.info(
                    f"{completed_steps} steps evaluation results \n"
                    f"{sacre_bleu_results} \n"
                    f"{bert_score_results} \n"
                )

                logger.info(
                    f"decoded sentences: "
                )
                for i, n in enumerate(sample_indices, 1):
                    logger.info(f"[{i}] (gt) {gt[n]}  ->  (pred) {preds[n]}")
                
                eval_progress_bar.close()

                # TODO: Model Checkpointing
                cur_lang = cfg.lang

                torch.save(model.encoder.state_dict(), f"{cfg.train_config.save_dir}/encoder/checkpoint_{completed_steps}.pt")
                torch.save(model.decoder.state_dict(), f"{cfg.train_config.save_dir}/decoder/{cur_lang}/checkpoint_{completed_steps}.pt")
                torch.save(model.graft_module.state_dict(), f"{cfg.train_config.save_dir}/graft_module/{cur_lang}/checkpoint_{completed_steps}.pt")
                
                model.train()
                
        progress_bar.close()

    

if __name__ == "__main__":
    
    main()