import random
import argparse
from tqdm import tqdm
from transformers.file_utils import CONFIG_NAME
import os

import yaml
import hydra
from omegaconf import OmegaConf
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_metric, load_from_disk
from transformers import AutoConfig, AutoTokenizer, BertTokenizerFast
from transformers import set_seed, get_cosine_schedule_with_warmup, AdamW

from model import GrafomerModel
from utils import preprocess_function_with_setting, load_data, postprocess_text, CustomDataCollator
import re

# : 앞 문자 제거
def remove_colon(example):
    texts = example['target']
    texts2 = example['text']
    
    text = re.sub(r".+:", "", texts).strip() 
    text2 = re.sub(r".+:", "", texts2).strip() 
    
    example['target'] = text
    example['text'] = text2
    return example


# 길거나 짧은 문장 제외
def filter_length(example):
    # 길이는 따로 설정해 주세요
    if len(example['inpud_ids']) < 10 or len(example['input_ids']) > 100:
        return False
    return True


# 모델 로드
def load_model_and_tokenizer(cfg):
    """
    모델을 로드하는 부분
    """
    # 저장된 모델의 경로만을 가지고 인코더 토크나이저와 디코더 토크나이저, 모델을 불러오는 코드를 작성한다.
    enc_name = cfg.encoder_name
    dec_name = cfg.decoder_name
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(dec_name)


    model = GrafomerModel(enc_name, dec_name, cfg)

    model.encoder.load_state_dict(torch.load(cfg.encoder_path))
    model.decoder.load_state_dict(torch.load(cfg.decoder_path))
    model.graft_module.load_state_dict(torch.load(cfg.graft_path))
    
    return model, encoder_tokenizer, decoder_tokenizer


# 데이터셋 => 데이터 전처리가 완료됨 데이터로더
# cfg는 decoder need_prefix와 batch_size만 설정하면 될듯
def preprocess_function(encoder_tokenizer, decoder_tokenizer, data_path, cfg):
    """
    전처리한 데이터셋 출력
    """
    dataset = load_from_disk(data_path)
    dataset = dataset.map(remove_colon)
    
    dataset = dataset.filter(lambda x: x['target'] != "")
    
    fn_kwargs = {"max_length": 512, "max_target_length": 1024}
    preprocess_dataset = dataset.map(preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, need_prefix=cfg.need_prefix), fn_kwargs = fn_kwargs)

    
    # 너무 길거나 짧은 문장은 제외
    # preprocess_dataset = preprocess_dataset.filter(filter_length)     
    return preprocess_dataset

def save_to_csv(preds, gt, output_path, sacre_bleu_results):
    pd.DataFrame({
        "preds": preds,
        "gt": gt
        }).to_csv(os.path.join(output_path, f"{sacre_bleu_results}.csv"), index=False)

@hydra.main(config_path='./configs', config_name="eval_config.yaml")
def main(cfg):
    
    print("\n====== Using Hydra Configurations ======")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    model, encoder_tokenizer, decoder_tokenizer = load_model_and_tokenizer(cfg)

    
    preprocess_dataset = preprocess_function(encoder_tokenizer, decoder_tokenizer, cfg.data_path, cfg)
    

    
    data_collator = CustomDataCollator(encoder_pad_token_id=encoder_tokenizer.pad_token_id, decoder_pad_token_id=decoder_tokenizer.pad_token_id)
    dataloader = DataLoader(preprocess_dataset, batch_size=cfg.batch_size, pin_memory=False, shuffle=False, drop_last=False, num_workers=4, collate_fn=data_collator)
    
    
        
    # 모델을 GPU로 업로드
    device = torch.device('cuda:0')
    model.to(device)
    
    sacre_bleu = load_metric("sacrebleu")
    
    model.eval()
    eval_progress_bar = tqdm(range(len(dataloader)), ncols=100)
    eval_loss = 0

    # 예시 문장 추가
    preds, gt = [], []
    print("전처리 확인!!!!!")
    print(encoder_tokenizer.batch_decode(next(iter(dataloader))["input_ids"]))
    print(decoder_tokenizer.batch_decode(next(iter(dataloader))["decoder_input_ids"]))
    

    for eval_step, eval_batch in enumerate(dataloader):

        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        """
        output = my_model.generate(input_ids, attention_mask=attention_mask , max_length=1025,
        pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        num_beams=5, temperature=0.9, top_k=50, top_p=1.0, 
        repetition_penalty=1.0, use_cache=True)
        """
        with torch.no_grad():
            generated_tokens = model.generate(eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], max_length=int(eval_batch["input_ids"].shape[1] * 1.3),
                                            pad_token_id=decoder_tokenizer.pad_token_id, eos_token_id=decoder_tokenizer.eos_token_id, bos_token_id=decoder_tokenizer.bos_token_id,
                                            do_sample=True, top_k=50, top_p=0.95, repetition_penalty=1.2)
            labels = eval_batch["labels"]

            decoded_preds = decoder_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            sacre_bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
            preds.extend(decoded_preds)
            gt.extend(decoded_labels)

        eval_progress_bar.update()
        
    
    sacre_bleu_results = sacre_bleu.compute()
    print(f'bleu score: {sacre_bleu_results}')
    save_to_csv(preds, gt, cfg.output_path, cfg.output_name) # 경로는 직접 지정해주세요. sacre_bleu_score를 파일이름으로 쓰면 더 좋을듯 합니다.
    
    eval_progress_bar.close()

if __name__ == "__main__":
    
    main()