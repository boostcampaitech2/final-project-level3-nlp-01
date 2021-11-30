import yaml
from datasets import load_from_disk

def read_yaml(cfg):
    try:
        with open(cfg) as f:
            model_args, data_args, training_args = yaml.load_all(f, Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print("Config File Not Found:", e)
    return model_args, data_args, training_args


def preprocess_function_with_setting(encoder_tokenizer, decoder_tokenizer, switch_language_pair=False):
    def preprocess_function(examples, max_length, max_target_length):

        src_sentence = examples["korean"] if not switch_language_pair else examples["japanese"]
        tgt_sentence = examples["japanese"] if not switch_language_pair else examples["korean"]
        
        tgt_sentence = [decoder_tokenizer.bos_token + ex for ex in tgt_sentence]

        model_inputs = encoder_tokenizer(src_sentence, max_length=max_length, padding="max_length", truncation=True)
        decoder_inputs = decoder_tokenizer(tgt_sentence, max_length=max_target_length, padding="max_length", truncation=True)

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
    raw_dataset = load_from_disk(path)
    return raw_dataset["train"], raw_dataset["validation"]