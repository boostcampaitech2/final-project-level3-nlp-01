import yaml
import torch

from model import GrafomerModel


def load_yaml(path):
    with open(f"{path}/config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def model_init(path):
    cfg = load_yaml(path)
    enc_name = cfg["encoder"]["name"]
    dec_name = cfg["decoder"]["name"]

    encoder_tokenizer = getattr(
        __import__("transformers"), cfg["encoder"]["tokenizer"]
    ).from_pretrained(enc_name, is_Fast=True)
    decoder_tokenizer = getattr(
        __import__("transformers"), cfg["decoder"]["tokenizer"]
    ).from_pretrained(dec_name, is_Fast=True)

    model = GrafomerModel(enc_name, dec_name, cfg)
    model.encoder.load_state_dict(torch.load(f"{path}/encoder.pt"))
    model.decoder.load_state_dict(torch.load(f"{path}/decoder.pt"))
    model.graft_module.load_state_dict(torch.load(f"{path}/graft_module.pt"))

    model.cuda()
    model.eval()

    @torch.no_grad()
    def generate(text):
        model_inputs = encoder_tokenizer(
            text, max_length=512, padding=False, truncation=True, return_tensors="pt"
        )
        generated_tokens = (
            model.generate(
                model_inputs["input_ids"].cuda(),
                attention_mask=model_inputs["attention_mask"].cuda(),
                max_length=int(model_inputs["input_ids"].shape[1] * 1.3),
                pad_token_id=decoder_tokenizer.pad_token_id,
                eos_token_id=decoder_tokenizer.eos_token_id,
                bos_token_id=decoder_tokenizer.bos_token_id,
                do_sample=True,
                repetition_penalty=1.2,
                top_k=50,
                top_p=0.95,
            )
            .cpu()
            .numpy()[0]
        )
        return decoder_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generate
