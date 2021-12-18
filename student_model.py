import torch
from torch import nn
import copy

from transformers import AutoConfig
from transformers.generation_utils import GenerationMixin
from transformers.file_utils import PushToHubMixin
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer
from transformers.models.bart.modeling_bart import _expand_mask, _make_causal_mask

from teacher_model import GraformerModelOutput

class StudentGraftAttentionModule(nn.Module):
    def __init__(self, bart_config, model_config, embed_dim, teacher_model):
        super().__init__()
        
        # self.embed_dim = embed_dim
        # self.graft_encoder = copy.deepcopy(teacher_model.graft_module.graft_encoder[-1])
        # self.graft_decoder = copy.deepcopy(teacher_model.graft_module.graft_decoder[-1])
        # self.graft_input_pooler = copy.deepcopy(teacher_model.graft_module.graft_input_pooler)
        # self.graft_output_pooler = copy.deepcopy(teacher_model.graft_module.graft_output_pooler)
        self.embed_dim = embed_dim
        self.graft_encoder = nn.ModuleList([BartEncoderLayer(bart_config) for _ in range(model_config["num_enc_layer"])])
        self.graft_decoder = nn.ModuleList([BartDecoderLayer(bart_config) for _ in range(model_config["num_dec_layer"])])
        self.graft_input_pooler = nn.Linear(self.embed_dim, 768)
        self.graft_output_pooler = nn.Linear(768, self.embed_dim)


    def forward(
        self, 
        encoder_hidden_states: torch.Tensor, 
        encoder_attention_mask: torch.Tensor,
        decoder_hidden_states: torch.Tensor=None, 
        decoder_attention_mask: torch.Tensor=None, 
        cross_attention_mask: torch.Tensor=None,
        output_attentions: bool = False, 
        cross_attn_head_mask=None, 
        use_cache=False, 
        head_mask=None,
        ):

        decoder_hidden_states = self.graft_input_pooler(decoder_hidden_states)
        
        # hidden_states = []
        # attentions = []
        
        for idx, encoder_layer in enumerate(self.graft_encoder):
            encoder_layer_outputs = encoder_layer(
                encoder_hidden_states, 
                encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions
            )
            encoder_hidden_states = encoder_layer_outputs[0]  
            # print(encoder_layer_outputs)  
            # hidden_states.append(encoder_layer_outputs)
            # attentions.append(encoder_layer_outputs.attentions)
        for idx, decoder_layer in enumerate(self.graft_decoder):
            decoder_layer_outputs = decoder_layer(
                decoder_hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # decoder_hidden_states = decoder_layer_outputs[0]
            # hidden_states.append(encoder_layer_outputs.hidden_states)
            # attentions.append(encoder_layer_outputs.attentions)

        decoder_hidden_states = self.graft_output_pooler(decoder_hidden_states)

        return decoder_hidden_states# , # hidden_states, attentions


class StudentGrafomerModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
    def __init__(self, enc_name, dec_name, cfg, teacher_model):
        super().__init__()

        self.encoder_config = AutoConfig.from_pretrained(enc_name, num_hidden_layers=6, output_attentions=True, output_hidden_states=True)
        self.decoder_config = AutoConfig.from_pretrained(dec_name, num_hidden_layers=6, output_attentions=True, output_hidden_states=True)

        self.encoder = getattr(__import__("transformers"), cfg.encoder.model).from_pretrained(enc_name, config=self.encoder_config)
        # self.decoder = getattr(__import__("transformers"), cfg.decoder.model).from_pretrained(dec_name, config=self.decoder_config)

        self.encoder.embeddings = copy.deepcopy(teacher_model.encoder.embeddings)
        self.encoder.encoder.layer = nn.ModuleList([copy.deepcopy(layer) for i, layer in enumerate(teacher_model.encoder.encoder.layer) if i%2==1])
        self.encoder.pooler = copy.deepcopy(teacher_model.encoder.pooler)

        # self.decoder.transformers.wte = copy.deepcopy(teacher_model.decoder.transformers.wte)
        # self.decoder.transformers.wpe = copy.deepcopy(teacher_model.decoder.transformers.wpe)
        # self.decoder.transformers.h = nn.ModuleList([copy.deepcopy(layer) for i, layer in enumerate(teacher_model.decoder.transformers.h) if i%2==1])
        # self.decoder.transforemrs.ln_f = copy.deepcopy(teacher_model.decoder.ln_f)
        # self.decoder.lm_head = copy.deepcopy(teacher_model.decoder.lm_head)

        self.config = self.decoder_config  # for compatibility in generate method
        self.config.is_encoder_decoder = True
        self.config.decoder_start_token_id = cfg.decoder.decoder_start_token_id
        # print(self.config)

        self.decoder_body = copy.deepcopy(teacher_model.decoder_body)
        self.decoder_body.h = nn.ModuleList([copy.deepcopy(layer) for i, layer in enumerate(teacher_model.decoder_body.h) if i%2==1])
        self.decoder_head = copy.deepcopy(teacher_model.decoder_head)

        self.bart_config = AutoConfig.from_pretrained("facebook/bart-base")
        print(self.decoder_body)
        self.decoder_embed_dim = cfg.decoder.embed_dim
        self.graft_module = StudentGraftAttentionModule(self.bart_config, cfg.graft_module_config, self.decoder_embed_dim, teacher_model)
    
    def get_encoder(self):
        return self.encoder
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0],
                                            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                                            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        # train
        if decoder_input_ids is not None:
            decoder_outputs = self.decoder_body(input_ids=decoder_input_ids,
                                                attention_mask=decoder_attention_mask,
                                                use_cache=use_cache)

            mask = _expand_mask(attention_mask, self.dtype)
            dec_mask = _make_causal_mask(decoder_attention_mask.shape, self.dtype).to(self.device) + _expand_mask(decoder_attention_mask, self.dtype)
            cross_mask = _expand_mask(attention_mask, self.dtype, tgt_len=decoder_input_ids.shape[1])
        
        # eval
        else:
            decoder_outputs = self.decoder_body(input_ids=input_ids, use_cache=use_cache)

            bsz, sql = input_ids.shape
            mask = _expand_mask(attention_mask, self.dtype)
            dec_mask = _make_causal_mask([bsz, sql], self.dtype).to(self.device)
            cross_mask = _expand_mask(attention_mask, self.dtype, tgt_len=sql)

        # Attention
        # Hidden layer Output
        # Prediction Output
        encoder_hidden_state = encoder_outputs[0]
        decoder_hidden_state = decoder_outputs[0]
        encoder_hidden_states = encoder_outputs.hidden_states
        decoder_hidden_states = decoder_outputs.hidden_states
        encoder_attentions = encoder_outputs.attentions
        decoder_attentions = decoder_outputs.attentions

        graformer_hidden_state = self.graft_module(
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=mask,
            decoder_hidden_states=decoder_hidden_state,
            decoder_attention_mask=dec_mask,
            cross_attention_mask=cross_mask,
            output_attentions=output_attentions,
        )

        output_hidden_states = decoder_hidden_state + graformer_hidden_state
        output_hidden_states = self.decoder_head(output_hidden_states)

        return Seq2SeqLMOutput(
            logits=output_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            # graft_hidden_states=graft_hidden_states,
            # graft_attentions=graft_attentions,
        )
        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past is not None:
            # print(past, type(past))
            # print(len(past))
            # print(past.shape)
            pass
        return {"input_ids": input_ids, **kwargs}
