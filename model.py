from transformers import (
    AutoModel, 
    AutoConfig,
    PreTrainedModel,
)
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer, _expand_mask

import torch
from torch import nn


class Grafomer(PreTrainedModel):
  def __init__(self, encoder_config, decoder_config):
    super().__init__()
    
    self.encoder = nn.ModuleList([BartEncoderLayer(encoder_config) for _ in range(config_encoder.encoder_layers)])
    self.decoder = nn.ModuleList([BartDecoderLayer(decoder_config) for _ in range(config_decoder.decoder_layers)])
    
    self.init_weights()

  def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_hidden_states=None,
        decoder_attention_mask=None,
        cross_attention_mask=None,
        output_attentions: bool = False,
        cross_attn_head_mask=None,
        use_cache=None,
        head_mask=None,
    ):

    for idx, encoder_layer in enumerate(self.encoder):
      encoder_layer_outputs = encoder_layer(
              encoder_hidden_states,
              encoder_attention_mask,
              layer_head_mask=(head_mask[idx] if head_mask is not None else None),
              output_attentions=output_attentions,
          )

      encoder_hidden_states = encoder_layer_outputs[0]    
    
    for idx, decoder_layer in enumerate(self.decoder):
      decoder_layer_outputs = decoder_layer(
          decoder_hidden_states,
          attention_mask=decoder_attention_mask,
          encoder_hidden_states=encoder_hidden_states,
          encoder_attention_mask=cross_attention_mask,
          layer_head_mask=(head_mask[idx] if head_mask is not None else None),
          cross_attn_layer_head_mask=(
              cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
          ),
          past_key_value=None,
          output_attentions=output_attentions,
          use_cache=use_cache,
      )
      decoder_hidden_states = decoder_layer_outputs[0]

    return decoder_hidden_states


class GrafomerModel(PreTrainedModel):
  def __init__(self, encoder_name, decoder_name):
    super().__init__()
    
    # self.encoder = mBERT
    # self.decoder = mGPT

    self.encoder = AutoModel.from_pretrained(encoder_name)
    self.decoder = AutoModel.from_pretrained(decoder_name)

    self.encoder_config = AutoConfig.from_pretrained(encoder_name)
    self.decoder_config = AutoConfig.from_pretrained(decoder_name)
    
    self.graformer = Grafomer(self.encoder_config, self.decodeer_config)
    self.init_weights()
  
  def make_cross_mask(self, enc_mask, dec_mask, dtype):
    nc_mask = enc_mask.to(dtype)
    dec_mask = dec_mask.to(dtype)
    return dec_mask.view(-1, 1, dec_mask.shape[1], 1) @ enc_mask.view(-1, 1, 1, enc_mask.shape[1])
    
  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    
    encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    encoder_hidden_state = encoder_outputs[0]
    
    decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            use_cache=False,
        )
    decoder_hidden_state = decoder_outputs[0]

    mask = _expand_mask(attention_mask, encoder_hidden_state.dtype)
    dec_mask = _expand_mask(decoder_attention_mask, decoder_hidden_state.dtype)
    cross_mask = make_cross_mask(attention_mask, decoder_attention_mask, dec_mask.dtype)

    graformer_hidden_state = self.graformer(
      encoder_hidden_states=encoder_hidden_state,
      encoder_attention_mask=attention_mask,
      decoder_hidden_states=decoder_hidden_state,
      decoder_attention_mask=decoder_attention_mask,
      cross_attention_mask=cross_mask,
      output_attentions=output_attentions,
    )
    
    output_hidden_states = decoder_hidden_state + graformer_hidden_state

    return output_hidden_states



class CustomSeq2SeqModel(PreTrainedModel):
  def __init__(self, encoder_name, decoder_name):
    super().__init__()

    self.encoder = AutoModel.from_pretrained(encoder_name)
    self.decoder = AutoModel.from_pretrained(decoder_name)

    self.encoder_config = AutoConfig.from_pretrained(encoder_name)
    self.decoder_config = AutoConfig.from_pretrained(decoder_name)
    
    self.init_weights()
    
  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    
    encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    encoder_hidden_state = encoder_outputs[0]
    
    decoder_outputs = self.decoder(
            input_ids=decoder_input_ids
        )
    decoder_hidden_state = decoder_outputs[0]

    # output_hidden_states = decoder_hidden_state + graformer_hidden_state

    return output_hidden_states