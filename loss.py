from torch.nn import MSELoss

import torch.nn as nn
import torch
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, pad_token_id, decoder_token_length):
        super().__init__()
        self.loss_mse = MSELoss()
        self.temperature = 1.0
        self.alpha = 0.35
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_token_id)   
        self.length = decoder_token_length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    
    def forward(self, teacher_outputs, student_outputs, labels):
        teacher_encoder_last_hidden = teacher_outputs.logits
        teacher_encoder_hidden, teacher_encoder_attns = teacher_outputs.encoder_hidden_states, teacher_outputs.encoder_attentions
        teacher_decoder_hidden, teacher_decoder_attns = teacher_outputs.decoder_hidden_states, teacher_outputs.decoder_attentions
        # teacher_graft_hidden, teacher_graft_attns = teacher_outputs.graft_hidden_states, teacher_outputs.graft_attentions

        student_encoder_last_hidden = student_outputs.logits
        student_encoder_hidden, student_encoder_attns = student_outputs.encoder_hidden_states, student_outputs.encoder_attentions
        student_decoder_hidden, student_decoder_attns = student_outputs.decoder_hidden_states, student_outputs.decoder_attentions
        # student_graft_hidden, student_graft_attns = student_outputs.graft_hidden_states, student_outputs.graft_attentions

        
        # teacher_encoder_last_hidden, teacher_encoder_hidden, teacher_encoder_attn = teacher_model.encoder()
        # teacher_decoder_last_hidden, teacher_decoder_hidden, teacher_decoder_attn = teacher_model.decoder.decoder_body(use_cache=False)
        # teacher_graft_hidden, teacher_graft_attn = teacher_model.graft_module()

        teacher_encoder_reps = [teacher_encoder_rep.detach() for teacher_encoder_rep in teacher_encoder_hidden]  # speedup 1.5x
        teacher_encoder_atts = [teacher_encoder_att.detach() for teacher_encoder_att in teacher_encoder_attns]
        teacher_decoder_reps = [teacher_decoder_rep.detach() for teacher_decoder_rep in teacher_decoder_hidden]  # speedup 1.5x
        teacher_decoder_atts = [teacher_decoder_att.detach() for teacher_decoder_att in teacher_decoder_attns]
        # teacher_graft_reps = [teacher_graft_rep.detach() for teacher_graft_rep in teacher_graft_hidden]  # speedup 1.5x
        # teacher_graft_atts = [teacher_graft_att.detach() for teacher_graft_att in teacher_graft_attns]

        student_encoder_layer_num = len(student_encoder_attns) # 각 layer의 개수
        student_decoder_layer_num = len(student_decoder_attns)
        # studnet_graft_layer_num = len(student_graft_attn)

        teacher_encoder_layer_num = len(teacher_encoder_atts)
        teacher_decoder_layer_num = len(teacher_decoder_atts)
        # teacher_graft_layer_num = len(teacher_graft_atts)

        assert teacher_encoder_layer_num % student_encoder_layer_num == 0
        assert teacher_decoder_layer_num % student_decoder_layer_num == 0
        # assert teacher_graft_layer_num % student_graft_layer_num == 0

        encoder_layers_per_block = int(teacher_encoder_layer_num / student_encoder_layer_num)
        decoder_layers_per_block = int(teacher_decoder_layer_num / student_decoder_layer_num)
        # graft_layers_per_block = int(teacher_graft_layer_num / student_graft_layer_num)

        # Attention matrix
        new_teacher_encoder_atts = [teacher_encoder_atts[i * encoder_layers_per_block + encoder_layers_per_block -1] for i in range(student_encoder_layer_num)]
        new_teacher_decoder_atts = [teacher_decoder_atts[i * decoder_layers_per_block + decoder_layers_per_block -1] for i in range(student_decoder_layer_num)]
        # new_teacher_graft_atts = [teacher_atts[i * graft_layers_per_block + graft_layers_per_block -1] for i in range(student_graft_layer_num)]

        # Attention Loss
        encoder_att_loss = 0 
        for student_att, teacher_att in zip(student_encoder_attns, new_teacher_encoder_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device), teacher_att)
            encoder_att_loss += self.loss_mse(student_att, teacher_att)
        
        decoder_att_loss = 0 
        for student_att, teacher_att in zip(student_decoder_attns, new_teacher_decoder_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device), teacher_att)
            decoder_att_loss += self.loss_mse(student_att, teacher_att)
        
        # encoder_att_loss /= student_encoder_layer_num
        # decoder_att_loss /= student_decoder_layer_num

        # graft_att_loss = 0
        # for student_att, teacher_att in zip(student_graft_attn, new_teacher_graft_atts):
        #     student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device), student_att)
        #     teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device), teacher_att)
        #     graft_att_loss += self.loss_mse(student_att, teacher_att)


        # Hidden states loss + Embedding loss
        new_teacher_encoder_reps = [teacher_encoder_reps[i * encoder_layers_per_block] for i in range(student_encoder_layer_num + 1)]
        new_teacher_decoder_reps = [teacher_decoder_reps[i * decoder_layers_per_block] for i in range(student_decoder_layer_num + 1)]
        # new_teacher_graft_reps = [teacher_reps[i * graft_layers_per_block] for i in range(student_graft_layer_num + 1)]
        
        encoder_rep_loss = 0
        for student_rep, teacher_rep in zip(student_encoder_hidden, new_teacher_encoder_reps):
            encoder_rep_loss += self.loss_mse(student_rep, teacher_rep)
        
        decoder_rep_loss = 0
        for student_rep, teacher_rep in zip(student_decoder_hidden, new_teacher_decoder_reps):
            decoder_rep_loss += self.loss_mse(student_rep, teacher_rep)
        
        # graft_rep_loss = 0
        # for student_rep, teacher_rep in zip(student_graft_hidden, new_teacher_graft_reps):
        #     graft_rep_loss += self.loss_mse(student_rep, teacher_rep)
        
        # prediction_loss
        student_logits = student_encoder_last_hidden.view(-1, self.length)
        teacher_logits = teacher_encoder_last_hidden.view(-1, self.length)

        pred_loss = self.soft_cross_entropy(student_logits/self.temperature, teacher_logits/self.temperature)
        
        cls_loss = self.cross_entropy(student_logits, labels)

        att_loss = encoder_att_loss + decoder_att_loss#  + graft_att_loss
        rep_loss = encoder_rep_loss + decoder_rep_loss# + graft_rep_loss

        pred_loss = (1-self.alpha)*cls_loss + self.alpha*pred_loss

        return att_loss, rep_loss, pred_loss

class DistilLoss(nn.Module):
    def __init__(self, pad_token_id, decoder_token_length):
        super().__init__()
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0

        # self.alpha_ce = 5.0 
        # self.alpha_mlm = 2.0
        # self.alpha_cos = 1.0
        # self.alpha_clm = 0.0
        # self.alpha_mse = 0.0

        self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss_fct = nn.MSELoss(reduction='sum')
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')

        # self.loss_mse = MSELoss()
        self.temperature = 1.0
        # self.alpha = 0.35
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        # self.length = decoder_token_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    
    def forward(self, teacher_outputs, student_outputs, labels, attention_mask, decoder_attention_mask):
        teacher_logit = teacher_outputs.logits
        teacher_encoder_hidden, teacher_encoder_attns = teacher_outputs.encoder_hidden_states, teacher_outputs.encoder_attentions
        teacher_decoder_hidden, teacher_decoder_attns = teacher_outputs.decoder_hidden_states, teacher_outputs.decoder_attentions

        student_logit = student_outputs.logits
        student_encoder_hidden, student_encoder_attns = student_outputs.encoder_hidden_states, student_outputs.encoder_attentions
        student_decoder_hidden, student_decoder_attns = student_outputs.decoder_hidden_states, student_outputs.decoder_attentions

        #Loss
        loss_ce = self.ce_loss_fct(F.log_softmax(student_logit/self.temperature, dim=-1),
                                   F.softmax(teacher_logit/self.temperature, dim=-1)) * (self.temperature)**2
        
        loss_mlm = self.lm_loss_fct(student_logit.view(-1, student_logit.size(-1)), labels.view(-1))

        # shift_logits = student_logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))

        # loss_mse = self.mse_loss_fct(student_encoder_last_hidden, teacher_encoder_last_hidden)/student_encoder_last_hidden.size(0) # Reproducing batchmean reduction

        # encoder
        s_en_hidden_states = student_encoder_hidden[-1]                              # (bs, seq_length, dim)
        t_en_hidden_states = teacher_encoder_hidden[-1]                              # (bs, seq_length, dim)
        en_mask = attention_mask.unsqueeze(-1).expand_as(s_en_hidden_states)     # (bs, seq_length, dim)
        en_mask = torch.as_tensor(en_mask, dtype=bool)
        assert s_en_hidden_states.size() == t_en_hidden_states.size()
        en_dim = s_en_hidden_states.size(-1)

        s_en_hidden_states_slct = torch.masked_select(s_en_hidden_states, en_mask)        # (bs * seq_length * dim)
        s_en_hidden_states_slct = s_en_hidden_states.view(-1, en_dim)                # (bs * seq_length, dim)
        t_en_hidden_states_slct = torch.masked_select(t_en_hidden_states, en_mask)        # (bs * seq_length * dim)
        t_en_hidden_states_slct = t_en_hidden_states.view(-1, en_dim)                # (bs * seq_length, dim)
    
        en_target = s_en_hidden_states_slct.new(s_en_hidden_states_slct.size(0)).fill_(1) # (bs * seq_length,)
        en_loss_cos = self.cosine_loss_fct(s_en_hidden_states_slct, t_en_hidden_states_slct, en_target)

        # decoder
        s_de_hidden_states = student_decoder_hidden[-1]                              # (bs, seq_length, dim)
        t_de_hidden_states = teacher_decoder_hidden[-1]                              # (bs, seq_length, dim)
        de_mask = decoder_attention_mask.unsqueeze(-1).expand_as(s_de_hidden_states)     # (bs, seq_length, dim)
        de_mask = torch.as_tensor(de_mask, dtype=bool)
        assert s_de_hidden_states.size() == t_de_hidden_states.size()
        de_dim = s_de_hidden_states.size(-1)
        
        s_de_hidden_states_slct = torch.masked_select(s_de_hidden_states, de_mask)        # (bs * seq_length * dim)
        s_de_hidden_states_slct = s_de_hidden_states.view(-1, de_dim)                # (bs * seq_length, dim)
        t_de_hidden_states_slct = torch.masked_select(t_de_hidden_states, de_mask)        # (bs * seq_length * dim)
        t_de_hidden_states_slct = t_de_hidden_states.view(-1, de_dim)                # (bs * seq_length, dim)
    
        de_target = s_de_hidden_states_slct.new(s_de_hidden_states_slct.size(0)).fill_(1) # (bs * seq_length,)
        de_loss_cos = self.cosine_loss_fct(s_de_hidden_states_slct, t_de_hidden_states_slct, de_target)

        loss_cos = en_loss_cos + de_loss_cos
        
        # loss = (self.alpha_ce*loss_ce) + (self.alpha_mlm * loss_mlm) + (self.alpha_clm * loss_clm) + (self.alpha_mse * loss_mse) + (self.alpha_cos * loss_cos)
        # loss = (self.alpha_ce*loss_ce) + (self.alpha_mlm * loss_mlm) + (self.alpha_cos * loss_cos)

        return loss_ce, loss_mlm, loss_cos