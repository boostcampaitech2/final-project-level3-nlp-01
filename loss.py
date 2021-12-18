from torch.nn import MSELoss

import torch.nn as nn
import torch

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