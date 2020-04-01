import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DKVMN_IRT(nn.Module):
    """Adapted from the TensorFlow implementation.
    https://github.com/ckyeungac/DeepIRT/blob/master/model.py
    """

    def __init__(
            self, 
            device,
            batch_size,
            n_questions, 
            memory_size, 
            memory_key_state_dim, 
            memory_value_state_dim,
            summary_vector_output_dim,
        ):
        super().__init__()

        self.n_questions = n_questions
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.summary_vector_output_dim = summary_vector_output_dim

        self.init_key_memory = torch.randn(self.memory_size, self.memory_key_state_dim).to(device)
        self.init_value_memory = torch.randn(self.memory_size, self.memory_value_state_dim).to(device)

        self.memory = DKVMN(
            self.memory_size,
            self.memory_key_state_dim,
            self.memory_value_state_dim,
            self.init_key_memory,
            self.init_value_memory.unsqueeze(0).repeat(batch_size, 1, 1),
        )
        self.q_embed_matrix = nn.Embedding(
            self.n_questions + 1, 
            self.memory_key_state_dim,
        )
        self.qa_embed_matrix = nn.Embedding(
            2 * self.n_questions + 1, 
            self.memory_value_state_dim,
        )
        self.summary_vector_fc = nn.Linear(
            self.memory_key_state_dim + self.memory_value_state_dim,
            self.summary_vector_output_dim,
        )
        self.student_ability_fc = nn.Linear(
            self.summary_vector_output_dim,
            1,
        )
        self.question_difficulty_fc = nn.Linear(
            self.memory_key_state_dim, 
            1,
        )

    def forward(self, q_data, qa_data, label):
        """
        q_data  : (batch_size, seq_len)
        qa_data : (batch_size, seq_len)
        label   : (batch_size, seq_len)
        """
        batch_size, seq_len = q_data.size(0), q_data.size(1)

        q_embed_data  = self.q_embed_matrix(q_data.long())
        qa_embed_data = self.qa_embed_matrix(qa_data.long())

        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)

        pred_zs, student_abilities, question_difficulties = [], [], []

        for i in range(seq_len):
            q = sliced_q_embed_data[i].squeeze(1)
            qa = sliced_qa_embed_data[i].squeeze(1)

            correlation_weight = self.memory.attention(q)
            read_content = self.memory.read(correlation_weight)
            new_memory_value = self.memory.write(correlation_weight, qa)

            mastery_level_prior_difficulty = torch.cat([read_content, q], dim=1)

            summary_vector = self.summary_vector_fc(
                mastery_level_prior_difficulty,
            )
            summary_vector = torch.tanh(summary_vector)
            student_ability = self.student_ability_fc(summary_vector)
            question_difficulty = self.question_difficulty_fc(q)
            question_difficulty = torch.tanh(question_difficulty)

            pred_z = 3.0 * student_ability - question_difficulty
            
            pred_zs.append(pred_z)
            student_abilities.append(student_ability)
            question_difficulties.append(question_difficulty)

        pred_zs = torch.cat(pred_zs, dim=1)
        student_abilities = torch.cat(student_abilities, dim=1)
        question_difficulties = torch.cat(question_difficulties, dim=1)

        return pred_zs, student_abilities, question_difficulties

    def get_loss(
            self,
            pred_z,
            student_ability,
            question_difficulty,
            label,
            epsilon = 1e-6,
        ):
        label_1d = label.view(-1)
        pred_z_1d = pred_z.view(-1)
        student_ability_1d = student_ability.view(-1)
        question_difficulty_1d = question_difficulty.view(-1)

        # remove missing data
        index = torch.where(label_1d != -1)[0]

        filtered_label = torch.gather(label_1d, 0, index)
        filtered_z = torch.gather(pred_z_1d, 0, index)
        filtered_student_ability = torch.gather(student_ability_1d, 0, index)
        filtered_question_difficulty = torch.gather(question_difficulty_1d, 0, index)

        filtered_pred = torch.sigmoid(filtered_z)

        # get prediction probability from logit
        clipped_filtered_pred = torch.clamp(
            filtered_pred,
            epsilon,
            1. - epsilon,
        )
        filtered_logits = torch.log(
            clipped_filtered_pred / (1. - clipped_filtered_pred),
        )

        loss = F.binary_cross_entropy_with_logits(filtered_logits, filtered_label)
        return loss


class DKVMN(nn.Module):
    """Adapted from the TensorFlow implementation.
    https://github.com/yjhong89/DKVMN/blob/master/model.py
    """

    def __init__(
            self,
            memory_size, 
            memory_key_state_dim, 
            memory_value_state_dim, 
            init_memory_key,
            init_memory_value,
        ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        
        self.key = DKVMN_Memory(
            self.memory_size,
            self.memory_key_state_dim,
        )
        self.value = DKVMN_Memory(
            self.memory_size,
            self.memory_value_state_dim,
        )
        self.memory_key = init_memory_key
        self.memory_value = init_memory_value

    def attention(self, q_embedded):
        correlation_weight = self.key.cor_weight(
            q_embedded, 
            self.memory_key,
        )
        return correlation_weight
    
    def read(self, c_weight):
        read_content = self.value.read(
            self.memory_value,
            c_weight,
        )
        return read_content

    def write(self, c_weight, qa_embedded):
        batch_size = c_weight.size(0)
        memory_value = self.value.write(
            self.memory_value,
            c_weight,
            qa_embedded,
        )
        self.memory_value = memory_value.detach()
        return memory_value


class DKVMN_Memory(nn.Module):
    """
    https://github.com/yjhong89/DKVMN/blob/master/memory.py
    """

    def __init__(self, memory_size, memory_state_dim):
        super().__init__()
        
        self.erase_linear = nn.Linear(memory_state_dim, memory_state_dim)
        self.add_linear = nn.Linear(memory_state_dim, memory_state_dim)

        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

    def cor_weight(self, embedded, key_matrix):
        """
        embedded : (batch size, memory_state_dim)
        key_matrix : (memory_size, memory_state_dim)
        """
        # (batch_size, memory_size)
        embedding_result = embedded @ key_matrix.t()
        correlation_weight = torch.softmax(embedding_result, dim=1)
        return correlation_weight

    def read(self, value_matrix, correlation_weight):
        """
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        """
        batch_size = value_matrix.size(0)
        vmtx_reshaped = value_matrix.view(
            batch_size * self.memory_size, 
            self.memory_state_dim,
        )
        cw_reshaped = correlation_weight.view(
            batch_size * self.memory_size,
            1
        )
        rc = vmtx_reshaped * cw_reshaped
        read_content = rc.view(
            batch_size,
            self.memory_size,
            self.memory_state_dim,
        )
        read_content = torch.sum(read_content, dim=1)

        return read_content

    def write(self, value_matrix, correlation_weight, qa_embedded):
        """
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        qa_embedded: (batch_size, memory_state_dim)
        """
        batch_size = value_matrix.size(0)

        erase_vector = self.erase_linear(qa_embedded)
        # (batch_size, memory_state_dim)
        erase_signal = torch.sigmoid(erase_vector)
        
        add_vector = self.add_linear(qa_embedded)
        # (batch_size, memory_state_dim)
        add_signal = torch.tanh(add_vector)

        erase_reshaped = erase_signal.view(
            batch_size,
            1,
            self.memory_state_dim,
        )
        cw_reshaped = correlation_weight.view(
            batch_size,
            self.memory_size,
            1,
        )
        erase_mul = erase_reshaped * cw_reshaped
        # (batch_size, memory_size, memory_state_dim)
        erase = value_matrix * (1 - erase_mul)
        
        # (batch_size, 1, memory_state_dim)
        add_reshaped = add_signal.view(
            batch_size, 
            1, 
            self.memory_state_dim,
        )
        add_mul = add_reshaped * cw_reshaped

        new_memory = erase + add_mul
        # (batch_size, memory_size, memory_state_dim)
        return new_memory
