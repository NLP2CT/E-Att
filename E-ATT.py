import torch
from torch import nn
from torch.nn.functional import softmax

from model.layer_norm import Identity, LayerNorm, UnlearnableLayerNorm

snn_threshold = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(MultiHeadSelfAttention, self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_of_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.factor = self.num_of_heads

        self.masked_value = float('-inf')

        self.attention_norm_factor = ((self.emb_size//fac) / self.num_of_heads) ** 0.5

        self.linear_in_weight = nn.Parameter(torch.zeros(size=(self.emb_size, 2 * self.emb_size // self.factor)), requires_grad=True)
        self.linear_in_bias = nn.Parameter(torch.zeros(size=(2 * self.emb_size // self.factor, )), requires_grad=True)

        self.linear_in_weight_v = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_in_bias_v = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        self.linear_out_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_out_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        self.dropout_attention = nn.Dropout(p=self.attention_dropout_prob)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob, inplace=True)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_qkv_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_pre == 'static':
            self.layer_norm_qkv_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_qkv_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_v_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_v_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_v_post = Identity()

        return

    def init_parameters(self):
        bound = (6 / self.emb_size / (1 + 2 / self.factor)) ** 0.5
        self.linear_in_weight.uniform_(-bound, bound)
        bound = (3 / self.emb_size * self.factor) ** 0.5
        self.linear_in_bias.uniform_(-bound, bound)

        bound = (3 / self.emb_size) ** 0.5
        self.linear_in_weight_v.uniform_(-bound, bound)
        self.linear_in_bias_v.uniform_(-bound, bound)
        self.linear_out_weight.uniform_(-bound, bound)
        self.linear_out_bias.uniform_(-bound, bound)

        return

    def forward(self, input_qkv, input_mask):
        normalized_qkv = self.layer_norm_qkv_pre(input_qkv)
        batch_size, length_out, _ = input_qkv.size()

        value = torch.add(torch.matmul(normalized_qkv, self.linear_in_weight_v), self.linear_in_bias_v)
        query, key = torch.split(torch.add(torch.matmul(spiking(normalized_qkv), self.linear_in_weight),
                                                  self.linear_in_bias), self.emb_size // self.factor, dim=-1)

        query = query.reshape(batch_size, length_out, self.num_of_heads, -1).transpose(1, 2)
        key = key.reshape(batch_size, length_out, self.num_of_heads, -1).transpose(1, 2)

        value = value.reshape(batch_size, length_out, self.num_of_heads, -1).transpose(1, 2)

        alignments = (query.unsqueeze(dim =-2)- key.unsqueeze(dim =-3)).norm(p=1,dim =-1)/ self.attention_norm_factor
        alignments = torch.neg(alignments)
        
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))

        context_vector = alignment_scores.matmul(value).transpose(1, 2).contiguous(). \
            view(batch_size, length_out, -1).contiguous()

        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight), self.linear_out_bias))
        return self.layer_norm_v_post(output + input_qkv)

    def forward_for_infer(self, input_qkv, buffers):
        normalized_qkv = self.layer_norm_qkv_pre(input_qkv)
        batch_size, beam_size, _, _ = input_qkv.size()
        
        buffered_k, buffered_v = buffers
        value = torch.add(torch.matmul(normalized_qkv, self.linear_in_weight_v), self.linear_in_bias_v)
        query, key= torch.split(torch.add(torch.matmul(spiking(normalized_qkv), self.linear_in_weight),
                                        self.linear_in_bias), self.emb_size // self.factor, dim=-1)

        query = query.reshape(batch_size, beam_size, 1, self.num_of_heads, -1).transpose(2, 3)
        key = key.reshape(batch_size, beam_size, 1, self.num_of_heads, -1).transpose(2, 3)
        value = value.reshape(batch_size, beam_size, 1, self.num_of_heads, -1).transpose(2, 3)

        buffered_k = torch.cat(tensors=(buffered_k, key), dim=3)
        buffered_v = torch.cat(tensors=(buffered_v, value), dim=3)

        alignments = (query.unsqueeze(dim =-2)- buffered_k.unsqueeze(dim =-3)).norm(p=1,dim =-1)/ self.attention_norm_factor
        alignments = torch.neg(alignments)
        
        alignment_scores = self.dropout_attention(softmax(alignments, dim=-1))

        context_vector = alignment_scores.matmul(buffered_v).transpose(2, 3).contiguous(). \
            view(batch_size, beam_size, 1, -1).contiguous()
        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight),
                                                 self.linear_out_bias))
        return self.layer_norm_v_post(output + input_qkv), (buffered_k, buffered_v)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class MultiHeadCrossAttention(nn.Module):
    def __init__(self,
                 emb_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(MultiHeadCrossAttention, self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_of_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.factor = self.num_of_heads

        self.masked_value = float('-inf')

        self.attention_norm_factor = (self.emb_size / self.num_of_heads) ** 0.5

        self.linear_q_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size // self.factor)), requires_grad=True)
        self.linear_q_bias = nn.Parameter(torch.zeros(size=(self.emb_size // self.factor, )), requires_grad=True)
        self.linear_kv_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size // self.factor)), requires_grad=True)
        self.linear_kv_bias = nn.Parameter(torch.zeros(size=(self.emb_size // self.factor, )), requires_grad=True)

        self.linear_out_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_out_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)
        
        self.linear_in_weight_kv = nn.Parameter(torch.zeros(size=(self.emb_size, self.emb_size)), requires_grad=True)
        self.linear_in_bias_kv = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        self.dropout_attention = nn.Dropout(p=self.attention_dropout_prob)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob, inplace=True)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_q_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            # self.layer_norm_q_pre = Identity()
            # self.layer_norm_kv_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_kv_pre = Identity()
        elif self.layer_norm_pre == 'static':
            self.layer_norm_q_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            # self.layer_norm_q_pre = Identity()
            # self.layer_norm_kv_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
            self.layer_norm_kv_pre = Identity()
        else:
            self.layer_norm_q_pre = Identity()
            self.layer_norm_kv_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_v_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_v_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_v_post = Identity()

        return

    def init_parameters(self):
        bound = (6 / self.emb_size / (1 + 1 / self.factor)) ** 0.5 # xavier uniform
        self.linear_kv_weight.uniform_(-bound, bound)
        bound = (3 / self.emb_size * self.factor) ** 0.5
        self.linear_kv_bias.uniform_(-bound, bound)


        bound = (6 / self.emb_size / (1 + 1 / self.factor)) ** 0.5
        self.linear_q_weight.uniform_(-bound, bound)
        bound = (3 / self.emb_size * self.factor) ** 0.5
        self.linear_q_bias.uniform_(-bound, bound)

        bound = (3 / self.emb_size) ** 0.5
        self.linear_in_weight_kv.uniform_(-bound, bound)
        self.linear_in_bias_kv.uniform_(-bound, bound)

        self.linear_out_weight.uniform_(-bound, bound)
        self.linear_out_bias.uniform_(-bound, bound)

        return

    def forward(self, input_q, input_kv, input_mask):
        normalized_q = self.layer_norm_q_pre(input_q)
        normalized_kv = self.layer_norm_kv_pre(input_kv)
        length_in = input_kv.size(1)
        batch_size, length_out, _ = input_q.size()

        query = torch.add(torch.matmul(spiking(normalized_q), self.linear_q_weight), self.linear_q_bias)
        key = torch.add(torch.matmul(spiking(normalized_kv), self.linear_kv_weight), self.linear_kv_bias)
        value = torch.add(torch.matmul(normalized_kv, self.linear_in_weight_kv), self.linear_in_bias_kv)

        query = query.reshape(batch_size, length_out, self.num_of_heads, -1).transpose(1, 2)
        key = key.reshape(batch_size, length_in, self.num_of_heads, -1).transpose(1, 2)
        value = value.reshape(batch_size, length_in, self.num_of_heads, -1).transpose(1, 2)

        alignments = (query.unsqueeze(dim =-2)-key.unsqueeze(dim =-3)).norm(p=1,dim =-1)/ self.attention_norm_factor
        alignments = torch.neg(alignments)
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))

        context_vector = alignment_scores.matmul(value).transpose(1, 2).contiguous().view(batch_size, length_out, -1).contiguous()

        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight), self.linear_out_bias))
        return self.layer_norm_v_post(output + input_q)

    def forward_for_infer(self, input_q, input_mask, buffers):
        normalized_q = self.layer_norm_q_pre(input_q)
        buffered_k, buffered_v = buffers

        batch_size, beam_size, _, _ = input_q.size()

        query = torch.add(torch.matmul(spiking(normalized_q), self.linear_q_weight), self.linear_q_bias)
        query = query.reshape(batch_size, beam_size, 1, self.num_of_heads, -1).transpose(2, 3)
        
        alignments = (query.unsqueeze(dim =-2) - buffered_k.unsqueeze(dim =-3)).norm(p=1,dim =-1)/ self.attention_norm_factor
        alignments = torch.neg(alignments)\
        alignments_masked = alignments.masked_fill(mask=input_mask.unsqueeze(dim=1), value=self.masked_value)
        alignment_scores = self.dropout_attention(softmax(alignments_masked, dim=-1))
        
        context_vector = alignment_scores.matmul(buffered_v).transpose(2, 3).contiguous().view(batch_size, beam_size, 1, -1).contiguous()
        output = self.dropout_residual(torch.add(torch.matmul(context_vector, self.linear_out_weight), self.linear_out_bias))
        
        return self.layer_norm_v_post(output + input_q), (buffered_k, buffered_v)

    def get_kv_buffers(self, input_kv):
        normalized_kv = self.layer_norm_kv_pre(input_kv)
        batch_size, _, length_in, _ = input_kv.size()
        
        value = torch.add(torch.matmul(normalized_kv, self.linear_in_weight_kv), self.linear_in_bias_kv)
        
        key= torch.add(torch.matmul(spiking(normalized_kv), self.linear_kv_weight), self.linear_kv_bias)
        key = key.reshape(batch_size, length_in, self.num_of_heads, -1).transpose(1, 2).unsqueeze(dim=1)
        value = value.reshape(batch_size, length_in, self.num_of_heads, -1).transpose(1, 2).unsqueeze(dim=1)
        return key, value
        
    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class SNNActivateFunction_Normal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(snn_threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temp = 0.79788456 * torch.exp(-2.0 * input ** 2)
        return grad_output * temp

spiking = SNNActivateFunction_Normal.apply
