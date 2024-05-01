import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size)) # Transformer Layer Normalization 에서 alpha 와 bias 는 per dimension
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps # 이거는 nn.Parameter 로 선언되지 않았기 때문에 학습가능한 매개변수가 아님

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    # why .transpose(-2,-1)
    #  --> seq_len_q * d_model_per_head X d_model_per_head * seq_len_k 으로 만들어서 곱하기 위하여
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
    # scores 의 dimension 은 n_batch * head * seq_len_q * seq_len_k 가 됨
    # 물론 seq_len_q 와 seq_len_k 는 서로 같음

    if mask is not None:
        mask = mask.unsqueeze(1) # The unsqueeze operation is performed to match the dimensions of the scores tensor. 추가된 dimension 은 head 의 dimension 임. head dimension 을 따라서 broadcasting 될 것
        scores = scores.masked_fill(mask == 0, -1e9) # if mask value is 0, then the negative infinite 왜냐하면, 0 은 numerical instability 를 유발할 수 있으므로. 근데 꼭 필요한가?

    scores = F.softmax(scores, dim=-1) # seq_len_k 에 대해서 softmax 를 취함

    if dropout is not None:
        scores = dropout(scores)

    # n_batch * head * seq_len_q * seq_len_k X n_batch * head * seq_len_k * d_model_per_head
    #                              weights                      k-set of each dim
    # --> n_batch * head * seq_len_q * d_model_per_head
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model) # out matrix 가 따로 있다는 사실을 기억해야 함

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # 배치 사이즈는 첫 디멘전.

        # 여기서 -1 은 sequence length 임
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # 뒤의 h 와 d_k 로 분리하겠다는 이야기
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 차원이 n_batch * head * seq_len * d_model_per_head 가 되도록 변형
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # .contiguous() 는 메모리상에서 연속적으로 배열되게 한다고 함
        # 마지막 두 dimension, 즉 self.h 와 self.d_k 를 하나로 합쳐서 self.d_model 이 되도록 함
        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)

        output = self.out(concat) # 마지막 차원인 d_model 에 대해서만 linear transformation 을 수행함

        return output

    # 마지막 피드포워드 네트워크임
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x