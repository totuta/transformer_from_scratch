import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Norm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()

    self.size = d_model

    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    self.eps = eps

  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
  # TODO: Check why .transpose(-2,-1)
  scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)

  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9) # if mask value is 0, then the negative infinite

  scores = F.softmax(scores, dim=-1)

  if dropout is not None:
    scores = dropout(scores)

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
    
    self.drop_out = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask=None):
    bs = q.size(0) # 배치 사이즈는 첫 디멘전

    # 여기서 -1 은 sequence length 임
    q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  
    k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
    v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

    # 차원이 n_batch * head * seq_len * d_model_per_head 가 되도록 변형
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    # TODO: attention 이 왜 갑자기 나와?
    scores = attention(q, k, v, self.d_k, mask, self.drop_out)
    # .contiguous() 는 메모리상에서 연속적으로 배열되게 한다고 함
    concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)

    output = self.out(concat)

    return output

    # 마지막 피드포워드 네트워크임
class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout=0.1):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    return self.linear_2(self.dropout(F.relu(self.linear_1(x))))