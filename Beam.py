import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):

    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2) # 1 x 1 x 1 <-- initialization 이므로
    e_output = model.encoder(src, src_mask) # 1 x 1 x d_model

    outputs = torch.LongTensor([[init_tok]]) # 일단 outputs 은 <sos> 하나만 가지고 시작
    if opt.device == torch.device('cuda'):
        outputs = outputs.cuda()

    trg_mask = nopeak_mask(1, opt) # [[[True]]] : 1 x 1 x 1

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask)) # 1 x 현재까지 진행된 토큰수 x trg_vocab_len
    out = F.softmax(out, dim=-1) # 형태 그대로 남기고 softmaxing 만 함. argmaxing 은 추가로 해줘야 함

    probs, ix = out[:, -1].data.topk(opt.k) # 1 x opt.k 사이즈. [:,-1] 는 차원하나 줄이는 역할. .data 는 없어도 됨. opt.k=3 이므로 top 3 token 의 softmaxed prob 과 indicex(ix) 를 리턴.

    log_probs = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0) # unsqueeze 하는 것은 shape [3] 을 원래대로 [1, 3] 으로 원복하기 위함

    # 위에서 [1,1] 사이즈의 초기 outputs 값이 한번 존재했었음. 그거를 확장해서 overwrite 하는 감각
    # 그러니까 같은 이름의 변수가 두 가지 형태로 쓰여서 혼동을 유발할 수 있음
    outputs = torch.zeros(opt.k, opt.max_len).long() # k x maxlen 형태로 0을 채움
    if opt.device == torch.device('cuda'):
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok # 0번째 토큰의 top 3 는 모두 <sos> 로 고정. broadcast
    outputs[:, 1] = ix[0] # 1번째 토큰의 top3 는 바로 위에서 src sentence 와 <sos> 를 가지고 만든 첫번째 토큰향 top3 를 넣음
    # outputs 을 0th, 1st 까지 opt.k 개를 채워서 내보낼 준비

    # encoder_outputs 임. e_output 의 opt.k 만큼의 복제 틀
    # opt.k x sent_len x d_model
    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1)) # e_outputs 는 여기서 처음 만들어짐
    if opt.device == torch.device('cuda'):
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0] # e_output 의 0번 index 를 k개 만큼 복제
    # 그러니까 encoder output 을 그냥 단순히 k개 만큼 복제한 것. 계산 편의를 위해서일 것

    # outputs: k x max_len 이고, 0th, 1st 까지는 index 로 채워져 있음. 빈 값들은 beam 과정에서 채워져 갈 것
    # e_outputs: k x src_len x d_model 이고, src sentence 의 encoder output 이 k 셋트 있음. 아마 이 값은 안 바뀔 것
    # log_probs: 1 x k 이고, 1st output 의 top k 토큰들의 log_prob 들임. 아마도 계속 바뀌어 가겠지?
    return outputs, e_outputs, log_probs

def k_best_outputs(outputs, out, log_probs, i, k):

    # [:, -1] 는 마지막 컬럼, 즉, 여기서는 i번째 토큰의 모든 prob
    # torch.Tensor 의 .topk() 는 거기서 top k 값과 그 indices 를 리턴
    probs, ix = out[:, -1].data.topk(k)
    # 여기가 핵심임. k x k 전체를 계산하는 방법이고, log_prob 이므로 더하면 됨. 기존의 k 개가 각각 k 개씩 새끼를 친 값들의 전체
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) \
                + log_probs.transpose(0,1) # 이거는 broadcasting 될 것
    k_probs, k_ix = log_probs.view(-1).topk(k) # 이거는 k x k 전체에서 top k 선발

    row = k_ix // k # k_ix 는 flatten 된 indexing 이므로 k 로 나눠줘야 행과 열 좌표가 나옴
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i] # i번째 이전까지의 전임자를 자기 전임자로 재배열(비터비스럽게). 그렇게 해야 나중에 그 row 만 가져다가 바로 문장으로 디코딩할 수 있을 것이므로
    outputs[:, i] = ix[row, col] # i번째는 이번에 선발된 자들의 indices 를 넣음

    log_probs = k_probs.unsqueeze(0)

    return outputs, log_probs

def beam_search(src, model, SRC, TRG, opt):

    # outputs: k x max_len 이고, 0th, 1st 까지는 top k indices 로 채워져 있음. 빈 값들은 beam 과정에서 채워져 갈 것
    # e_outputs: k x src_len x d_model 이고, src sentence 의 encoder output 이 k 셋트 있음. 아마 이 값은 안 바뀔 것
    # log_probs: 1 x k 이고, 1st output 의 top k 토큰들의 log_prob 들임. 아마도 계속 바뀌어 가겠지?
    outputs, e_outputs, log_probs = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2) # init_vars 내부에서 만들어 썼던 것과 같음
    ind = None
    for i in range(2, opt.max_len): # init_vars 에서 0, 1 까지는 만들어서 왔기 때문에 2부터 시작

        trg_mask = nopeak_mask(i, opt) # init_vars 에서는 i 가 1이었고, 지금은 2임. lower triangle 로 만들어짐

        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask))
        # 여기서 입력은 i개 까지만 넣음. 불필요한 계산은 아예 할 필요가 없으므로
        # out 은 k x i x d_model 이겠지?

        out = F.softmax(out, dim=-1) # dimension 은 그대로 두고서 softmaxing 만 함

        # out 에는 i번째의 softmax'ed 된 토큰 확률들이 들어있음
        # 그거를 재료로 해서 top k 개의 인덱스를 찾아서, output 의 다음 컬럼에 넣어서 가져옴
        # log_probs 도 최신 top k 에 대한 것으로 업데이트 될 것
        outputs, log_probs = k_best_outputs(outputs, out, log_probs, i, opt.k)

        eoses = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long) # TODO: 이건 뭐지?
        for vec in eoses:
            i = vec[0] # k 개 중에서 몇번째인지
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k: # k 개 문장이 다 끝날때까지 일단 populate 를 함
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_probs)**alpha) # log_probs 와 타입을 맞춤
            _, ind = torch.max(log_probs * div, 1) # 몇번째 문장이 베스트인지 찾아서 idx 에 넣음. 확률 그 자체(_) 는 안 씀
            ind = ind.data[0] # 스칼라값으로 빼줌
            break

    if ind is None:
        try:
            length = (outputs[0]==eos_tok).nonzero()[0] # 첫번째 eos 의 index 가 length 가 됨. 그래서 eos 가 아예 없으면 에러가 남
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        except:
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:]])
    else:
        try:
            length = (outputs[ind]==eos_tok).nonzero()[0]
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
        except:
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:]])
