import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):

    init_tok = TRG.vocab.stoi['<sos>'] # the index of <sos> is 2
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2) # 1 x 1 x sent_len
    e_output = model.encoder(src, src_mask) # 1 x sent_len x d_model

    outputs = torch.LongTensor([[init_tok]]) # 일단 outputs 은 <sos> 하나만 가지고 시작을 한다는 이야기인 것 같은데?
    if opt.device == torch.device('cuda'):
        outputs = outputs.cuda()

    trg_mask = nopeak_mask(1, opt) # [[[True]]] : 1 x 1 x 1

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask)) # 1 x 1(일단 첫 trg_mask 가 사이즈 1이므로?) x trg_vocab_len
    out = F.softmax(out, dim=-1) # 형태 그대로 남기고 softmaxing 만 함. argmaxing 은 추가로 해줘야 함

    probs, ix = out[:, -1].data.topk(opt.k) # 1 x opt.k 사이즈. [:,-1] 는 차원하나 줄이는 역할. .data 는 없어도 됨. opt.k=3 이므로 top 3 token 의 softmaxed prob 과 indicex(ix) 를 리턴.
    # 지금은 2, 7, 4 가 top 3 인데 각각, <sos>, la, de 임. 아래에서 바로 활용되지만, 이것이 <sos> 바로 다음 토큰의 top 3 임
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0) # unsqueeze 하는 것은 shape [3] 을 원래대로 [1, 3] 으로 원복하기 위함
    # 그리고 log_scores 라기보다는 log_prob 이 더 좋은 명칭

    outputs = torch.zeros(opt.k, opt.max_len).long() # max_len 만큼의 top 3 들을 넣어서 돌려줄 준비
    if opt.device == torch.device('cuda'):
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok # 0번째 토큰의 top 3 는 모두 <sos> 로 고정
    outputs[:, 1] = ix[0] # 1번째 토큰의 top3 는 바로 위에서 만든 top 3 를 넣음. 근데 지금 <sos> 가 1등이기 때문에, 아마 stuck 될 듯

    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1)) # e_outputs 는 여기서 처음 만들어짐
    # opt.k x sent_len x d_model
    # e_output 의 dimension 에서 첫번째 dim 만 opt.k 로 늘린 것
    if opt.device == torch.device('cuda'):
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0] # TODO: 뭐하는 부분이지?

    return outputs, e_outputs, log_scores # e_outputs 의 역할을 잘 모르겠음s

def k_best_outputs(outputs, out, log_scores, i, k):

    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt): # 아휴 여기서 이름을 src 로 바꽈놓으니 가독성이 떨어지지

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2) # init_vars 내부에서 만들어 썼던 것과 같음
    ind = None
    for i in range(2, opt.max_len): # init_vars 에서 0, 1 까지는 만들어서 왔기 때문에 2부터 시작

        trg_mask = nopeak_mask(i, opt) # init_vars 에서는 i 가 1이었고, 지금은 2임. lower triangle 로 만들어짐

        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask)) # init_vars 와 다른 점은 1) trg_mask 가 [1,i,i] 차원이고, 2) outputs 가 [3,i] 이며, 3) e_outputs 를 쓴다는 것
        # 출력 out 은 [3,i,929]

        out = F.softmax(out, dim=-1) # 이것도 dimension 은 그대로 두고서 softmaxing 만 함

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)

        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        # 일단은 eos 가 안나오고 있음
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long) # TODO: 이건 뭐지?
        for vec in ones: # ones 가 텅텅 비어 있으므로 그냥 넘어감
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
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
