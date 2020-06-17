# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # query_numpy = query.cpu().data.numpy()
    # key_numpy = key.cpu().data.numpy()
    # value_numpy = value.cpu().data.numpy()
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # scores_numpy = scores.cpu().data.numpy()
    p_attn = F.softmax(scores, dim=-1)
    # p_attn_numpy = p_attn.cpu().data.numpy()
    # p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn_numpy1 = p_attn.cpu().data.numpy()
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Embeddings(nn.Module):
    def __init__(self, glove_size, vocab, filename, embed_weight_requires_grad):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, glove_size)
        # self.glove_size = glove_size
        # print(vocab, glove_size)
        if filename != None:
            self.init_weight(filename, embed_weight_requires_grad)

    def init_weight(self, filename, requires_grad=True):
        embeding_init_weight = np.load(filename)
        self.lut.weight.data.copy_(torch.from_numpy(embeding_init_weight))
        if not requires_grad:
            self.lut.weight.requires_grad = False

    def forward(self, x):
        return self.lut(x)

# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)],
#                          requires_grad=False)
#         return self.dropout(x)

class LSTMEncoding(nn.Module):
    "Replace Positional Encoding"
    def __init__(self, d_model, embeding_size, dropout):
        super(LSTMEncoding, self).__init__()
        self.d_model = d_model
        self.embeding_size = embeding_size
        self.encoding_lstm = nn.LSTM(self.embeding_size, d_model, 1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(1, batch_size, self.d_model)).cuda()
        c0 = Variable(torch.zeros(1, batch_size, self.d_model)).cuda()
        output, _ = self.encoding_lstm(x, (h0, c0))
        return self.dropout(output)

class AMV(CaptionModel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, glove_size=300):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = PositionalEncoding(d_model, dropout)
        lstm = LSTMEncoding(d_model, glove_size, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(glove_size, tgt_vocab, self.embed_weight_file, self.embed_weight_requires_grad), c(lstm)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1 and p.size() != model.tgt_embed[0].lut.weight.size():
                nn.init.xavier_uniform(p)
        return model

    def __init__(self, opt):
        super(AMV, self).__init__()
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512
        self.embed_weight_file = opt.embed_weight_file
        self.embed_weight_requires_grad = opt.embed_weight_requires_grad

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        # self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        # self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        # self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
        #                         nn.ReLU(),
        #                         nn.Dropout(self.drop_prob_lm))
        # self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 #nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        # self.logit_layers = getattr(opt, 'logit_layers', 1)
        # if self.logit_layers == 1:
        #     self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        # else:
        #     self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
        #     self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        # self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(0, tgt_vocab,
                                     N=opt.num_layers,
                                     d_model=opt.input_encoding_size,
                                     d_ff=opt.rnn_size)

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
    #             weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    # def _prepare_feature(self, fc_feats, att_feats, att_masks):

    #     # embed fc and att feats
    #     fc_feats = self.fc_embed(fc_feats)
    #     att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

    #     # Project the attention feats first to reduce memory and computation comsumptions.
    #     p_att_feats = self.ctx2att(att_feats)

    #     return fc_feats, att_feats, p_att_feats

    def _prepare_feature(self, att_feats, num_bbox, seq=None):
        att_masks = np.zeros(att_feats.size()[:2], dtype='float32')
        num_bbox_numpy = num_bbox.cpu().data.numpy()
        # print(num_bbox)
        for i in range(len(att_masks)):
            # print(num_bbox_numpy[i])
            att_masks[i, :int(num_bbox_numpy[i])] = 1
        # set att_masks to None if attention features have same length
        if att_masks.sum() == att_masks.size:
            att_masks = att_feats.data.new(att_feats.shape[:2]).long().fill_(1)
            att_masks = Variable(att_masks)
        else:
            att_masks = Variable(torch.from_numpy(att_masks), requires_grad=False).cuda()

        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.input_encoding_size,)))

        # if att_masks is None:
        #     # att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        #     att_masks = att_feats.data.new(att_feats.shape[:2]).long().fill_(1)
        att_masks = att_masks.unsqueeze(-2)
        # att_masks = Variable(att_masks)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = Variable(seq_mask)
            seq_mask = seq_mask & Variable(subsequent_mask(seq.size(-1)).type_as(seq_mask.data))
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def forward(self, att_feats, num_bbox, seq, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, num_bbox, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask)

        # batch_size = fc_feats.size(0)
        # state = self.init_hidden(batch_size)

        # # outputs = []
        # outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        # for i in range(seq.size(1) - 1):
        #     if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
        #         sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
        #         sample_mask = sample_prob < self.ss_prob
        #         if sample_mask.sum() == 0:
        #             it = seq[:, i].clone()
        #         else:
        #             sample_ind = sample_mask.nonzero().view(-1)
        #             it = seq[:, i].data.clone()
        #             #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
        #             #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
        #             # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
        #             prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
        #             it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
        #     else:
        #         it = seq[:, i].clone()
        #     # break if all the sequences end
        #     if i >= 1 and seq[:, i].sum() == 0:
        #         break

        #     output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
        #     outputs[:, i] = output
        #     # outputs.append(output)

        outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                                ys,
                                Variable(subsequent_mask(ys.size(1)).type_as(memory.data)))
        logprobs = self.model.generator(out[:, -1])

        return logprobs, [ys.unsqueeze(0)]

    def get_logprobs_state_ensemble(self, it, memory, mask, state):
        # 'it' is Variable contraining a word index
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                                ys,
                                Variable(subsequent_mask(ys.size(1)).type_as(memory.data)))
        logprobs = self.model.generator(out[:, -1])
        probs = torch.exp(logprobs)
        # return probs, state
        return probs, [ys.unsqueeze(0)]

    def _sample_beam(self, att_feats, num_bbox, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, num_bbox)
        memory = self.model.encode(att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = memory[k:k + 1].expand(*((beam_size,) + memory.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = att_feats.data.new(beam_size).long().zero_()
                    xt = Variable(it, requires_grad=False)

                logprobs, state = self.get_logprobs_state(xt, tmp_memory, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, att_feats, num_bbox, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(att_feats, num_bbox, opt)

        batch_size = att_feats.shape[0]

        att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, num_bbox)

        state = None
        memory = self.model.encode(att_feats, att_masks)

        seq = []
        seqLogprobs = []

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = att_feats.data.new(batch_size).long().zero_()
                xt = Variable(it, requires_grad=False)
                # it = att_feats.new_zeros(batch_size, dtype=torch.long)
                # xt = Variable(it, requires_grad=False)

            logprobs, state = self.get_logprobs_state(xt, memory, att_masks, state)
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                xt = Variable(it, requires_grad=False)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
                xt = Variable(it, requires_grad=False)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            # seq[:, t] = it
            # seqLogprobs[:, t] = sampleLogprobs.view(-1)
            seq.append(it)  # seq[t] the input of t+2 time step

            seqLogprobs.append(sampleLogprobs.view(-1))
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        # return seq, seqLogprobs
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


    def single_sample_beam_ensemble(self, batch_ind, att_feats, num_bbox, opt={}):
        beam_size = opt.get('beam_size', 1)

        batch_size = att_feats.size(0)

        att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, num_bbox)
        memory = self.model.encode(att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # lets process every image independently for now, for simplicity

        # state = self.init_hidden(beam_size)
        state = None
        # tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
        tmp_memory = memory[batch_ind:batch_ind + 1].expand(*((beam_size,) + memory.size()[1:])).contiguous()
        tmp_att_masks = att_masks[batch_ind:batch_ind + 1].expand(*((beam_size,) + att_masks.size()[1:])).contiguous()
        self.tmp_memory = tmp_memory
        self.tmp_att_masks = tmp_att_masks

        # input <bos>
        it = att_feats.data.new(beam_size).long().zero_()
        xt = Variable(it, requires_grad=False)

        probs, state = self.get_logprobs_state_ensemble(xt, tmp_memory, tmp_att_masks, state)
        self.mystate = state

        # return state, output
        return state, probs
