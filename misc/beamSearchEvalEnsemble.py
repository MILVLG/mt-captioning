import torch
from torch.autograd import  Variable
import torch.nn.functional as F

def beam_search(logprobs, model_list, opt):
    # args are the miscelleous inputs to the core in addition to embedded word and state
    # kwargs only accept opt

    def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state_list):
        # INPUTS:
        # logprobsf: probabilities augmented after diversity
        # beam_size: obvious
        # t        : time instant
        # beam_seq : tensor contanining the beams
        # beam_seq_logprobs: tensor contanining the beam logprobs
        # beam_logprobs_sum: tensor contanining joint logprobs
        # OUPUTS:
        # beam_seq : tensor containing the word indices of the decoded captions
        # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
        # beam_logprobs_sum : joint log-probability of each beam

        ys, ix = torch.sort(logprobsf, 1, True)
        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size
        if t == 0:
            rows = 1
        for c in range(cols):  # for each column (word, essentially)
            for q in range(rows):  # for each beam expansion
                # compute logprob of expanding beam q with word in (sorted) position c
                local_logprob = ys[q, c]
                candidate_logprob = beam_logprobs_sum[q] + local_logprob
                candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
        candidates = sorted(candidates, key=lambda x: -x['p'])

        new_state_list = [[tmp.clone() for tmp in state] for state in state_list]
        # beam_seq_prev, beam_seq_logprobs_prev
        if t >= 1:
            # we''ll need these as reference when we fork beams around
            beam_seq_prev = beam_seq[:t].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
        for vix in range(beam_size):
            v = candidates[vix]
            # fork beam index q into index vix
            if t >= 1:
                beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
            # rearrange recurrent states
            for i in range(len(state_list)):
                for state_ix in range(len(new_state_list[i])):
                    #  copy over state in previous beam q to new beam at vix
                    new_state_list[i][state_ix][:, vix] = state_list[i][state_ix][:, v['q']]  # dimension one is time step
            # append new end terminal at the end of this beam
            beam_seq[t, vix] = v['c']  # c'th word is the continuation
            beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
            beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
        state_list = new_state_list
        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state_list, candidates

    # start beam search
    beam_size = opt.get('beam_size', 10)
    seq_length = opt.get('seq_length')

    beam_seq = torch.LongTensor(seq_length, beam_size).zero_()
    beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).zero_()
    beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
    done_beams = []

    state_list = []
    for model in model_list:
        state = model.mystate
        state_list.append(state)

    for t in range(seq_length):

        """pem a beam merge. that is,
        for every previous beam we now many new possibilities to branch out
        we need to resort our beams to maintain the loop invariant of keeping
        the top beam_size most likely sequences."""
        logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
        # suppress UNK tokens in the decoding
        logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

        beam_seq, \
        beam_seq_logprobs, \
        beam_logprobs_sum, \
        state_list, \
        candidates_divm = beam_step(logprobsf,
                                    beam_size,
                                    t,
                                    beam_seq,
                                    beam_seq_logprobs,
                                    beam_logprobs_sum,
                                    state_list)

        for vix in range(beam_size):
            # if time's up... or if end token is reached then copy beams
            if beam_seq[t, vix] == 0 or t == seq_length - 1:
                final_beam = {
                    'seq': beam_seq[:, vix].clone(),
                    'logps': beam_seq_logprobs[:, vix].clone(),
                    'p': beam_logprobs_sum[vix]
                }
                done_beams.append(final_beam)
                # don't continue beams from finished sequences
                beam_logprobs_sum[vix] = -1000

        # encode as vectors
        it = beam_seq[t]
        tmp_state_list = []

        ouputs = None
        state = None
        for i, model in enumerate(model_list):
            if hasattr(model, 'tmp_memory'):
                ouput, state = model.get_logprobs_state_ensemble(Variable(it.cuda()), model.tmp_memory,
                                                                 model.tmp_att1_masks, state_list[i])
            else:
                ouput, state = model.get_logprobs_state_ensemble(Variable(it.cuda()), model.tmp_avg_att_feats, model.tmp_att_feats, model.tmp_p_att_feats, state_list[i])
            tmp_state_list.append(state)
            if ouputs is None:
                ouputs = Variable(ouput.data.new(ouput.size()).zero_()).cuda()
            ouputs += ouput
        state_list = tmp_state_list
        probs = ouputs.div(len(model_list))
        logprobs = torch.log(probs)


    done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
    return done_beams


def sample_beam(model_list, att_feats, num_bbox, att_masks=None, opt={}):
    beam_size = opt.get('beam_size')
    vocab_size = opt.get('vocab_size')
    seq_length = opt.get('seq_length')

    batch_size = att_feats.size(0)

    assert beam_size <= vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
    seq = torch.LongTensor(seq_length, batch_size).zero_()
    seqLogprobs = torch.FloatTensor(seq_length, batch_size)

    done_beams = [[] for _ in range(batch_size)]
    for k in range(batch_size):
        outputs = None
        for i, model in enumerate(model_list):
            state, t_ouput = model.single_sample_beam_ensemble(k, att_feats, num_bbox, att_masks, opt)
            if outputs is None:
                outputs = Variable(t_ouput.data.new(t_ouput.size()).zero_())
            outputs += t_ouput

        probs = outputs.div(len(model_list))
        logprobs = torch.log(probs)
        done_beams[k] = beam_search(logprobs, model_list, opt=opt)
        seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
        seqLogprobs[:, k] = done_beams[k][0]['logps']

    return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)