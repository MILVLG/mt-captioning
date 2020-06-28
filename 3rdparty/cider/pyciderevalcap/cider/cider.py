# Filename: cider.py
#
#
# Description: Describes the class to compute the CIDEr
# (Consensus-Based Image Description Evaluation) Metric
#          by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and
# Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cider_scorer import CiderScorer


class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, df="corpus"):
        """
        Initialize the CIDEr scoring function
        : param n (int): n-gram size
        : param df (string): specifies where to get the IDF values from
                    takes values 'corpus', 'coco-train'
        : return: None
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        : param  gts (dict) : {image:tokenized reference sentence}
        : param res (dict)  : {image:tokenized candidate sentence}
        : return: cider (float) : computed CIDEr score for the corpus
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()

        for res_id in res:

            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            tmp_cider_scorer += (hypo[0], ref)

        (score, scores) = tmp_cider_scorer.compute_score()

        return score, scores


    def my_compute_score(self, gts, res, avg_refs=True):
        """
        res a list of list
        gts a list of list
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()

        scores = []
        for _gts, _res in zip(gts, res):

            tmp = tmp_cider_scorer.my_get_cider(_gts, _res)
            if avg_refs:
                tmp = np.mean(tmp, 1)
            else:
                tmp = np.mean(tmp, 1)
            scores.append(tmp)

        scores = np.array(scores)
        score = np.mean(scores)

        return score, scores

    def my_self_cider(self, res):
        """
        gts a list of list
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()

        scores = []
        for  _res in res:

            tmp = tmp_cider_scorer.my_get_self_cider(_res)
            scores.append(tmp)

        return scores


    def method(self):
        return "CIDEr"
