#!/usr/bin/env python
# coding: utf-8

"""
@Time: Created on 2021-02-23 | Updated on 2021-02-23
@Author: 小小磊
@Website:
@Description: 基于 TF-IDF 的文本表示。具体地，
              1、获取关键词表；
              2、计算每个关键词的 TF-IDF 值；
              3、使用 TF-IDF 表示一个文本。；
"""

import jieba
import numpy as np
import math


class TfIdf(object):
    """基于 TF-IDF 的文本表示法"""

    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = self.get_vocab()

    def get_vocab(self):
        """获取关键词表"""
        vocab = list()
        for doc in self.corpus:
            for w in jieba.lcut(doc):
                if w not in vocab and len(w) > 1:
                    vocab.append(w)
        return vocab

    def get_tfidf(self, doc):
        """计算每个关键词的 TF-IDF 值，此处：
        TFtd=Ftd/Sd，关键词 t 在文档 d 中的词频=关键词 t 在文档 d 中出现的次数/
                                                所有关键词在文档 d 中出现的总次数，
        IDFt=ln((1+|D|)/|Dt|)，关键词 t 的逆向文档频率=ln((1+文档总数)/
                                                       包含关键词 t 的文档数量)。
        """
        item_tfidf = dict()
        word_list = [w for w in jieba.lcut(doc) if w in self.vocab]
        word_set = set([w for w in word_list])
        word_len = len(word_list)
        corpus_len = len(self.corpus)
        for word in word_set:
            tf = word_list.count(word) / word_len
            idf = math.log((1 + corpus_len) /
                           len([1 for doc in self.corpus
                                if word in jieba.lcut(doc)]))
            item_tfidf[word] = tf * idf
        return item_tfidf

    def transform(self, item_tfidf):
        """使用 TF-IDF 表示一个文本"""
        arr = list()
        for w in self.vocab:
            arr.append(item_tfidf.get(w, 0.0))
        return np.array(arr).reshape([-1, len(self.vocab)])

    def run(self, mode=1):
        item_rst = dict()
        array_rst = np.empty([0, len(self.vocab)])
        for idx, doc in enumerate(self.corpus):
            item_tfidf = self.get_tfidf(doc)
            item_rst[idx] = item_tfidf
            arr = self.transform(item_tfidf)
            # arr /= np.sqrt(np.sum(np.power(arr, 2)))  # l2正则
            array_rst = np.append(array_rst, arr, axis=0)
        if mode == 0:
            return item_rst
        elif mode == 1:
            return array_rst


if __name__ == '__main__':
    text = ["教育、戏曲、悬疑、悬疑、科幻、军事、教育、戏曲、\
            动作、科幻、科幻、科幻、动作、资讯、资讯", \
            "悬疑、科幻、科幻、动作、资讯、资讯、教育、戏曲、科幻、戏曲"]
    print(TfIdf(text).get_vocab())
    print("\n")
    print(TfIdf(text).run(mode=0))
    print("\n")
    print(TfIdf(text).run(mode=1))
