# coding=utf-8
from ctypes import *
import re


# 调用c++库进行相关处理，包括断句，分词，token，其他语言的预处理

class PreProcess:
    def __init__(self):
        self.LinuxLoad()
        self.LibraryType()
        self.load_resource()

    def LinuxLoad(self):
        path = './lib/'
        self.sentsplit = cdll.LoadLibrary(path + 'libtrt_splitsent_warp-gpp-1_00.so')  # 断句

    def LibraryType(self):
        self.sentsplit.SentSplit_Gen.argtypes = []
        self.sentsplit.SentSplit_Gen.restype = c_void_p
        self.sentsplit.SentSplit_Split.argtypes = [c_void_p, c_char_p, c_char_p]
        self.sentsplit.SentSplit_Split.restype = c_void_p
        self.sentsplit.SentSplit_ResultSentNum.argtypes = [c_void_p]
        self.sentsplit.SentSplit_ResultSentNum.restype = c_int
        self.sentsplit.SentSplit_ResultSent.argtypes = [c_void_p, c_int]
        self.sentsplit.SentSplit_ResultSent.restype = c_char_p
        self.sentsplit.SentSplit_DelResult.argtypes = [c_void_p]

        self.ssobj = self.sentsplit.SentSplit_Gen()

    def load_resource(self):
        path = './lib/'
        self.sentsplit.SentSplit_Initialize(self.ssobj, path + "/jar/no_abbr_no_BreakLine_RE.txt",
                                            path + "jar/BrevWords.txt")
        print 'Sent split Init Success'

    def break_into_sentences(self, src):
        rules = r'(\d+\s+\D{1})'

        def fun(m):
            return m.group(1).replace(' ', '')

        res = self.sentsplit.SentSplit_Split(self.ssobj, src, 'zh')
        sentcount = self.sentsplit.SentSplit_ResultSentNum(res)
        sents = list()
        for index in xrange(sentcount):
            # print self.sentsplit.SentSplit_ResultSent(res,index)
            tgt = self.sentsplit.SentSplit_ResultSent(res, index)
            sents.append(re.sub(rules, fun, tgt))
            # sents.append(self.sentsplit.SentSplit_ResultSent(res,index))
        self.sentsplit.SentSplit_DelResult(res)
        return sents


global preprocess
preprocess = PreProcess()
