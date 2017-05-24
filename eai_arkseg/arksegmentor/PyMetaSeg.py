'''
Created on Aug 12, 2015

@author: chen.xiao2@huawei.com
'''
from ctypes import *
class PyMetaModel(object):
    modelHandle = long(0)

    def __init__(self, arkSegPath, dictPath, modelPath, userDictPath, ambDictPath, reloadTime):
        cdll.LoadLibrary(arkSegPath)
        self.arkSeglib = CDLL(arkSegPath)
        if self.arkSeglib is None:
            self.modelHandle = long(0)
            return
        if (dictPath is None or len(dictPath) <= 0) and (modelPath is None or len(modelPath) <= 0):
            self.modelHandle = long(0)
            return
        
        self.modelHandle = long(self.arkSeglib.arkseg_create_model(\
            c_char_p(dictPath), c_char_p(modelPath), c_char_p(userDictPath), c_char_p(ambDictPath), c_int(reloadTime)))
        print "model handle:", self.modelHandle
    
    def __del__(self):
        if self.modelHandle != 0:
            self.arkSeglib.arkseg_release_model(c_long(self.modelHandle))
            self.modelHandle = long(0)
            print "model released:", self.modelHandle
        return
    
class PyMetaSegmentor(object):
    segHandle = long(0)
    dictlev = 0
    edlev = 3
    reserve_all = True
    output_raw_eng = False
    hybridThres = 0.48
    logistic_k = 1.0
    bwtag = False
    def __init__(self, model, maxInputLen):
        if model.arkSeglib is None or model.modelHandle == 0:
            self.segHandle = long(0)
            return
        self.segHandle = long(model.arkSeglib.arkseg_create_segger(c_long(model.modelHandle), c_int(maxInputLen)))
        self.arkSegLib = model.arkSeglib
        self.maxAcceptLen = maxInputLen
        print "seg handle : ", self.segHandle
        return
        
    def __del__(self):
        if self.segHandle != 0:
            self.arkSegLib.arkseg_release_segger(c_long(self.segHandle))
            self.segHandle = long(0)
            print "segger released :", self.segHandle
        return
    
    def get_label(self, _il):
        sLabel = "WRD";
        if _il == 1:
            sLabel = "FNA"
        elif _il == 2:
            sLabel = "NAM"
        elif _il == 3:
            sLabel = "LCS"
        elif _il == 4:
            sLabel = "WRD"
        elif _il == 5:
            sLabel = "ENG"
        elif _il == 6:
            sLabel = "PSN"
        elif _il == 7:
            sLabel = "LOC"
        elif _il == 8:
            sLabel = "ORG"
        elif _il == 60:
            sLabel = "AMB"
        elif _il == 61:
            sLabel = "SKP"
        elif _il == 62:
            sLabel = "PUN"
        elif _il == 63:
            sLabel = "DIG"
        elif _il == 64:
            sLabel = "ALS"
        elif _il == 65:
            sLabel = "DAS"
        elif _il == 66:
            sLabel = "CHN"
        elif _il == 67:
            sLabel = "OTH"
        else:
            sLabel = "WRD"
        
        return sLabel;
        
    def segment_complete(self, inputText, _dictlev, _edlev, _reserve_all, _output_raw_eng, _hybridThres, _logistic_k, _bwtag):
        validText = inputText
        if len(inputText) > self.maxAcceptLen:
            validText = inputText[:self.maxAcceptLen]
        validLen = len(validText)
        input_array = (c_ushort * (validLen + 2))()
        for idx in range(0, validLen):
            input_array[idx] = ord(validText[idx])
#             print input_array[idx]
        outputLen = (len(validText) * 3 + 6)
        output_array = (c_int * outputLen)()
        seglen = self.arkSegLib.arkseg_segment(c_long(self.segHandle), input_array, c_int(validLen), output_array, c_int(outputLen), \
                                               c_int(_dictlev), c_int(_edlev), \
                                               c_bool(_reserve_all), c_bool(_output_raw_eng), \
                                               c_float(_hybridThres), c_float(_logistic_k), c_bool(_bwtag))
        stride = 2
        if (_bwtag) :
            stride += 1
        
        wordsLen = seglen / stride;
        words = []
        for pos in range(0, wordsLen):
            if output_array[pos * stride] + output_array[pos * stride + 1] > validLen:
                break
            if _bwtag:
                words.append(validText[output_array[pos * stride] : output_array[pos * stride] \
                                       + output_array[pos * stride + 1]] \
                                       + "|||" + self.get_label(output_array[pos * stride + 2]))
            else:
                words.append(validText[output_array[pos * stride] : output_array[pos * stride] + output_array[pos * stride + 1]])
        return words
    
    def segment(self, inputText):
        return self.segment_complete(inputText, self.dictlev, self.edlev, self.reserve_all, \
                       self.output_raw_eng, self.hybridThres, self.logistic_k, self.bwtag)
        
    def segment_wtag(self, inputText, _bwtag):
        return self.segment_complete(inputText, self.dictlev, self.edlev, self.reserve_all, \
                       self.output_raw_eng, self.hybridThres, self.logistic_k, _bwtag)
        
    def segment_hybrid(self, inputText, _hybridThes):
        return self.segment_complete(inputText, self.dictlev, self.edlev, self.reserve_all, \
                       self.output_raw_eng, _hybridThes, self.logistic_k, self.bwtag)
        
    def segment_hybrid_wtag(self, inputText, _hybridThes, _bwtag):
        return self.segment_complete(inputText, self.dictlev, self.edlev, self.reserve_all, \
                       self.output_raw_eng, _hybridThes, self.logistic_k, _bwtag)     
    
