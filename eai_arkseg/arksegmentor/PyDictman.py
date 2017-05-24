# -*- coding: utf-8 -*-
'''
Created on 2015-11-05

@author: chen.xiao2@huawei.com
'''
from ctypes import * 

class PyDictman(object):
    dictHolderHandle = long(0)
    dictPath = ''
    def __init__(self, _libpath):
        if isinstance(_libpath, CDLL):
            self.dictmanLib = _libpath
        else:
            cdll.LoadLibrary(_libpath)
            self.dictmanLib = CDLL(_libpath)
        if self.dictmanLib is None:
            self.dictHolderHandle = long(0)
        return
    def __del__(self):
        if self.dictHolderHandle != long(0):
            self.dictmanLib.dictman_release(self.dictHolderHandle)
            self.dictHolderHandle = long(0)
            print "Dictman release:", self.dictHolderHandle
    def instance(self, _dictPath, maxWordLen, suffix):
        self.dictHolderHandle = long(self.dictmanLib.dictman_instance(c_char_p(_dictPath), c_int(maxWordLen), c_char_p(suffix)))
        if self.dictHolderHandle == long(0):
            return False
        else:
            self.dictPath = _dictPath
            return True
    
    def reload(self):
        self.dictmanLib.dictman_reload(c_long(self.dictHolderHandle), c_char_p(self.dictPath))
        return
    
    def release(self):
        self.dictmanLib.dictman_release(c_long(self.dictHolderHandle))
        self.dictHolderHandle = long(0)
        return
    
    def start_periodic_remap(self, remap_time):
        self.dictmanLib.dictman_start_periodic_remap(c_long(self.dictHolderHandle), c_int(remap_time))    
        return

class PyDictLookup(object):
    dictBufHandle = long(0)
    maxAcceptLen = 0
    def __init__(self, dictmanPath):
        if isinstance(dictmanPath, CDLL):
            self.dictmanLib = dictmanPath
        else:
            cdll.LoadLibrary(dictmanPath)
            self.dictmanLib = CDLL(dictmanPath)
        if self.dictmanLib is None:
            self.dictBufHandle = long(0)
        return
        
    def __del__(self):
        if self.dictBufHandle != long(0):
            self.dictmanLib.dictman_release_buf(c_long(self.dictBufHandle))
            self.dictBufHandle = long(0)
            print "dictman released:", self.dictBufHandle
        return     
       
    def instance(self, _dictman, maxInputLen, suffix):
        self.dictBufHandle = long(self.dictmanLib.dictman_create_buf(c_long(_dictman.dictHolderHandle), c_int(maxInputLen), c_char_p(suffix)))
        if self.dictBufHandle == long(0):
            return False
        else:
            self.maxAcceptLen = maxInputLen
            self.dictman = _dictman
            return True
    
    def release(self):
        self.dictmanLib.dictman_release_buf(c_long(self.dictBufHandle))
        self.dictBufHandle = long(0)
        return

    def sent_lookup(self, sent):
        validText = sent
        if len(sent) > self.maxAcceptLen:
            validText = sent[:self.maxAcceptLen]
        validLen = len(validText)
        input_array = (c_ushort * (validLen + 2))()
        for idx in range(0, validLen):
            input_array[idx] = ord(validText[idx])
        stride = 3
        outputLen = (len(validText) * stride + 2)
        output_array = (c_int * outputLen)()

        retlen = self.dictmanLib.dictman_match_unicode_sent(c_long(self.dictBufHandle), \
                        input_array, c_int(validLen), \
                        output_array, c_int(outputLen))
        self.dictmanLib.dictman_get_value.restype = c_char_p
        segmlen = retlen / stride;
        segments = []
        for idx in range(0, segmlen):
            sg = dict()
            startpos = output_array[idx * stride]
            length = output_array[idx * stride + 1]
            sg['pos'] = startpos
            sg['len'] = length
            val = (self.dictmanLib.dictman_get_value(long(self.dictBufHandle), output_array[idx * stride + 2]))
#             val = self.dictmanLib.dictman_get_value(c_long(self.dictBufHandle), 1000)
            sg['tgt'] = val
            segments.append(sg)
        return segments
    
