# -*- coding: utf-8 -*-
'''
Created on Aug 12, 2015

@author: chen.xiao2@huawei.com
'''
import os
from arksegmentor import PyMetaSeg
if __name__ == '__main__':
    model = PyMetaSeg.PyMetaModel(os.path.abspath('.') + "/libarkseg.so", \
                                  os.path.abspath('.') + "/data/dict.bin", \
                                  os.path.abspath('.') + "/data/model.bin", \
                                  os.path.abspath('.') + "/data/user_dict.txt", \
                                  os.path.abspath('.') + "/data/ambi_dict.txt", \
                                  0)
    segger = PyMetaSeg.PyMetaSegmentor(model, 400)
    # strinput = u'\"人们老是指着我,好像我是动物园的猩猩。\"他说。'
    strinput = '\"人们老是指着我,好像我是动物园的猩猩。\"他说。'
    segwords = segger.segment_hybrid(strinput, 0.55);
    for word in segwords:
        print word,
    print
    print '======================================================'
    segwords = segger.segment(strinput.decode('utf-8'));
    for word in segwords:
        print word,
    print
    print '======================================================'
    segwords = segger.segment_complete(strinput, 0, 1, True, False, 0.75, 1.0, True);
    for word in segwords:
        print word,
    segwords = segger.segment_hybrid_wtag(strinput, 0.49, True);
    for word in segwords:
        print word,
    pass
