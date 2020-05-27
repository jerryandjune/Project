#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

def get_query_pat(query):
    return re.compile('({})'.format('|'.join(query.split())))

def highlight_keywords(pat,document):
    return pat.sub(repl='<span style="background:yellow;">\g<1></span>',string=document)

if __name__ == '__main__':
    text = '美国有线电视新闻网援引美国军方官员的话说'
    pat = r'(新闻|官员)'
    #print(re.compile(pat).sub(repl='**\g<1>**',string=text))
    
    #print(get_query_pat('美军 司令 航母'))

    print(highlight_keywords(get_query_pat('新闻 司令 航母'), text))

    a = {}

    a['1']=1
    a['2']=2
    a['3']=3

    print(' '.join(a.keys()))