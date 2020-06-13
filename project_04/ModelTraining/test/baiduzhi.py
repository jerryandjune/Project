# -*- coding: utf-8 -*-

import requests
from selectolax.parser import HTMLParser
import re
import time, datetime
import random



r_sessoion = requests.session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362",
    "Cookie":"Hm_lvt_6859ce5aaf00fb00387e6434e4fcc925=1583242256; BAIDUID=FDE715991556767D7E2FA3F55EB2A5B6:FG=1",
    "Connection":"Keep-Alive",
    "Host":"zhidao.baidu.com",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":"en-AU,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3",
    "Upgrade-Insecure-Requests":"1",
    "Accept-Encoding":"gzip, deflate, br"
}


r = r_sessoion.get('https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=%CA%C0%BD%E7%C9%CF%D7%EE%B4%F3%B5%C4%B2%DD%D4%AD', headers = headers)
with open(r'c:\temp\baidu.txt','w',encoding='utf-8') as f:
    f.write(r.content.decode('GB2312'))