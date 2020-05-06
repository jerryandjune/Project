# -*- coding: utf-8 -*-
from app.models.webcrawler.meituan import MeiTuan
from app.models.webcrawler.dianping import DianPing
from app.models.service.rating_process_service import RatingProcessService

# 美团调度器


def MeiTuanSchd():
    MeiTuan.Init()
    MeiTuan.Process()
    print('meituan finish')
    pass

# 点评调度器


def DianPingSchd():
    DianPing.Init()
    DianPing.Process()
    print('dianping finish')
    pass

# 点评数据处理调度器


def RatingProcessSchd():
    RatingProcessService.Init()
    RatingProcessService.Process()
    pass
