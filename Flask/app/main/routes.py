# -*- coding: utf-8 -*-

# Copyright 2019 Arie Bregman
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
from flask import render_template, request, jsonify
from app.main import bp  # noqa
from app.models.job import Job
from app.models.forms import NewsForm
from app.models.forms import SentimentAnalysisForm
from app.newsdata import NewsData
from app.sentence2vec import sentence2vec
import json
from app.database import DB
import pymongo
import re
from bson import json_util
from app.sentiment_analysis import SentimentAnalysis

# about页面
@bp.route("/", methods=['Get', 'Post'])
def Index():
    cleanMemory(1)
    newform = GetNews()
    if newform.validate_on_submit():
        pass
    return render_template("Index.html", form=newform)

# index页面
@bp.route("/About", methods=['Get'])
def About():
    return render_template("About.html")

# 获取摘要信息
@bp.route("/GetNewSummary", methods=['Post'])
def GetNewSummary():
    NewsTitle = request.values['NewsTitle']
    NewsContent = request.values['NewsContent']
    NewSummaryLength = int(request.values['NewSummaryLength'])

    news_title = '{}'.format(NewsTitle)
    news_content = '{}'.format(NewsContent)
    # todo
    summary = sentence2vec.get_summarize(
        news_content, news_title, weight=0.7, top_n=NewSummaryLength)

    return jsonify({"result": summary})


@bp.route("/LoadData", methods=['GET'])
def LoadData():
    newform = GetNews()
    # d = newform.GetDict()
    # jsonret = json.dumps(newform.__dict__,ensure_ascii=False)
    return jsonify({"result":
                    {
                        "NewsTitle": newform.NewsTitle,
                        "NewsContent": newform.NewsContent,
                        "NewSummaryLength": newform.NewSummaryLength
                    }
                    })


def GetNews():
    newform = NewsData.GetNewData()
    return newform


@bp.route("/SentimentAnalysis/Index", methods=['Get', 'Post'])
def SentimentAnalysisIndex():
    cleanMemory(2)
    return render_template("SentimentAnalysis.html", form=None)

# 获取评论情感占比
@bp.route("/SentimentAnalysis/GetSentimentCount", methods=['Get'])
def GetSentimentCount():

    TotalSentiment = DB.DATABASE['RatingProcessed'].find().count()

    NegativeSentiment = DB.DATABASE['RatingProcessed'].find(
        {'sentiment_label': 0}).count()
    PositiveSentiment = DB.DATABASE['RatingProcessed'].find(
        {'sentiment_label': 1}).count()

    return jsonify({"result":
                    {
                        "TotalSentiment": TotalSentiment,
                        "NegativeSentiment": NegativeSentiment,
                        "PositiveSentiment": PositiveSentiment
                    }
                    })

# 获取来源占比
@bp.route("/SentimentAnalysis/getSentimentSource", methods=['Get'])
def getSentimentSource():

    MeiTuan = DB.DATABASE['RatingProcessed'].find({'source': '美团'}).count()
    DaZhongDianPing = DB.DATABASE['RatingProcessed'].find(
        {'source': '大众点评'}).count()

    return jsonify({"result":
                    {
                        "MeiTuan": MeiTuan,
                        "DaZhongDianPing": DaZhongDianPing,
                    }
                    })

# 获取最新评论
@bp.route("/SentimentAnalysis/getRecentSentiment", methods=['Get'])
def getRecentSentiment():

    result = DB.DATABASE['RatingProcessed'].find().sort(
        'created_date', pymongo.DESCENDING).limit(5)
    row = list(result)
    total = len(row)
    return json_util.dumps({"total": total,
                            "rows": row,
                            })


# 情感分析搜索
@bp.route("/SentimentAnalysis/Search", methods=['Get', 'Post'])
def SentimentAnalysisSearch():
    cleanMemory(2)
    return render_template("SentimentAnalysisSearch.html", form=None)

# 获取搜索结果
@bp.route("/SentimentAnalysis/GetSentimentAnalysisSearchResult", methods=['Get', 'Post'])
def GetSentimentAnalysisSearchResult():
    KeyWord = request.values['KeyWord']
    Limit = int(request.values['limit'])
    Offset = int(request.values['offset'])

    # 评论中包含关键字的数据
    if KeyWord != '':
        result = DB.DATABASE['RatingProcessed'].find({'comment': re.compile(KeyWord)}).sort(
            'created_date', pymongo.DESCENDING).skip(Offset).limit(Limit)
    else:
        result = DB.DATABASE['RatingProcessed'].find().sort(
            'created_date', pymongo.DESCENDING).skip(Offset).limit(Limit)
    total = result.count()

    return json_util.dumps({"total": total,
                            "rows": list(result),
                            })

# 情感分析-20分类
@bp.route("/SentimentAnalysis/Analysis", methods=['Get', 'Post'])
def SentimentAnalysisAnalysis():

    cleanMemory(2)

    if 'reviewId' in request.values:
        reviewId = int(request.values['reviewId'])
        result = DB.DATABASE['RatingProcessed'].find({'reviewId': reviewId}).sort(
            'created_date', pymongo.DESCENDING).limit(1)[0]
        AnalysisFrom = SentimentAnalysisForm(result['comment'])
    else:
        comment = SentimentAnalysis.get_test_comment()

        AnalysisFrom = SentimentAnalysisForm(comment)
    return render_template("SentimentAnalysisAnalysis.html", form=AnalysisFrom)


@bp.route("/SentimentAnalysis/GetSentimentAnalysis", methods=['Get', 'Post'])
def GetSentimentAnalysis():
    Comment = request.values['Comment']
    data = None
    if Comment != '':
        data = SentimentAnalysis.predict(Comment)

    # 评论分析
    return json_util.dumps({"result":
                    {
                        "data": data,
                    }
                    })


# 按照项目编号清理内存


def cleanMemory(projectnum):
    # 文本摘要
    if projectnum == 1:
        project01init()
        project02clean()
    # 情感分析
    elif projectnum == 2:
        project01clean()
        project02init()
    # 关键信息自动高亮
    elif projectnum == 3:
        project01clean()
        project02clean()
    # project04
    elif projectnum == 4:
        project01clean()
        project02clean()


def project01init():
    sentence2vec.init()
    NewsData.init()


def project01clean():
    sentence2vec.clean()
    NewsData.clean()


def project02init():
    SentimentAnalysis.Init()


def project02clean():
    SentimentAnalysis.clean()
