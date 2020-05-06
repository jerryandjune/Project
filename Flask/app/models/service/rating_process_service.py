# -*- coding: utf-8 -*-
from app.models.ratingprocessed import RatingProcessed
from app.database import DB
from app.sentiment_classification import SentimentClassification
# 处理点评数据


class RatingProcessService(object):

    @staticmethod
    def Init():
        SentimentClassification.Init()
        pass

    @staticmethod
    def Process():
        while(True): 
            hasdata = False 
            # 获取
            for rating in DB.DATABASE['Ratings'].find({'processed': 0}).sort(
                    [("created_date", -1)]).limit(10):
                hasdata = True
                if rating['comment'] != '':
                    sentiment = 0.
                    sentiment = SentimentClassification.get_sentiment(
                        rating['comment'])
                    ratingprocessed = RatingProcessed(int(rating['reviewId']), rating['userId'],
                                                    rating['username'], rating['restId'], rating['resttitle'],
                                                    rating['rating'], rating['comment'], rating['url'],
                                                    rating['timestamp'], rating['source'],
                                                    sentiment['sentiment_label'], sentiment['sentiment_classification'],
                                                    float(sentiment['negative_prob']), float(sentiment['positive_prob']))

                    ratingprocessed.insert()

                # 设置为已处理
                DB.DATABASE['Ratings'].update({"reviewId": rating['reviewId']}, {
                    '$set': {"processed": 1}})
                    
            if hasdata == False:
                break
            pass
        
