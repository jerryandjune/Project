# -*- coding: utf-8 -*-

from urllib import parse
import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'My_SECRET_KEY'

    JOBS = [
        {  # 第二个任务，每隔5S执行一次
            'id': 'job1',
            'func': 'app.models.scheduler:MeiTuanSchd',  # 方法名
            # 'trigger': 'interval',  # interval表示循环任务
            # 'seconds': 300,
            'trigger': 'cron',
            'hour': 22,
            'minute': 12,
            'second': 0,
        },
        {  # 第二个任务，每隔5S执行一次
            'id': 'job2',
            'func': 'app.models.scheduler:DianPingSchd',  # 方法名
            # 'trigger': 'interval',  # interval表示循环任务
            # 'seconds': 300,
            'trigger': 'cron',
            'hour': 22,
            'minute': 12,
            'second': 0,
        },
        # {  # 第三个任务，每隔5S执行一次
        # 'id': 'job3',
        # 'func': 'app.models.scheduler:RatingProcessSchd',  # 方法名
        # 'trigger': 'interval',  # interval表示循环任务
        # 'seconds': 300,
        # 'trigger': 'cron',
        # 'hour': 20,
        # 'minute': 57,
        # 'second': 0,
        # },
    ]
    # 是否启用mongodb
    MongoDbEnable = True
    MongoDbAuth = True
    MongoDbUsername = parse.quote_plus('xxx')
    MongoDbPassword = parse.quote_plus('xxx')
    MongoDbHost = 'xxx'
    MongoDbPort = 'xxx'
    MongoDbName = 'xxx'

    # 新闻摘要长度
    SummaryLength = 5

    if os.name == 'nt':
        # model01-文本摘要
        # 词向量模型
        NewsFile = 'model01\\sqlResult_1558435.csv'
        # WordsModelFile = 'model01\\zhwiki_news.FastText.model'
        #WordsModelFile = 'model01\\zhwiki_news.word2vec.model'
        WordsModelFile = 'model01\\zhwiki_news.word2vec_min_count5.model'
        # 词向量模型处理方法
        ModelMethod = 'Word2Vec'
        # ModelMethod = 'FastText'

        # model02-情感分析二分类
        # 初始参数设置
        dict_path = 'model02\\chinese_L-12_H-768_A-12\\vocab.txt'
        model_path = 'model02\\bertkeras_model.h5'

        # model02-情感分析20分类
        # 初始参数设置
        train = 'model02\\train.csv'
        best_weight = 'model02\\best_weight'
        stopwords_file = 'model02\\data\\stopwords.txt'
        vocab_file = 'model02\\data\\vocab.txt'
        label_file = 'model02\\data\\label_names.txt'
        test_comment_file = 'model02\\data\\test_comments.csv'

        #文件上传
        UPLOAD_FOLDER = 'static\\uploads'
        pdf2htmlEX = 'static\\pdf2htmlEX\\pdf2htmlEX.exe'

        # model03-pdf关键词高亮
        index_build200 = 'model03\\bk_index_build200.index'
        reverse_word_index = 'model03\\reverse_word_index.pkl'   
        bigram_char = 'model03\\sgns.baidubaike.bigram-char'
        word_index = 'model03\\word_index.pkl'

    else:
        # model01-文本摘要
        # 词向量模型
        NewsFile = 'model01/sqlResult_1558435.csv'
        WordsModelFile = 'model01/zhwiki_news.FastText.model'
        #WordsModelFile = 'model01/zhwiki_news.word2vec.model'
        # WordsModelFile = 'model01/zhwiki_news.word2vec_min_count5.model'
        # 词向量模型处理方法
        #ModelMethod = 'Word2Vec'
        ModelMethod = 'FastText'

        # model02-情感分析二分类
        # 初始参数设置
        dict_path = 'model02/chinese_L-12_H-768_A-12/vocab.txt'
        model_path = 'model02/bertkeras_model.h5'

        # model02-情感分析20分类
        # 初始参数设置
        train = 'model02/train.csv'
        best_weight = 'model02/best_weight'
        stopwords_file = 'model02/data/stopwords.txt'
        vocab_file = 'model02/data/vocab.txt'
        label_file = 'model02/data/label_names.txt'
        test_comment_file = 'model02/data/test_comments.csv'

        #文件上传
        UPLOAD_FOLDER = 'static/uploads'
        pdf2htmlEX = 'static/pdf2htmlEX/pdf2htmlEX.exe'

        # model03-pdf关键词高亮
        index_build200 = 'model03/bk_index_build200.index'
        reverse_word_index = 'model03/reverse_word_index.pkl'   
        bigram_char = 'model03/sgns.baidubaike.bigram-char'
        word_index = 'model03/word_index.pkl'

    # 使用GPU计算
    GPUEnable = False

    # 启用爬虫
    WebCrawlerEnable = False
    # 启用数据处理
    ProcessDataEnable = False


    #PDF文件关键信息自动高亮
    ALLOWED_EXTENSIONS = set(['PDF','pdf'])
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    HtmlPath = './Flask/app/static/uploads/'
    Uploads = '/static/uploads/'