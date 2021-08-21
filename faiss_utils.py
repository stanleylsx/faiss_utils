# -*- coding: utf-8 -*-
# @Time : 2020/12/6 17:16 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : faiss_utils.py 
# @Software: PyCharm
import time
import faiss
import numpy as np


class FaissUtils:
    def __init__(self, logger):
        self.logger = logger
        self.index_path = 'models/faiss_index/question_index_FlatIP_IVFFlat.index'
        self.m = 10  # 压缩成8bits
        self.dim = 300  # 向量维度
        self.nlist = 2500  # 聚类中心的个数
        self.k = 10  # 查找最相似的k个向量,定义召回向量个数
        self.nprobe = 10  # 查找聚类中心的个数,若nprobe=nlist则等同于精确查找
        self.index = None

    def train_index(self, vectors):
        """
        输入向量矩阵
        """
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)  # 归一化

        # 定义量化器为L2距离，即欧式距离（越小越好）
        # quantizer = faiss.IndexFlatL2(self.dim)
        # 定义量化器为点乘(内积)，归一化的向量点乘即cosine相似度（越大越好）
        quantizer = faiss.IndexFlatIP(self.dim)  # the other index

        """
        faiss定义了两种衡量相似度的方法(metrics)
        faiss.METRIC_L2:欧式距离
        faiss.METRIC_INNER_PRODUCT:向量内积
        """
        # index = faiss.IndexIVFPQ(quantizer, dim, nlist, self.m, 8)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)

        self.logger.info('Training...')
        assert not index.is_trained
        index.train(vectors)
        assert index.is_trained
        index.nprobe = self.nprobe
        index.add(vectors)

        start = time.time()
        # 近似最近邻搜索
        dis, ind = index.search(vectors[3:10], self.k)
        end = time.time()
        self.logger.info('Time usage for search: {} seconds.'.format(round((end - start), 4)))

        # 保存索引和加载索引
        faiss.write_index(index, self.index_path)
        return ind, dis

    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    def get_query_result(self, query_matrix):
        vectors = np.array(query_matrix).astype('float32')
        dis, ind = self.index.search(vectors, self.k)
        return ind, dis
