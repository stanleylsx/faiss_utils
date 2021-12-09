# -*- coding: utf-8 -*-
# @Time : 2020/12/6 17:16 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : faiss_utils.py 
# @Software: PyCharm
import faiss
import numpy as np


class FaissUtils:
    def __init__(self, logger):
        self.logger = logger
        self.index_path = 'api/engines/models/faiss_index/vectors.index'
        # index_flat_l2、index_ivf_flat、index_ivf_pq
        self.method = 'index_ivf_flat'
        self.m = 4  # 每个向量分m段
        self.dim = 200  # 向量维度
        self.nlist = 256  # 簇心的个数
        self.nprobe = 10  # 执行搜索的簇心数，增大nprobe可以得到与brute-force更为接近的结果，nprobe就是速度与精度的调节器
        self.index = None

    def index_flat_l2(self, vectors):
        """
        暴力的(brute-force)精确搜索计算L2距离
        :param vectors:
        :return:
        """
        index = faiss.IndexFlatL2(self.dim)
        index.add(vectors)
        return index

    def index_ivf_flat(self, vectors):
        """
        倒排索引的办法
        :param vectors:
        :return:
        """
        quantizer = faiss.IndexFlatL2(self.dim)
        # 定义量化器为点乘(内积)，归一化的向量点乘即cosine相似度（越大越好）
        # quantizer = faiss.IndexFlatIP(self.dim)

        # faiss定义了两种衡量相似度的方法(metrics)
        # faiss.METRIC_L2:欧式距离
        # faiss.METRIC_INNER_PRODUCT:向量内积
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.nprobe
        return index

    def index_ivf_pq(self, vectors):
        """
        IndexIVFPQ索引可以用来压缩向量，具体的压缩算法就是PQ。
        :param vectors:
        :return:
        """
        quantizer = faiss.IndexFlatL2(self.dim)
        # 定义量化器为点乘(内积)，归一化的向量点乘即cosine相似度（越大越好）
        # quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, 8)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.nprobe
        return index

    def train_index(self, vectors):
        """
        输入向量矩阵
        """
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)  # 归一化
        if self.method == 'index_flat_l2':
            index = self.index_flat_l2(vectors)
        elif self.method == 'index_ivf_flat':
            index = self.index_ivf_flat(vectors)
        else:
            index = self.index_ivf_pq(vectors)
        # 保存索引和加载索引
        faiss.write_index(index, self.index_path)
        self.logger.info('save index successful.')

    def load_index(self):
        """
        加载索引
        :return:
        """
        self.index = faiss.read_index(self.index_path)
        self.logger.info('load index successful.')

    def get_query_result(self, query_vector, k):
        """
        执行向量搜索
        :param query_vector:
        :param k:
        :return:
        """
        vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(vector)
        distance, index = self.index.search(vector, k)
        distance, index = list(distance[0]), list(index[0])
        return index, distance

