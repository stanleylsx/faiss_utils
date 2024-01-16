# -*- coding: utf-8 -*-
# @Time : 2020/12/6 17:16 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : faiss_utils.py 
# @Software: PyCharm
from loguru import logger
import faiss
import numpy as np
import time


class FaissUtils:
    def __init__(self, logger, dim, m=4, nlist=256, nprobe=10, use_gpu=False):
        # m：PQ的子空间数量，用于IndexIVFPQ索引中
        # nlist：nlist是聚类的数量，用于IndexIVFFlat和IndexIVFPQ等索引中
        # nprobe：决定在执行查询操作时要检查多少个聚类或倒排列表，越大越慢但是效果越好。
        self.logger = logger
        self.dim = dim
        self.m = m
        self.nlist = nlist
        self.nprobe = nprobe
        self.nbits = 8
        self.index = None
        self.data_path = './data/data.joblib'
        self.index_path = './data/faiss_index.pkl'
        self.use_gpu = use_gpu
        if use_gpu:
            self.res = faiss.StandardGpuResources()

    def create_index(self, vectors, texts, mids, index_type='ivf_flat', metric='ip'):
        """
        Create vector index.
        """
        if metric == 'l2':
            metric = faiss.METRIC_L2
        else:
            metric = faiss.METRIC_INNER_PRODUCT

        self.logger.info('creating index...')
        start_time = time.time()
        vectors = np.array(vectors).astype('float32')
        if index_type == 'l2':
            self.index = faiss.IndexFlatL2(self.dim)
        elif index_type == 'ip':
            self.index = faiss.IndexFlatIP(self.dim)
        elif index_type == 'cosine':
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(self.dim)
        elif index_type == 'ivf_flat':
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, metric)
        elif index_type == 'ivf_pq':
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits, metric)
        elif index_type == 'hnsw':
            # 构建HNSW的时候，每个向量和多少个最近邻相连接。
            self.index = faiss.IndexHNSWFlat(self.dim, 64, metric)
            # efConstruction：在构建图的时候，每层查找多少个点。
            self.index.hnsw.efConstruction = 64
            # efSearch：在搜索的时候，每层查询多少个点。
            self.index.hnsw.efSearch = 32

        self.index.nprobe = self.nprobe

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

        self.logger.info(f'if index is trained:{self.index.is_trained}')
        if not self.index.is_trained:
            self.logger.info('train index...')
            self.index.train(vectors)
        self.index.add(vectors)
       self.logger.info('Time consumption of create index: %.3f(ms)' % ((time.time() - start_time) * 1000))

    def save_index(self):
        """
        Save index.
        """
        faiss.write_index(self.index, self.index_path)
        joblib.dump((self.corresponding_texts, self.corresponding_mids), self.data_path)

    def load_index(self):
        """
        Load index.
        """
        self.index = faiss.read_index(self.index_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        self.corresponding_texts, self.corresponding_mids = joblib.load(self.data_path)

    def add_vectors(self, vectors):
        """
        Add new vectors to the existing index.
        """
        vectors = np.array(vectors).astype('float32')
        # only add vectors to a trained index
        if self.index.is_trained:
            self.index.add(vectors)
            self.logger.info('add new vectors successful...')
        else:
            self.logger.error('index is not trained, cannot add vectors...')

    def search(self, query_vector, k):
        """
        Execute vector search.
        """
        vector = np.array([query_vector]).astype('float32')
        start_time = time.time()
        faiss.normalize_L2(vector)
        distance, index = self.index.search(vector, k)
        self.logger.info('Time consumption of search: %.3f(ms)' % ((time.time() - start_time) * 1000))
        distance, index = list(distance[0]), list(index[0])
        return index, distance


if __name__ == '__main__':
    faiss_utils = FaissUtils(logger, 768)

