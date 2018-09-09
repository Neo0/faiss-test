import numpy as np
import pandas as pd
import faiss                   # make faiss available
from faiss import normalize_L2
import datetime


d_list = [32, 64, 128, 256, 512]                           # dimension
nb_list = [100000, 1000000, 10000000, 100000000]                      # database size
k_list = [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
results = []

nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible

for nb in nb_list:
    for d in d_list:
        for k in k_list:
            result = {}
            result["dim"] = d
            result["nb"] = nb
            result["k"] = k
            xb = np.random.random((nb, d)).astype('float32')
            xb[:, 0] += np.arange(nb) / 1000.
            xq = np.random.random((nq, d)).astype('float32')
            xq[:, 0] += np.arange(nq) / 1000.

            normalize_L2(xb)
            normalize_L2(xq)

            nlist = nb / 10000
            quantizer = faiss.IndexFlatIP(d)  # the other index
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = nlist / 4
            index.verbose = True
            assert not index.is_trained
            index.train(xb)
            assert index.is_trained

            index.add(xb)                  # add may be a bit slower as well
            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.search(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["IVF_avg_spent"] = sum(spent) / 100
            result["IVF_max_spent"] = max(spent)

            # GPU single
            res = faiss.StandardGpuResources()
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index_flat.add(xb)

            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.gpu_index_flat(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["GPU_IVF_avg_spent"] = sum(spent) / 100
            result["GPU_IVF_max_spent"] = max(spent)

            # multiple GPUs
            ngpus = faiss.get_num_gpus()
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            gpu_index.add(xb)

            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.gpu_index_flat(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["GPUs_IVF_avg_spent"] = sum(spent) / 100
            result["GPUs_IVF_max_spent"] = max(spent)

            results.append(result)

            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efConstruction = 40
            index.verbose = True
            index.hnsw.efSearch = 256
            index.add(xb)

            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.search(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["HNSW_avg_spent"] = sum(spent) / 100
            result["HNSW_max_spent"] = max(spent)

            # GPU single
            res = faiss.StandardGpuResources()
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index_flat.add(xb)

            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.gpu_index_flat(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["GPU_HNSW_avg_spent"] = sum(spent) / 100
            result["GPU_HNSW_max_spent"] = max(spent)

            # multiple GPUs
            ngpus = faiss.get_num_gpus()
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            gpu_index.add(xb)

            spent = []
            for i in range(100):
                start = datetime.datetime.now()
                D, I = index.gpu_index_flat(xq[:2], k)     # actual search
                end = datetime.datetime.now()
                s = end - start
                spent.append(s.total_seconds())
            result["GPUs_HNSW_avg_spent"] = sum(spent) / 100
            result["GPUs_HNSW_max_spent"] = max(spent)

            results.append(result)

print(results)
df = pd.DataFrame(results)
df.to_csv("result.csv")
            







