# MatchPapers

文本匹配是一项自然语言理解任务，包括自然语言推断、问答匹配、信息检索等多种应用场景

## Papers
- Detecting Near-Duplicates for Web Crawling (WWW 2007) [[paper]](http://www.wwwconference.org/www2007/papers/paper215.pdf) - ***Simhash***
- Product Quantization for Nearest Neighbor Search (IEEE Transactions on Pattern Analysis and Machine Intelligence 2011) [[paper]](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf) - ***PQ***
- ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms (Information Systems 2019) [[paper]](https://arxiv.org/abs/1807.05614)[[code]](https://github.com/erikbern/ann-benchmarks)
- Learning Robust Models for e-Commerce Product Search (ACL 2020) [[paper]](https://arxiv.org/abs/2005.03624) - ***QUARTS***
- Embedding-based Retrieval in Facebook Search (KDD 2020) [[paper]](https://arxiv.org/abs/2006.11632)

## Survey/Tutorial
- Deep Learning for Matching in Search and Recommendation (SIGIR 2018) [[slides]](http://staff.ustc.edu.cn/~hexn/sigir18-deep.pdf)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3209978.3210181)

## Repositories/Resources
- [Baidu / Familia](https://github.com/baidu/Familia) - A Toolkit for Industrial Topic Modeling
- [NTMC-Community / MatchZoo](https://github.com/NTMC-Community/MatchZoo)
- [NTMC-Community / MatchZoo-py](https://github.com/NTMC-Community/MatchZoo-py)
- [lanwuwei / SPM_toolkit](https://github.com/lanwuwei/SPM_toolkit)
- [pengming617 / text_matching](https://github.com/pengming617/text_matching)
- [THUNLP / NeuIRPapers](https://github.com/thunlp/NeuIRPapers) - Must-read Papers on Neural Information Retrieval
- [THUNLP / OpenMatch](https://github.com/thunlp/OpenMatch)
- [shenweichen / DeepMatch](https://github.com/shenweichen/DeepMatch)
- [ShuaichiLi / Chinese-sentence-similarity-task](https://github.com/ShuaichiLi/Chinese-sentence-similarity-task) - 中文问题句子相似度计算比赛及方案汇总
- [zhaogaofeng611 / TextMatch](https://github.com/zhaogaofeng611/TextMatch) - 基于Pytorch的中文语义相似度匹配模型

### ANN
- [aaalgo / KGraph](https://github.com/aaalgo/kgraph) - A Library for Approximate Nearest Neighbor Search
- [erikbern / ann-benchmarks](https://github.com/erikbern/ann-benchmarks) - Benchmarks of approximate nearest neighbor libraries in Python
- [facebookresearch / Faiss](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors
- [FALCONN-LIB / FALCONN](https://github.com/FALCONN-LIB/FALCONN) - LSH-based FAst Lookups of Cosine and Other Nearest Neighbors
- [Jina AI / Jina](https://github.com/jina-ai/jina) - An easier way to build neural search in the cloud
- [kayzhu / LSHash](https://github.com/kayzhu/LSHash) - A fast Python implementation of LSH
- [leonsim / simhash](https://github.com/leonsim/simhash) - A Python Implementation of Simhash Algorithm
- [Microsoft / SPTAG](https://github.com/microsoft/SPTAG) - A distributed approximate nearest neighborhood search (ANN) library
- [milvus-io / Milvus](https://github.com/milvus-io/milvus) - An open source vector similarity search engine
- [pixelogik / NearPy](https://github.com/pixelogik/NearPy) - Python framework for fast ANN search in large, high-dimensional datasets
- [primetang / pyflann](https://github.com/primetang/pyflann) - python bindings for FLANN
- [Spotify / Annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk
- [Vearch / Vearch](https://github.com/vearch/vearch) - A distributed system for efficient similarity search of embedding vectors
- [wangzhegeek / DSSM-Lookalike](https://github.com/wangzhegeek/DSSM-Lookalike)
- [yanyiwu / simhash](https://github.com/yanyiwu/simhash) - A C++ Implementation of Simhash for Chinese
- [matsui528 / Rii](https://github.com/matsui528/rii) - IVFPQ-based fast and memory efficient ANN search method with a subset-search functionality
- [mukul5sharma / SearchEngine](https://github.com/mukul5sharma/SearchEngine) - A simple search engine using BM25 ranking algorithm

## Articles/Posts
### Chinese
- [阿里 / 神马搜索如何提升搜索的时效性？](https://mp.weixin.qq.com/s?__biz=MzU5ODUxNzEyNA==&mid=2247486440&idx=1&sn=1b16a5a1c50ac4b7b5772746cff00239&chksm=fe43bf03c9343615c3e2e357858510081f01dfd53a8efcd9868e98d37411499d82ac17fe78ab&mpshare=1&scene=1&srcid=&sharer_sharetime=1593581511883&sharer_shareid=aaa99b8738c1d6c23f064aa833392e12&key=7332e1fe92797b0dfacc2584cb5fa656a9d4f63c1617ec6a105cdccd3f1221d8fb326c240f67966115c2e3b8449e47e75d6c4f76d96e5728b95fb465947a680cda987fee39bb0c3e0667abedb8f7f4e3&ascene=1&uin=Mjg1NTM0NDcyMw%3D%3D&devicetype=Windows+10+x64&version=6209007b&lang=zh_CN&exportkey=AxwRTPrVyBTDHxYD%2FlxGSJw%3D&pass_ticket=5Y5bD3DNxLYgw6cCdFKC63ta6u65HT0Q%2FaW3BTTRDQXXvvbOiMcDHlnYR607tPf3)
- [coolhok / faiss-learning学习文档](https://github.com/coolhok/faiss-learning)
- [邓邓最棒 / 海量文本求topk相似：faiss库初探](https://mp.weixin.qq.com/s/lS4sn1BFf-kvEKi4Ve74pQ)
- [Dezhi Ye / Embedding-based Retrieval in Facebook Search论文解读](https://zhuanlan.zhihu.com/p/152570715)
- [科学空间 / 从EMD、WMD到WRD：文本向量序列的相似度计算](https://kexue.fm/archives/7388)
- [liqima / Faiss wiki in Chinese](https://github.com/liqima/faiss_note)
- [Yong Yuan / 图像检索：向量索引](http://yongyuan.name/blog/vector-ann-search.html)
- [叉烧 / ACL2020 | 线上搜索结果大幅提升！亚马逊提出对抗式query-doc相关性模型](https://zhuanlan.zhihu.com/p/144911605)
- [丁香园 / 丁香园在语义匹配任务上的探索与实践](https://zhuanlan.zhihu.com/p/69356170)
- [丁香园 / 搜索中的Query扩展技术](https://zhuanlan.zhihu.com/p/138551957)
- [腾讯 / 移动腾讯网召回算法实践总结](https://zhuanlan.zhihu.com/p/139256086)
- [知乎 / Query 理解和语义召回在知乎搜索中的应用](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247496409&idx=1&sn=7b2f5984d71454e1a2812321f6018cf8&scene=21#wechat_redirect)
