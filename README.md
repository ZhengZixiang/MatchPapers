# MatchPapers
Worth-reading papers and related awesome resources on matching task. Matching task is common in many tasks, like natural language inference (NLI), question answering (QA), recommendation system (RecSys), information retrieval (IR) and advertising. This repository also contains many relative research field of this task, including approximately approximate nearest neighbor (ANN), text matching algorithm, CTR, LTR (learning-to-rank) and so on.

Suggestions about adding papers, repositories and other resources are welcomed! 

*Since I am Chinese, I mainly focus on Chinese resources. Welcome to recommend excellent resources in English or other languages!*

值得一读的匹配任务相关论文与资源集合。匹配任务常见于自然语言推断、问答、推荐系统、信息检索、广告等场景。本仓库还包含该任务的许多相关研究领域，包括最近邻搜索、文本匹配算法和CTR、LTR等。

欢迎新增论文、代码仓库与其他资源等建议！

## Papers
### Text Matching
- **Enhanced-RCNN: An Efficient Method for Learning Sentence Similarity**. *Shuang Peng, Hengbin Cui, Niantao Xie, Sujian Li, Jiaxing Zhang, Xiaolong Li*. (WWW 2020) [[paper]](https://dl.acm.org/doi/10.1145/3366423.3379998)
- **Match^2: A Matching over Matching Model for Similar Question Identification**. *Zizhen Wang, Yixing Fan, Jiafeng Guo, Liu Yang, Ruqing Zhang, Yanyan Lan, Xueqi Cheng, Hui Jiang, Xiaozhao Wang*. (SIGIR 2020) [[paper]](https://arxiv.org/abs/2006.11719)

### Recommadation System Matching
- **CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks**. *Dong-Kyu Chae, Jinsoo  Kang, Sangwook  Kim, Jungtae  Lee*. (CIKM 2018) [[paper]](https://dl.acm.org/doi/10.1145/3269206.3271743)[[code]](https://github.com/ARASC/CFGAN)
- **Multi-Interest Network with Dynamic Routing for Recommendation at Tmall**. *Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, Dik Lun Lee*. (CIKM 2019) [[paper]](https://arxiv.org/abs/1904.08030) - ***MIND***
- **Learning Robust Models for e-Commerce Product Search**. *Thanh V. Nguyen, Nikhil Rao, Karthik Subbian*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.03624) - ***QUARTS***

### CTR
- **Deep Session Interest Network for Click-Through Rate Prediction**. *Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, Keping Yang*. (IJCAI 2019) [[paper]](https://arxiv.org/abs/1905.06482)[[codee]](https://github.com/shenweichen/DSIN) - ***DSIN***
- **Behavior Sequence Transformer for E-commerce Recommendation in Alibaba**. *Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, Wenwu Ou*. (DLP-KDD 2019) [[paper]](https://arxiv.org/abs/1905.06874) - ***BST***
- **Deep Match to Rank Model for Personalized Click-Through Rate Prediction**. *Ze Lyu, Yu Dong, Chengfu Huo, Weijun Ren*. (AAAI 2020) [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/5346)[[code]](https://github.com/lvze92/DMR)[[blog]](https://developer.aliyun.com/article/749482) - ***DMR***
- **GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction**. *Tongwen Huang, Qingyun She, Zhiqiang Wang, Junlin Zhang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.03519)
- **Deep Feedback Network for Recommendation**. *Ruobing Xie, Cheng Ling, Yalong Wang, Rui Wang, Feng Xia, Leyu Lin*. (IJCAI 2020) [[paper]](https://www.ijcai.org/Proceedings/2020/349)[[code]](https://github.com/qqxiaochongqq/DFN) - ***DFN***

### LTR
- **IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models**. *Jun Wang, Lantao Yu, Weinan Zhang, Yu Gong, Yinghui Xu, Benyou Wang, Peng Zhang, Dell Zhang*. (SIGIR 2017) [[paper]](https://arxiv.org/abs/1705.10513)[[code]](https://github.com/geek-ai/irgan)

### Embedding & ANN
- **Detecting Near-Duplicates for Web Crawling**. *Gurmeet Singh Manku, Arvind  Jain profile, Anish Das Sarma*.
 (WWW 2007) [[paper]](http://www.wwwconference.org/www2007/papers/paper215.pdf) - ***Simhash***
- **Product Quantization for Nearest Neighbor Search**. *Hervé Jégou, Matthijs Douze, Cordelia Schmid*. (IEEE Transactions on Pattern Analysis and Machine Intelligence 2011) [[paper]](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf) - ***PQ***
- **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs**. *Yu. A. Malkov, D. A. Yashunin*. (IEEE Trans. Pattern Anal. Mach. Intell. 42(4)) [[paper]](https://arxiv.org/abs/1603.09320) - ***HNSW***
- **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms**. *Martin Aumüller, Erik Bernhardsson, Alexander Faithfull*. (Information Systems 2019) [[paper]](https://arxiv.org/abs/1807.05614)[[code]](https://github.com/erikbern/ann-benchmarks)
- **Embedding-based Retrieval in Facebook Search**. *Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, Linjun Yang*. (KDD 2020) [[paper]](https://arxiv.org/abs/2006.11632)

## Survey/Tutorial
- **Deep Learning for Matching in Search and Recommendation**. *Jun Xu, Xiangnan He, Hang Li*. (SIGIR 2018) [[slides]](http://staff.ustc.edu.cn/~hexn/sigir18-deep.pdf)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3209978.3210181)

## Repositories/Resources
- [Baidu / Familia](https://github.com/baidu/Familia) - A Toolkit for Industrial Topic Modeling
- [NTMC-Community / MatchZoo](https://github.com/NTMC-Community/MatchZoo)
- [NTMC-Community / MatchZoo-py](https://github.com/NTMC-Community/MatchZoo-py)
- [lanwuwei / SPM_toolkit](https://github.com/lanwuwei/SPM_toolkit)
- [pengming617 / text_matching](https://github.com/pengming617/text_matching)
- [RediSearch / RediSearch](https://github.com/RediSearch/RediSearch) - Fulltext Search and Secondary Index module for Redis
- [THUNLP / NeuIRPapers](https://github.com/thunlp/NeuIRPapers) - Must-read Papers on Neural Information Retrieval
- [THUNLP / OpenMatch](https://github.com/thunlp/OpenMatch)
- [THUwangcy / ReChorus](https://github.com/THUwangcy/ReChorus) [[video]](https://www.bilibili.com/video/BV1Qt4y1Q7jW) - "Chorus" of recommendation models: a PyTorch framework for Top-K recommendation with implicit feedback
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

## Dataset
### Natural Language Inference
- **Adversarial NLI: A New Benchmark for Natural Language Understanding**. *Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, Douwe Kiela*. (ACL 2020) [[paper]](https://arxiv.org/abs/1910.14599)[[data]](https://github.com/facebookresearch/anli)[[blog]](https://ai.facebook.com/blog/introducing-a-new-large-scale-dynamic-data-set-to-push-the-limits-of-natural-language-processing)

### Recommendation System
- **MIND: A Large-scale Dataset for News Recommendation**. *Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, Ming Zhou*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.331/)[[data]](https://msnews.github.io/)

## Articles/Posts
### Chinese
- [阿里 / 神马搜索如何提升搜索的时效性？](https://mp.weixin.qq.com/s/WpITPvYmixMHa0ha0MgWVA)
- [coolhok / faiss-learning学习文档](https://github.com/coolhok/faiss-learning)
- [Chenny / 传统文本匹配算法详解（附代码）](https://zhuanlan.zhihu.com/p/138848302)
- [CNU小学生 / 一文看懂HNSW算法理论的来龙去脉](https://blog.csdn.net/u011233351/article/details/85116719)
- [叉烧 / ACL2020 | 线上搜索结果大幅提升！亚马逊提出对抗式query-doc相关性模型](https://zhuanlan.zhihu.com/p/144911605)
- [Dezhi Ye / Embedding-based Retrieval in Facebook Search论文解读](https://zhuanlan.zhihu.com/p/152570715)
- [邓邓最棒 / 海量文本求topk相似：faiss库初探](https://mp.weixin.qq.com/s/lS4sn1BFf-kvEKi4Ve74pQ)
- [丁香园 / 丁香园在语义匹配任务上的探索与实践](https://zhuanlan.zhihu.com/p/69356170)
- [丁香园 / 搜索中的Query扩展技术](https://zhuanlan.zhihu.com/p/138551957)
- [Giant / K近邻算法哪家强？KDTree、Annoy、HNSW原理和使用方法介绍](https://zhuanlan.zhihu.com/p/152522906)
- [科学空间 / 从EMD、WMD到WRD：文本向量序列的相似度计算](https://kexue.fm/archives/7388)
- [liqima / Faiss wiki in Chinese](https://github.com/liqima/faiss_note)
- [Merria28 / 相似度检测——hnsw参数选择](https://zhuanlan.zhihu.com/p/98028479)
- [Tree / Ranking算法评测指标之 CG、DCG、NDCG](https://zhuanlan.zhihu.com/p/136199536)
- [清雨影 / TOP N 推荐神器 Ranknet加速史（附Pytorch实现）](https://zhuanlan.zhihu.com/p/148262580)
- [腾讯 / 移动腾讯网召回算法实践总结](https://zhuanlan.zhihu.com/p/139256086)
- [腾讯 / 个性化推荐如何满足用户口味？微信看一看的技术这样做](https://mp.weixin.qq.com/s/OGBQvSNce6PGbpTH9yBD3A)
- [腾讯 / 详文解读微信「看一看」多模型内容策略与召回](https://mp.weixin.qq.com/s/EId6TB8SSo3rWXXs9oefkA)
- [Yong Yuan / 图像检索：向量索引](http://yongyuan.name/blog/vector-ann-search.html)
- [一小撮人 / Fiass - Getting started](https://zhuanlan.zhihu.com/p/102278151)
- [一小撮人 / Fiass - Faster search、Lower memory 、Run on GPUs](https://zhuanlan.zhihu.com/p/102281896)
- [一小撮人 / Fiass - clustering, PCA, quantization](https://zhuanlan.zhihu.com/p/102287226)
- [一小撮人 / Faiss - Guidelines to choose an index](https://zhuanlan.zhihu.com/p/102287915)
- [一小撮人 / Faiss - Basic index](https://zhuanlan.zhihu.com/p/102573699)
- [一小撮人 / Faiss - Binary indexes, Composite indexes](https://zhuanlan.zhihu.com/p/105214010)
- [一小撮人 / Fiass - 常见问题总结](https://zhuanlan.zhihu.com/p/107241260)
- [一小撮人 / 一文带你了解Annoy！](https://zhuanlan.zhihu.com/p/109633593)
- [知乎 / Query 理解和语义召回在知乎搜索中的应用](https://mp.weixin.qq.com/s/MAfK4B2F8sPXRLodXkwnmw)
