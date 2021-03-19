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
- **CLEAR: Contrastive Learning for Sentence Representation**. *Zhuofeng Wu, Sinong Wang, Jiatao Gu, Madian Khabsa, Fei Sun, Hao Ma*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2012.15466)
- **Using Prior Knowledge to Guide BERT's Attention in Semantic Textual Matching Tasks**. *Tingyu Xia, Yue Wang, Yuan Tian, Yi Chang*. (WWW 2021) [[paper]](https://arxiv.org/abs/2102.10934)[[code]](https://github.com/xiatingyu/Bert_sim)

### Text Retrieval
- **DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding**. *Yuyu Zhang, Ping Nie, Xiubo Geng, Arun Ramamurthy, Le Song, Daxin Jiang*. (SIGIR 2020) [[paper]](https://arxiv.org/abs/2002.12591)
- **Dense Passage Retrieval for Open-Domain Question Answering**. *Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.04906)[[code]](https://github.com/facebookresearch/DPR) - ***DPR***
- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**. *Omar Khattab, Matei Zaharia*. (SIGIR 2020) [[paper]](https://arxiv.org/abs/2004.12832)[[code]](https://github.com/stanford-futuredata/ColBERT)
- **Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring**. *Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, Jason Weston*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1905.01969)[[unofficial code]](https://github.com/sfzhou5678/PolyEncoder)
- **Pre-training Tasks for Embedding-based Large-scale Retrieval**. *Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, Sanjiv Kumar*. (ICLR 2020) [[paper]](https://arxiv.org/abs/2002.03932)
- **Distilling Knowledge from Reader to Retriever for Question Answering**. *Gautier Izacard, Edouard Grave*. (ICLR 2021) [[paper]](https://arxiv.org/abs/2012.04584)[[code]](https://github.com/lucidrains/distilled-retriever-pytorch)

### Recommadation System Retrieval & Matching
- **CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks**. *Dong-Kyu Chae, Jinsoo  Kang, Sangwook  Kim, Jungtae  Lee*. (CIKM 2018) [[paper]](https://dl.acm.org/doi/10.1145/3269206.3271743)[[code]](https://github.com/ARASC/CFGAN)
- **Multi-Interest Network with Dynamic Routing for Recommendation at Tmall**. *Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, Dik Lun Lee*. (CIKM 2019) [[paper]](https://arxiv.org/abs/1904.08030) - ***MIND***
- **SDM: Sequential Deep Matching Model for Online Large-scale Recommender System**. *Fuyu Lv, Taiwei Jin, Changlong Yu, Fei Sun, Quan Lin, Keping Yang, Wilfred Ng*. (CIKM 2019) [[paper]](https://arxiv.org/abs/1909.00385)[[code]](https://github.com/alicogintel/SDM)
- **Learning Robust Models for e-Commerce Product Search**. *Thanh V. Nguyen, Nikhil Rao, Karthik Subbian*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.03624) - ***QUARTS***
- **Internal and Contextual Attention Network for Cold-start Multi-channel Matching in Recommendation**. *Ruobing Xie, Zhijie Qiu, Jun Rao, Yi Liu, Bo Zhang, Leyu Lin*. (IJCAI 2020) [[paper]](https://www.ijcai.org/Proceedings/2020/379) - ***ICAN***
- **Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations**. *Weihao Gao, Xiangjun Fan, Jiankai Sun, Kai Jia, Wenzhi Xiao, Chong Wang, Xiaobing Liu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.07203)

### CTR
- **Deep & Cross Network for Ad Click Predictions**. *Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang*. (KDD 2017) [[paper]](https://arxiv.org/abs/1708.05123) - ***DCN***
- **DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems**. *Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.13535)
- **Deep Session Interest Network for Click-Through Rate Prediction**. *Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, Keping Yang*. (IJCAI 2019) [[paper]](https://arxiv.org/abs/1905.06482)[[codee]](https://github.com/shenweichen/DSIN) - ***DSIN***
- **Behavior Sequence Transformer for E-commerce Recommendation in Alibaba**. *Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, Wenwu Ou*. (DLP-KDD 2019) [[paper]](https://arxiv.org/abs/1905.06874) - ***BST***
- **Deep Match to Rank Model for Personalized Click-Through Rate Prediction**. *Ze Lyu, Yu Dong, Chengfu Huo, Weijun Ren*. (AAAI 2020) [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/5346)[[code]](https://github.com/lvze92/DMR)[[blog]](https://developer.aliyun.com/article/749482) - ***DMR***
- **Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction**. *Qi Pi, Xiaoqiang Zhu, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Kun Gai*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.05639) - ***SIM***
- **GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction**. *Tongwen Huang, Qingyun She, Zhiqiang Wang, Junlin Zhang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.03519)
- **Deep Feedback Network for Recommendation**. *Ruobing Xie, Cheng Ling, Yalong Wang, Rui Wang, Feng Xia, Leyu Lin*. (IJCAI 2020) [[paper]](https://www.ijcai.org/Proceedings/2020/349)[[code]](https://github.com/qqxiaochongqq/DFN) - ***DFN***
- **Deep Interest with Hierarchical Attention Network for Click-Through Rate Prediction**. *Weinan Xu, Hengxu He, Minshi Tan, Yunming Li, Jun Lang, Dongbai Guo*. (SIGIR 2020) [[paper]](https://arxiv.org/abs/2005.12981) [[code]](https://github.com/stellaxu/DHAN) - ***DHAN***
- **MiNet: Mixed Interest Network for Cross-Domain Click-Through Rate Prediction**. *Wentao Ouyang, Xiuwu Zhang, Lei Zhao, Jinmei Luo, Yu Zhang, Heng Zou, Zhaojie Liu, Yanlong Du*. (CIKM 2020) [[paper]](https://arxiv.org/abs/2008.02974)[[blog]](https://mp.weixin.qq.com/s/jkGb_qkfmEOEs030ZRIJIw)
- **Operation-aware Neural Networks for User Response Prediction**. *Yi Yang, Baile Xu, Furao Shen, Jian Zhao*. (Neural Networks Volume 121, January 2020) [[paper]](https://arxiv.org/abs/1904.12579) - ***ONN NFFM***
- **CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction**. *Guorui Zhou, Weijie Bian, Kailun Wu, Lejian Ren, Qi Pi, Yujing Zhang, Can Xiao, Xiang-Rong Sheng, Na Mou, Xinchen Luo, Chi Zhang, Xianjie Qiao, Shiming Xiang, Kun Gai, Xiaoqiang Zhu, Jian Xu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2011.05625)[[code]](https://github.com/CAN-Paper/Co-Action-Network)[[blog]](https://mp.weixin.qq.com/s/gf_Esw96j5TzVEVDNW45aA)
- **FuxiCTR: An Open Benchmark for Click-Through Rate Prediction**. *Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2009.05794)

### Sequential RecSys
- **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer**. *Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang*. (CIKM 2019) [[paper]](https://arxiv.org/abs/1904.06690)[[code]](https://github.com/FeiSun/BERT4Rec)
- **Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation**. *Chang Liu, Xiaoguang Li, Guohao Cai, Zhenhua Dong, Hong Zhu, Lifeng Shang*. (AAAI 2021) [[paper]](http://arxiv.org/abs/2103.03578)[[Chinese blog]](https://mp.weixin.qq.com/s/FQLMEyVYJiP3OXRLFwCqkQ) - ***NOVA-BERT****

### LTR
- **IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models**. *Jun Wang, Lantao Yu, Weinan Zhang, Yu Gong, Yinghui Xu, Benyou Wang, Peng Zhang, Dell Zhang*. (SIGIR 2017) [[paper]](https://arxiv.org/abs/1705.10513)[[code]](https://github.com/geek-ai/irgan)

### Embedding & ANN
- **Detecting Near-Duplicates for Web Crawling**. *Gurmeet Singh Manku, Arvind  Jain profile, Anish Das Sarma*.
 (WWW 2007) [[paper]](http://www.wwwconference.org/www2007/papers/paper215.pdf) - ***Simhash***
- **Product Quantization for Nearest Neighbor Search**. *Hervé Jégou, Matthijs Douze, Cordelia Schmid*. (IEEE Transactions on Pattern Analysis and Machine Intelligence 2011) [[paper]](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf) - ***PQ***
- **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs**. *Yu. A. Malkov, D. A. Yashunin*. (IEEE Trans. Pattern Anal. Mach. Intell. 42(4)) [[paper]](https://arxiv.org/abs/1603.09320) - ***HNSW***
- **The Design and Implementation of a Real Time Visual Search System on JD E-commerce Platform**. *Jie Li, Haifeng Liu, Chuanghua Gui, Jianyu Chen, Zhenyun Ni, Ning Wang*. (Middleware Industry 2018) [[paper]](https://arxiv.org/abs/1908.07389)[[code]](https://github.com/vearch)
- **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms**. *Martin Aumüller, Erik Bernhardsson, Alexander Faithfull*. (Information Systems 2019) [[paper]](https://arxiv.org/abs/1807.05614)[[code]](https://github.com/erikbern/ann-benchmarks)
- **Embedding-based Retrieval in Facebook Search**. *Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, Linjun Yang*. (KDD 2020) [[paper]](https://arxiv.org/abs/2006.11632)
- **Accelerating Large-Scale Inference with Anisotropic Vector Quantization**. *Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar*. [[paper]](https://arxiv.org/abs/1908.10396)[[code]](https://github.com/google-research/google-research/tree/master/scann) - ***ScaNN***

### Architecture & System
- **Real-time Attention Based Look-alike Model for Recommender System**. *Yudan Liu, Kaikai Ge, Xu Zhang, Leyu Lin*. (KDD 2019) [[paper]](https://arxiv.org/abs/1906.05022) - ***RALM***
- **Applying Deep Learning To Airbnb Search**. *Malay Haldar, Mustafa Abdool, Prashant Ramanathan, Tao Xu, Shulin Yang, Huizhong Duan, Qing Zhang, Nick Barrow-Williams, Bradley C. Turnbull, Brendan M. Collins, Thomas Legrand*. (KDD 2019) [[paper]](https://arxiv.org/abs/1810.09591)
- **MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu's Sponsored Search**. *Miao Fan, Jiacheng Guo, Shuai Zhu, Shuo Miao, Mingming Sun, Ping Li*. (KDD 2019) [[paper]](https://dl.acm.org/doi/10.1145/3292500.3330651)
- **Embedding-based Retrieval in Facebook Search**. *Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, Linjun Yang*. (KDD 2020) [[paper]](https://arxiv.org/abs/2006.11632)
- **Learning to Build User-tag Profile in Recommendation System**. *Su Yan, Xin Chen, Ran Huo, Xu Zhang, Leyu Lin*. (CIKM 2020) [[paper]](https://dl.acm.org/doi/10.1145/3340531.3412719)
- **Managing Diversity in Airbnb Search**. *Mustafa Abdool, Malay Haldar, Prashant Ramanathan, Tyler Sax, Lanbo Zhang, Aamir Mansawala, Shulin Yang, Thomas Legrand*. (KDD 2020) [[paper]](https://arxiv.org/abs/2004.02621)

## Survey/Tutorial
- **Deep Learning for Matching in Search and Recommendation**. *Jun Xu, Xiangnan He, Hang Li*. (SIGIR 2018) [[slides]](http://staff.ustc.edu.cn/~hexn/sigir18-deep.pdf)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3209978.3210181)
- **A Survey on Knowledge Graph-Based Recommender Systems**. *Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong, Qing He*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2003.00911)
- **Graph Learning Approaches to Recommender Systems: A Review**. *Shoujin Wang, Liang Hu, Yan Wang, Xiangnan He, Quan Z. Sheng, Mehmet A. Orgun, Longbing Cao, Nan Wang, Francesco Ricci, Philip S. Yu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.11718)
- **Adversarial Machine Learning in Recommender Systems: State of the art and Challenges**. *Yashar Deldjoo, Tommaso Di Noia, Felice Antonio Merra*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2005.10322)
- **A Comparison of Supervised Learning to Match Methods for Product Search**. *Fatemeh Sarvi, Nikos Voskarides, Lois Mooiman, Sebastian Schelter, Maarten de Rijke*. (SIGIR 2020) [[paper]](https://arxiv.org/abs/2007.10296)[[code]](https://github.com/arezooSarvi/sigir2020-eComWorkshop-LTM-for-product-search)

## Repositories/Resources
- [Baidu / Familia](https://github.com/baidu/Familia) - A Toolkit for Industrial Topic Modeling
- [chihming / competitive-recsys](https://github.com/chihming/competitive-recsys)
- [Coder-Yu / RecQ](https://github.com/Coder-Yu/RecQ)
- [DA-southampton / NLP_ability](https://github.com/DA-southampton/NLP_ability) - 梳理自然语言处理工程师(NLP)需要积累的各方面知识
- [DA-southampton / Tech_Aarticle](https://github.com/DA-southampton/Tech_Aarticle) - 深度学习模型在各大公司实际生产环境的应用讲解文章
- [deepset-ai / haystack](https://github.com/deepset-ai/haystack) - Transformers at scale for question answering & neural search. Using NLP via a modular Retriever-Reader-Pipeline.
- [guoday / ctrNet-tool](https://github.com/guoday/ctrNet-tool) - This's the tool for CTR, including FM, FFM, NFFM and so on
- [guyulongcs / Deep-Reinforcement-Learning-for-Recommender-Systems](https://github.com/guyulongcs/Deep-Reinforcement-Learning-for-Recommender-Systems) - DRL for RecSys papers
- [JepsonWong / Text_Matching](https://github.com/JepsonWong/Text_Matching) - 文本匹配算法
- [jrzaurin / pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep)
- [lanwuwei / SPM_toolkit](https://github.com/lanwuwei/SPM_toolkit)
- [NTMC-Community / MatchZoo](https://github.com/NTMC-Community/MatchZoo)
- [NTMC-Community / MatchZoo-py](https://github.com/NTMC-Community/MatchZoo-py)
- [mengfeizhang820 / Paperlist-for-Recommender-Systems](https://github.com/mengfeizhang820/Paperlist-for-Recommender-Systems)
- [microsoft / recommenders](https://github.com/microsoft/recommenders) - Best Practices on Recommendation Systems
- [mJackie / RecSys](https://github.com/mJackie/RecSys)
- [pengming617 / text_matching](https://github.com/pengming617/text_matching)
- [RediSearch / RediSearch](https://github.com/RediSearch/RediSearch) - Fulltext Search and Secondary Index module for Redis
- [RUCAIBox / RecBole](https://github.com/RUCAIBox/RecBole) - A unified, comprehensive and efficient recommendation library
- [shenweichen / DeepMatch](https://github.com/shenweichen/DeepMatch)
- [shenweichen / DeepCTR](https://github.com/shenweichen/DeepCTR)
- [shenweichen / DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [shenweichen / GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)
- [ShuaichiLi / Chinese-sentence-similarity-task](https://github.com/ShuaichiLi/Chinese-sentence-similarity-task) - 中文问题句子相似度计算比赛及方案汇总
- [tensorflow / recommenders](https://github.com/tensorflow/recommenders) - A library for building recommender system models using TensorFlow
- [THUNLP / NeuIRPapers](https://github.com/thunlp/NeuIRPapers) - Must-read Papers on Neural Information Retrieval
- [THUNLP / OpenMatch](https://github.com/thunlp/OpenMatch)
- [THUwangcy / ReChorus](https://github.com/THUwangcy/ReChorus) [[video]](https://www.bilibili.com/video/BV1Qt4y1Q7jW) - "Chorus" of recommendation models: a PyTorch framework for Top-K recommendation with implicit feedback
- [wangle1218 / deep_text_matching](https://github.com/wangle1218/deep_text_matching) - Implementation several deep text match (text similarly) models for Keras
- [wzhe06 / Reco-papers](https://github.com/wzhe06/Reco-papers)
- [zhaogaofeng611 / TextMatch](https://github.com/zhaogaofeng611/TextMatch) - 基于Pytorch的中文语义相似度匹配模型
- [ZiyaoGeng / Recommender-System-with-TF2.0](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0) - Recurrence the recommender paper with Tensorflow2.0

### ANN
- [aaalgo / KGraph](https://github.com/aaalgo/kgraph) - A Library for Approximate Nearest Neighbor Search
- [erikbern / ann-benchmarks](https://github.com/erikbern/ann-benchmarks) - Benchmarks of approximate nearest neighbor libraries in Python
- [facebookresearch / Faiss](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors
- [FALCONN-LIB / FALCONN](https://github.com/FALCONN-LIB/FALCONN) - LSH-based FAst Lookups of Cosine and Other Nearest Neighbors
- [google-research / ScaNN](https://github.com/google-research/google-research/tree/master/scann) - a method for efficient vector similarity search at scale
- [Jina AI / Jina](https://github.com/jina-ai/jina) - An easier way to build neural search in the cloud
- [kayzhu / LSHash](https://github.com/kayzhu/LSHash) - A fast Python implementation of LSH
- [leonsim / simhash](https://github.com/leonsim/simhash) - A Python Implementation of Simhash Algorithm
- [Microsoft / SPTAG](https://github.com/microsoft/SPTAG) - A distributed approximate nearest neighborhood search (ANN) library
- [milvus-io / Milvus](https://github.com/milvus-io/milvus) - An open source vector similarity search engine
- [pixelogik / NearPy](https://github.com/pixelogik/NearPy) - Python framework for fast ANN search in large, high-dimensional datasets
- [primetang / pyflann](https://github.com/primetang/pyflann) - python bindings for FLANN
- [scatterlab / faiss-serving](https://github.com/scatterlab/faiss-serving) - A lightweight Faiss HTTP Server
- [Spotify / Annoy](https://github.com/spotify/annoy) - Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk
- [Vearch / Vearch](https://github.com/vearch/vearch) - A distributed system for efficient similarity search of embedding vectors
- [Vearch / Gamma](https://github.com/vearch/gamma) - Real time vector search engine
- [wangzhegeek / DSSM-Lookalike](https://github.com/wangzhegeek/DSSM-Lookalike)
- [yanyiwu / simhash](https://github.com/yanyiwu/simhash) - A C++ Implementation of Simhash for Chinese
- [matsui528 / Rii](https://github.com/matsui528/rii) - IVFPQ-based fast and memory efficient ANN search method with a subset-search functionality
- [mukul5sharma / SearchEngine](https://github.com/mukul5sharma/SearchEngine) - A simple search engine using BM25 ranking algorithm

## Dataset
### Natural Language Inference
- **Adversarial NLI: A New Benchmark for Natural Language Understanding**. *Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, Douwe Kiela*. (ACL 2020) [[paper]](https://arxiv.org/abs/1910.14599)[[data]](https://github.com/facebookresearch/anli)[[blog]](https://ai.facebook.com/blog/introducing-a-new-large-scale-dynamic-data-set-to-push-the-limits-of-natural-language-processing)
- **OCNLI: Original Chinese Natural Language Inference**. *Hai Hu, Kyle Richardson, Liang Xu, Lu Li, Sandra Kuebler, Lawrence S. Moss*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.05444)[[data]](https://github.com/CLUEbenchmark/OCNLI)
- **ConjNLI: Natural Language Inference Over Conjunctive Sentences**. *Swarnadeep Saha, Yixin Nie, Mohit Bansal*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.10418)[[data]](https://github.com/swarnaHub/ConjNLI)

### Recommendation System
- **MIND: A Large-scale Dataset for News Recommendation**. *Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, Ming Zhou*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.331/)[[data]](https://msnews.github.io/)

## Articles & Posts
### Chinese
- [Coggle / 深入理解推荐系统：十大序列化推荐算法梳理](https://mp.weixin.qq.com/s/x8G6Hkq_ocdO1efnCOVfHg)
- [EE_NovRain / 各大公司广泛使用的在线学习算法FTRL详解](http://www.cnblogs.com/EE-NovRain/p/3810737.html)
- [Johnson0722 / CTR预估算法之FM, FFM, DeepFM及实践](https://blog.csdn.net/john_xyz/article/details/78933253)
- [Ruhjkg / 2020年精排模型调研](https://zhuanlan.zhihu.com/p/335781101)
- [良口三 / 从Triplet Loss看推荐系统中文章Embedding](https://zhuanlan.zhihu.com/p/138850070)
- [辛俊波 / 线下auc涨，线上ctr/cpm跌的原因和解决办法](https://www.zhihu.com/question/32218407/answer/1172781649)
- [七便士 / SDM(Sequential Deep Matching Model)的复现之路](https://zhuanlan.zhihu.com/p/141411747)
- [深度传送门 / 工业界深度推荐系统与CTR预估必读的论文汇总](https://zhuanlan.zhihu.com/p/83103245)
- [朱亚东 / 排序学习综述](https://mp.weixin.qq.com/s/gP57ROWN27on1wHj3rp-Iw)
- [阿里 / 神马搜索如何提升搜索的时效性？](https://mp.weixin.qq.com/s/WpITPvYmixMHa0ha0MgWVA)
- [阿里 / Life-long兴趣建模视角CTR预估模型：Search-based Interest Model](https://zhuanlan.zhihu.com/p/148416488)
- [阿里 / 阿里妈妈深度树匹配技术演进3.0：TDM->JTM->BSAT](https://mp.weixin.qq.com/s/Nd9vCggZ3RfWLMpZ9JRKdQ)
- [阿里 / 阿里粗排技术体系与最新进展](https://mp.weixin.qq.com/s/CN3a4Zb4yEjgi4mkm2lX6w)
- [coolhok / faiss-learning学习文档](https://github.com/coolhok/faiss-learning)
- [Chenny / 传统文本匹配算法详解（附代码）](https://zhuanlan.zhihu.com/p/138848302)
- [cmathx / 搜索推荐召回&&粗排相关性优化最新进展—2020](https://zhuanlan.zhihu.com/p/195548025)
- [CNU小学生 / 一文看懂HNSW算法理论的来龙去脉](https://blog.csdn.net/u011233351/article/details/85116719)
- [策略算法工程师之路 / Query纠错算法](https://zhuanlan.zhihu.com/p/145198390)
- [叉烧 / ACL2020 | 线上搜索结果大幅提升！亚马逊提出对抗式query-doc相关性模型](https://zhuanlan.zhihu.com/p/144911605)
- [Dezhi Ye / Embedding-based Retrieval in Facebook Search论文解读](https://zhuanlan.zhihu.com/p/152570715)
- [邓邓最棒 / 海量文本求topk相似：faiss库初探](https://mp.weixin.qq.com/s/lS4sn1BFf-kvEKi4Ve74pQ)
- [丁香园 / 丁香园在语义匹配任务上的探索与实践](https://zhuanlan.zhihu.com/p/69356170)
- [丁香园 / 搜索中的Query扩展技术](https://zhuanlan.zhihu.com/p/138551957)
- [丁香园 / 搜索中的Query扩展技术（二）](https://mp.weixin.qq.com/s/2Phu-syd8IKLuwJph6URhA)
- [Giant / K近邻算法哪家强？KDTree、Annoy、HNSW原理和使用方法介绍](https://zhuanlan.zhihu.com/p/152522906)
- [花椒 / 智能推荐算法在直播场景中的应用](https://mp.weixin.qq.com/s/Pf0BFcBIyJnbIsz24X2NMQ)
- [京东 / 深度解析京东个性化推荐系统演进史](https://blog.csdn.net/qq_40027052/article/details/78733365)
- [京东 / 京东搜索在线学习探索实践](https://mp.weixin.qq.com/s/vG9grX_V0ksxIOJwWKMVPw)
- [科学空间 / 从EMD、WMD到WRD：文本向量序列的相似度计算](https://kexue.fm/archives/7388)
- [liqima / Faiss wiki in Chinese](https://github.com/liqima/faiss_note)
- [李rumor / 21个经典深度学习句间关系模型｜代码&技巧](https://mp.weixin.qq.com/s/SH2FaDZH6lw8hy3Ym8esTw)
- [Merria28 / 相似度检测——hnsw参数选择](https://zhuanlan.zhihu.com/p/98028479)
- [每天都要机器学习 / 基于向量的深层语义相似文本召回？你需要bert和faiss](https://mp.weixin.qq.com/s/3RPEabFtgIloIEMMPqPrlw)
- [美团 / 深度学习在美团点评推荐平台排序中的运用](https://mp.weixin.qq.com/s/847h4ITQMtUlZcurJ9Vlvg?scene=25##)
- [美团 / MT-BERT在文本检索任务中的实践](https://tech.meituan.com/2020/08/20/mt-bert-in-document.html)
- [美团 / BERT在美团搜索业务中的应用](https://www.bilibili.com/video/BV1vC4y147px)
- [PaperWeekly / BERT在文档级排序中的应用](https://mp.weixin.qq.com/s/IlQxziByXkIz5aiSb6FneQ)
- [平安寿险PAI / AAAI 2020 | 基于Transformer的对话选择语义匹配模型](https://mp.weixin.qq.com/s/s-P_YVkdafUUT0VA_rBaZw)
- [浅梦的学习笔记 / DCN-M：Google提出改进版DCN，用于大规模排序系统的特征交叉学习(附代码)](https://mp.weixin.qq.com/s/0qidwbxyfTkODTw2DIiRWw)
- [清雨影 / TOP N 推荐神器 Ranknet加速史（附Pytorch实现）](https://zhuanlan.zhihu.com/p/148262580)
- [Tree / Ranking算法评测指标之 CG、DCG、NDCG](https://zhuanlan.zhihu.com/p/136199536)
- [腾讯 / 神盾推荐——MAB算法应用总结](https://mp.weixin.qq.com/s/Xyqwrf2RudGtbMtwoqCTYA)
- [腾讯 / 移动腾讯网召回算法实践总结](https://zhuanlan.zhihu.com/p/139256086)
- [腾讯 / 个性化推荐如何满足用户口味？微信看一看的技术这样做](https://mp.weixin.qq.com/s/OGBQvSNce6PGbpTH9yBD3A)
- [腾讯 / 详文解读微信「看一看」多模型内容策略与召回](https://mp.weixin.qq.com/s/EId6TB8SSo3rWXXs9oefkA)
- [腾讯 / 万字长文读懂微信“看一看”内容理解与推荐](https://mp.weixin.qq.com/s/ZS1NUPJmkWzWcEU2x3jq1A)
- [腾讯 / 搜你所想，从Query意图识别到类目识别的演变](https://mp.weixin.qq.com/s/s8swIdAPw_VeAWnZTL1riA)
- [腾讯 / 全面理解搜索Query：当你在搜索引擎中敲下回车后，发生了什么？](https://zhuanlan.zhihu.com/p/112719984)
- [王鸿伟 / DNN可以进行高阶特征交互，为什么Wide&Deep和DeepFM等模型仍然需要显式构造Wide部分？](https://www.zhihu.com/question/364517083)
- [吴海波 / 乱弹机器学习评估指标AUC](https://zhuanlan.zhihu.com/p/52930683)
- [Yong Yuan / 图像检索：向量索引](http://yongyuan.name/blog/vector-ann-search.html)
- [夕小瑶的卖萌屋 / 2020深度文本匹配最新进展：精度、速度我都要！](https://mp.weixin.qq.com/s/UcNhNgiASKhxBbcXGEz0tw)
- [夕小瑶的卖萌屋 / Facebook刷新开放域问答SOTA：模型训模型！Reader当Teacher！](https://mp.weixin.qq.com/s/90eWqedpiS5o037QByXe0A)
- [一块小蛋糕 / 推荐系统总结之深度召回模型（上）](https://zhuanlan.zhihu.com/p/267263561)
- [一小撮人 / Fiass - Getting started](https://zhuanlan.zhihu.com/p/102278151)
- [一小撮人 / Fiass - Faster search、Lower memory 、Run on GPUs](https://zhuanlan.zhihu.com/p/102281896)
- [一小撮人 / Fiass - clustering, PCA, quantization](https://zhuanlan.zhihu.com/p/102287226)
- [一小撮人 / Faiss - Guidelines to choose an index](https://zhuanlan.zhihu.com/p/102287915)
- [一小撮人 / Faiss - Basic index](https://zhuanlan.zhihu.com/p/102573699)
- [一小撮人 / Faiss - Binary indexes, Composite indexes](https://zhuanlan.zhihu.com/p/105214010)
- [一小撮人 / Fiass - 常见问题总结](https://zhuanlan.zhihu.com/p/107241260)
- [一小撮人 / 一文带你了解Annoy！](https://zhuanlan.zhihu.com/p/109633593)
- [知乎 / Query 理解和语义召回在知乎搜索中的应用](https://mp.weixin.qq.com/s/MAfK4B2F8sPXRLodXkwnmw)
- [知乎 / 知乎搜索文本相关性与知识蒸馏](https://mp.weixin.qq.com/s/xgCtgEMRZ1VgzRZWjYIjTQ)
- [知乎 / 知乎搜索排序模型的演进](https://mp.weixin.qq.com/s/DZZ_BCiNw0EZg7V0KvhXVw)
- [字节跳动 / 3分钟了解今日头条推荐算法原理（附视频+PPT）](http://www.sohu.com/a/217314007_297710)
- [搜索推荐广告排序艺术 谷育龙Eric / CIKM 2020 搜索推荐广告论文集锦](https://mp.weixin.qq.com/s/YYogCRJlIU7smQ3FJpaPPg)
