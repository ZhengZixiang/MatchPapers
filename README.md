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

### Sentence Embedding
- **On the Sentence Embeddings from Pre-trained Language Models**. *Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2011.05864)[[code]](https://github.com/bohanli/BERT-flow) - ***BERT-flow***

### Query Expansion
- **BERT-QE: Contextualized Query Expansion for Document Re-ranking**. *Zhi Zheng, Kai Hui, Ben He, Xianpei Han, Le Sun, Andrew Yates*. (Findings of EMNLP 2020) [[paper]](https://arxiv.org/abs/2009.07258)[[code]](https://github.com/zh-zheng/BERT-QE)

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
- [guyulongcs / Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising)
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
- [RUC AI Box / Awesome-RSPapers](https://github.com/RUCAIBox/Awesome-RSPapers) - Recommender System Papers
- [RUC AI Box / RecBole](https://github.com/RUCAIBox/RecBole) - A unified, comprehensive and efficient recommendation library
- [RUC AI Box / RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) - This is a repository of public data sources for Recommender Systems (RS).
- [shenweichen / algonotes](https://github.com/shenweichen/algonotes) - 浅梦学习笔记 公众号文章汇总
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
- [wubinzzu / NeuRec](https://github.com/wubinzzu/NeuRec) - Next RecSys Library
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
