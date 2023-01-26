# Spatio-Temporal Representation Learning

Representation learning for general graph data and five types of spatio-temporal data:

- [Graph Embedding](#Graph-Embedding)
- [POI/Location](#POI/Location)
- [Road Network](#Road-Network)
- [Region](#Region)
- [Trajectory](#Trajectory)
- [Check-in Sequence](#Check-in-Sequence)


## Graph Embedding

| Model                 | Paper                                                        | Publication                                            | Code                                                         | Remarks                                                      |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| General Graph/Network | [OpenNE](https://github.com/thunlp/OpenNE)                   | [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) | [graph_nets](https://github.com/dsgiitr/graph_nets)               |                                                              |
| DeepWalk              | DeepWalk: Online Learning of Social Representations          | KDD 2014                                               | [Code](https://github.com/phanein/deepwalk)                  | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |
| Node2Vec              | node2vec: Scalable Feature Learning for Networks             | KDD 2016                                               | [Code](https://github.com/aditya-grover/node2vec)            | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |
| LINE                  | Line: Large-scale information network embedding              | WWW 2015                                               | [Code](https://github.com/shenweichen/GraphEmbedding，https://github.com/tangjianpku/LINE) | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |
| ChebConv              | Convolutional neural networks on graphs with fast localized spectral filtering | NIPS 2016                                              | [Code](https://github.com/mdeff/cnn_graph)                   | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |
| GCN                   | Semi-Supervised Classification with Graph Convolutional Networks | ICLR 2017                                              | [Code](https://github.com/tkipf/gcn)                         |                                                              |
| GAT                   | Graph Attention Networks                                     | ICLR 2017                                              | [Code](https://github.com/gordicaleksa/pytorch-GAT)          | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |
| GraphSAGE             | Inductive Representation Learning on Large Graphs            | NIPS 2017                                              | [Code](https://github.com/williamleif/GraphSAGE)             |                                                              |
| metapath2vec          | metapath2vec: Scalable representation learning for heterogeneous networks | KDD 2017                                               | [Code](https://github.com/apple2373/metapath2vec)            |                                                              |
| Geom-GCN              | Geom-GCN: Geometric Graph Convolutional Networks.            | ICLR 2020                                              | [Code](https://github.com/graphdml-uiuc-jlu/geom-gcn)        | [LibCity-Road-Representation](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation) |


## POI/Location
| Model      | Paper                                                        | Publication                                                  | Code                                                         | Remarks |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- |
| ship-gram  | Efficient estimation of word representations in vector space | ICLR 2013                                                    | [Code](https://paperswithcode.com/paper/efficient-estimation-of-word-representations) |         |
| GE         | Learning Graph-based POI Embedding for Location-based Recommendation | CIKM 2016                                                    |                                                              |         |
| POI2Vec    | POI2Vec: Geographical Latent Representation for Predicting Future Visitors | AAAI 2017                                                    |                                                              |         |
| Geo-teaser | Geo-teaser: Geo-temporal sequential embedding rank for point-of-interest recommendation | WWW 2017                                                     |                                                              |         |
| CAPE       | Content-Aware Hierarchical Point-of-Interest Embedding Model for Successive POI Recommendation | IJCAI 2018                                                   |                                                              |         |
| DKFM       | Location Embeddings for Next Trip Recommendation             | WWW 2019                                                     |                                                              |         |
| HIER       | Learning Fine Grained Place Embeddings with Spatial Hierarchy from Human Mobility Trajectories. | arxiv 2020                                                   |                                                              |         |
| TALE       | Pre-training Time-Aware Location Embeddings from Spatial-Temporal Trajectories | TKDE 2021                                                    |                                                              |         |
| CTLE       | Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction | AAAI 2021                                                    | [Code](https://github.com/Logan-Lin/CTLE)                            |         |
| PPR        | Spatio-Temporal Representation Learning with Social Tie for Personalized POI Recommendation | Data Science and Engineering 2022                            |                                                              |         |
| CatEM      | Pre-Trained Semantic Embeddings for POI Categories Based on Multiple Contexts | TKDE 2022                                                    |                                                              |         |
| HE-LMF     | POI Recommendation System using Hypergraph Embedding and Logical Matrix Factorization | Journal of Artificial Intelligence and Capsule Networks 2022 |                                                              |         |

## Road Network

| Model   | Paper                                                        | Publication        | Code                                             | Remarks                       |
| ------- | ------------------------------------------------------------ | ------------------ | ------------------------------------------------ | ----------------------------- |
| IRN2Vec | Learning Embeddings of Intersections on Road Networks        | SIGSPATIAL 2019    | [Code](https://github.com/Leo-Bright/IRN2vec)            | Intersections                 |
| RFN     | Graph Convolutional Networks for Road Networks               | SIGSPATIAL 2019    |                                                  | Intersections                 |
| SRN2Vec | On Representation Learning for Road Networks                 | TIST 2020          |                                                  | Intersections/Road Segment    |
| HRNR    | Learning Effective Road Network Representation with Hierarchical Graph Neural Networks | KDD 2020           | [Code](https://gitee.com/solaris_wn/HRNR)                | Road Segment，supervised      |
| Toast   | Robust Road Network Representation Learning: When Traffic Patterns Meet Traveling Semantics | CIKM 2021          | [Code](https://github.com/panda361/TrajFormer_Baselines) | Road Segment，self-supervised |
|         | A Multiview Representation Learning Framework for Large-Scale Urban Road Networks | MDPI 2022          |                                                  | Road Segment                  |
| JCLRNT  | Jointly Contrastive Representation Learning on Road Network and Trajectory | CIKM 2022          | [Code](https://github.com/mzy94/JCLRNT)                  | Road Segment，self-supervised |
| SARN    | Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning | EBDT 2023（CCF-B） |                                                  | Road Segment，self-supervised |

## Region

| Model      | Paper                                                        | Publication                                             | Code                                                 | Remarks             |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------- | ---------------------------------------------------- | ------------------- |
| HDGE       | Region representation learning via mobility flow             | CIKM 2017                                               |                                                      | word2vec            |
| ZE-Mob     | Representing urban functions through zone embedding with human mobility patterns | IJCAI 2018                                              |                                                      | word2vec            |
|            | Learning urban community structures: A collective embedding perspective with periodic spatial-temporal mobility graphs | TIST 2018                                               |                                                      | Auto-Encoder        |
| CGAL       | Unifying inter-region autocorrelation and intra-region structures for spatial embedding via collective adversarial learning | KDD 2019                                                |                                                      | Auto-Encoder        |
| MP-VN      | Efficient region embedding with multi-view spatial networks: A perspective of locality-constrained spatial autocorrelations | AAAI 2019                                               |                                                      | Auto-Encoder        |
| GEML       | GEML: Learning Geo-Contextual Embeddings for Commuting Flow Prediction | AAAI 2020                                               | [Code](https://github.com/jackmiemie/GMEL)                   |                     |
| MVURE      | Multi-View Joint Graph Representation Learning for Urban Region Embedding | IJCAI 2020                                              | [Code](https://github.com/mingyangzhang/mv-region-embedding) | multi-graph         |
| HUGAT      | Effective Urban Region Representation Learning Using Heterogeneous Urban Graph Attention Network | arxiv 2022                                              |                                                      | heterogeneous graph |
| Region2Vec | Region2Vec: Urban Region Profiling via A Multi-Graph Representation Learning Framework | CIKM 2022                                               |                                                      | multi-graph         |
| MGFN       | Multi-Graph Fusion Networks for Urban Region Embedding       | IJCAI 2022                                              | [Code](https://github.com/wushangbin/MGFN)                   | multi-graph         |
| RELM       | Learning Time and Type Aware Representations for Urban Zones | SSRN 2022                                               |                                                      | time-aware          |
| HGI        | Learning urban region representations with POIs and hierarchical graph infomax | ISPRS Journal of Photogrammetry and Remote Sensing 2023 | [Code](https://github.com/RightBank/HGI)                     | POI-Region          |
|            | Unsupervised Representation Learning of Spatial Data via Multimodal Embedding | CIKM 2019                                               |                                                      | Multimodal          |
| Urban2Vec  | Urban2Vec: Incorporating Street View Imagery and POIs for Multi-Modal Urban Neighborhood Embedding | AAAI 2020                                               |                                                      | Multimodal          |
|            | Learning Neighborhood Representation from Multi-Modal Multi-Graph: Image, Text, Mobility Graph and Beyond | arxiv 2021                                              |                                                      | Multimodal          |

## Trajectory

| Model          | Paper                                                        | Publication                  | Code                                             | Remarks           |
| -------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------ | ----------------- |
| trajectory2vec | trajectory2vec: Trajectory clustering via deep representation learning | IJCNN 2017                   | [Code](https://github.com/yaodi833/trajectory2vec)       | encoder-decoder   |
| t2vec          | Deep Representation Learning for Trajectory Similarity Computation | ICDE 2018                    | [Code](https://github.com/boathit/t2vec)                 | encoder-decoder   |
| Trembr         | Trembr: Exploring Road Networks for Trajectory Representation Learning | TIST 2020                    | [Code](https://github.com/panda361/TrajFormer_Baselines) | self-supervised   |
| Path-InfoMax   | Unsupervised Path Representation Learning with Curriculum Negative Sampling | IJCAI 2021                   | [Code](https://github.com/Sean-Bin-Yang/Path-InfoMax)    | self-supervised   |
| Toast          | Robust Road Network Representation Learning: When Traffic Patterns Meet Traveling Semantics | CIKM 2021                    | [Code](https://github.com/panda361/TrajFormer_Baselines) | self-supervised   |
| JCLRNT         | Jointly Contrastive Representation Learning on Road Network and Trajectory | CIKM 2022                    | [Code](https://github.com/mzy94/JCLRNT)                  | self-supervised   |
| CSTRM          | CSTRM: Contrastive Self-Supervised Trajectory Representation Model for trajectory similarity computation | Computer Communications 2022 |                                                  | self-supervised   |
| WSCCL          | Weakly-supervised Temporal Path Representation Learning with Contrastive Curriculum Learning | ICDE 2022                    | [Code](https://github.com/Sean-Bin-Yang/TPR)             | weakly-supervised |
| START          | Self-supervised Trajectory Representation Learning with Temporal Regularities and Travel Semantics | ICDE 2023                    | [Code](https://github.com/aptx1231/start)                | self-supervised   |
| CSTTE          | Contrastive Pre-training of Spatial-Temporal Trajectory Embeddings | arxiv 2022                   |                                                  | self-supervised   |

## Check-in Sequence

| Model    | Paper                                                        | Publication | Code | Remarks         |
| -------- | ------------------------------------------------------------ | ----------- | ---- | --------------- |
| TRED     | Semi-supervised Trajectory Understanding with POI Attention for End-to-End Trip Recommendation | TSAS 2020   |      | semi-supervised |
| GTS      | A graph-based approach for trajectory similarity computation in spatial networks | KDD 2021    |      |                 |
| SelfTrip | Self-supervised Representation Learning for Trip Recommendation | KBS 2022    |      | self-supervised |
| CTLTR    | Contrastive Trajectory Learning for Tour Recommendation      | TIST 2022   |      | self-supervised |
|          | Contrastive Pre-training with Adversarial Perturbations for Check-in Sequence Representation Learning | AAAI 2023   |      | self-supervised |

