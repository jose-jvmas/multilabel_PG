# Name:
Multilabel Prototype Generation for Data Reduction in k-Nearest Neighbour classification


# Authors:
Jose J. Valero-Mas [I1] , Antonio Javier Gallego [I1], Pablo Alonso-Jiménez [I2], Xavier Serra [I2]

[I1] University Institute for Computer Research, University of Alicante, Spain
[I2] Music Technology Group, Universitat Pompeu Fabra, Spain


# Description:
Supplementary material for the submission to Pattern Recognition for reproducible research and further development and exploitation of the proposed methods 


# Proposed Multilabel Prototype Generation methods:
- MRHC : Direct implementation of the work in [1]
- MChen : Novel adaptation of the Chen method [2] to the multilabel space introduced in this work
- MRSP1 : Novel adaptation of the RSP (version 1) PG method [3] to the multilabel space introduced in this work
- MRSP2 : Novel adaptation of the RSP (version 2) PG method [3] to the multilabel space introduced in this work
- MRSP3 : Novel adaptation of the RSP (version 3) PG method [3] to the multilabel space introduced in this work


# Multilabel Classifiers implemented (for the experimental evaluation):
- Binary Relevance kNN (BRkNN)
- Label Powerset kNN (LP-kNN)
- Multilabel kNN (MLkNN)


# Corpora considered:
- 12 benchmarking corpora with three levels of label noise


# Usage (reproduction of the results in the manuscript):
1) Install dependecies:
$ pip install -r requirements.txt

2) Run experimentation:
- General experiments:
$ python experiments.py
- Time execution 
$ python time_exection_benchmark.py

# References:
[1] Ougiaroglou, S., Filippakis, P., & Evangelidis, G. (2021, September). Prototype Generation for Multi-label Nearest Neighbours Classification. In International Conference on Hybrid Artificial Intelligence Systems (pp. 172-183). Springer, Cham.
[2] Chen, C. H., & Jóźwik, A. (1996). A sample set condensation algorithm for the class sensitive artificial neural network. Pattern Recognition Letters, 17(8), 819-823.
[3] Sánchez, J. S. (2004). High training set size reduction by space partitioning and prototype abstraction. Pattern Recognition, 37(7), 1561-1564.
