# HPO2Vec+

__Requirements__

- python 3.6
- networkx
- numpy
- scikit-learn
- gensim


__Code References__

1. Original node2vec: Aditya Grover and Jure Leskovec. https://github.com/aditya-grover/node2vec

2. Implementation of node2vec: https://github.com/lucashu1/link-prediction

@misc{lucas_hu_2018_1408472,
   author       = {Lucas Hu and
                   Thomas Kipf and
                   Gökçen Eraslan},
   title        = {{lucashu1/link-prediction: v0.1: FB and Twitter 
                    Networks}},
   month        = sep,
   year         = 2018,
   doi          = {10.5281/zenodo.1408472},
   url          = {https://doi.org/10.5281/zenodo.1408472}
}

Details on how to generate graph pickle based on pairwise node input, such as

```
236 186
122 285
24 346
```
and how to generate embeddings based on pickle file can be found in Ref 2.

__Citing__

If you find HPO2Vec+ useful for your research, please cite the following paper (Reference BibTex will be updated later after the final version come out):

https://www.sciencedirect.com/science/article/pii/S1532046419301650?via%3Dihub
