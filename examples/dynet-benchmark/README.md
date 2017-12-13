# dynet-benchmark (last updated Dec 13, 2017)

- [rnnlm-batch](rnnlm-batch.jl): A recurrent neural network language model with mini-batched training.
- [bilstm-tagger](bilstm-tagger.jl): A tagger that runs a bi-directional LSTM and selects a tag for each word.
- [bilstm-tagger-withchar](bilstm-tagger-withchar.jl): Similar to bilstm-tagger, but uses characer-based embeddings for unknown words.
- [trenn](trenn.jl): A text tagger based on tree-structured LSTMs.

This directory contains examples implemented for [dynet-benchmark](https://github.com/neulab/dynet-benchmark) repo. See [DyNet technical report](https://arxiv.org/abs/1701.03980) for the architectural details of the implemented examples.

## Results on K80

| Model                                               | Metric    |  Knet    | DyNet     | Chainer     |
| ----------------------------------------------------| --------- | -------- | --------- |------------ |
| [rnnlm-batch](rnnlm-batch.jl)                       | words/sec | 28500    | 18700     | 16000       |
| [bilstm-tagger](bilstm-tagger.jl)                   | words/sec | 6800     | 1200      | 157         |
| [bilstm-tagger-withchar](bilstm-tagger-withchar.jl) | words/sec | 1300     | 900       | 128         |
| [treenn](trenn.jl)                                  | sents/sec | 43       | 68        | 10          |
