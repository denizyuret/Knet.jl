# dynet-benchmark (last updated Dec 14, 2017)

This directory contains examples implemented for [dynet-benchmark](https://github.com/neulab/dynet-benchmark) repo. See [DyNet technical report](https://arxiv.org/abs/1701.03980) for the architectural details of the implemented examples.

- [rnnlm-batch](rnnlm-batch.jl): A recurrent neural network language model on [PTB](https://catalog.ldc.upenn.edu/ldc99t42) corpus.
- [bilstm-tagger](bilstm-tagger.jl): A bidirectional LSTM network that predicts a tag for each word. It is trained on [WikiNER](https://github.com/neulab/dynet-benchmark/tree/master/data/tags) dataset.
- [bilstm-tagger-withchar](bilstm-tagger-withchar.jl): Similar to bilstm-tagger, but uses characer-based embeddings for unknown words.
- [treenn](treenn.jl): A tree-structured LSTM sentiment classifier trained on [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset.


## Results on Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz, Tesla K80

See the [dynet-benchmark](https://github.com/neulab/dynet-benchmark)
repo for the source code for each model and each framework.

| Model                                               | Metric    |  Knet    | DyNet     | Chainer     |
| ----------------------------------------------------| --------- | -------- | --------- |------------ |
| [rnnlm-batch](rnnlm-batch.jl)                       | words/sec | 28.5k    | 18.7k     | 16k         |
| [bilstm-tagger](bilstm-tagger.jl)                   | words/sec | 6800     | 1200      | 157         |
| [bilstm-tagger-withchar](bilstm-tagger-withchar.jl) | words/sec | 1300     | 900       | 128         |
| [treenn](treenn.jl)                                 | sents/sec | 43       | 68        | 10          |


## Example Usage

Simply call each script with `-h` (or `--help`) option to see all possible script options. For instance,

```
$ julia bilstm-tagger.jl -h
usage: bilstm-tagger.jl [--usegpu] [--embed EMBED] [--hidden HIDDEN]
                        [--mlp MLP] [--timeout TIMEOUT]
                        [--epochs EPOCHS] [--minoccur MINOCCUR]
                        [--report REPORT] [--valid VALID]
                        [--seed SEED] [-h]

Bidirectional LSTM Tagger in Knet

optional arguments:
  --usegpu             use GPU or not
  --embed EMBED        word embedding size (type: Int64, default: 128)
  --hidden HIDDEN      LSTM hidden size (type: Int64, default: 50)
  --mlp MLP            MLP size (type: Int64, default: 32)
  --timeout TIMEOUT    max timeout (in seconds) (type: Int64, default:
                       600)
  --epochs EPOCHS      number of training epochs (type: Int64,
                       default: 100)
  --minoccur MINOCCUR  word min occurence limit (type: Int64, default:
                       6)
  --report REPORT      report period in iters (type: Int64, default:
                       500)
  --valid VALID        valid period in iters (type: Int64, default:
                       10000)
  --seed SEED          random seed (type: Int64, default: -1)
  -h, --help           show this help message and exit
```

Use `--usegpu` option to run examples on GPU,

```
$ julia bilstm-tagger.jl --usegpu
```

To run examples without time limit, pass a non-positive integer to `--timeout` option,

```
$ julia bilstm-tagger.jl --usegpu --timeout 0
```

