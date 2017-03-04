# Recurrent Neural Networks

## Motivation

- fixed size api from karpathy
- turing completeness, program analogy
- parameter sharing perspective, goodfellow: compare with 1-D convolution.
- simple examples with irnn: adding, mnist-by-pixel, lm, timit (do we have data?)
- other possible examples: postag, charner.

## Architectures (s2s vs), Examples (lm,mt vs)

Modeling sequences: (hinton)
- input to output sequence (speech, synched, unsynched, when does output start/stop if unsynched (ctc))
- predict next token (lm)
- sequence classification
- s2s models
- Karpathy's graph is more clear
- Hinton's providing input and teaching signals variations
- deeplearningbook 379 (fig 10.3,4,5) has example design patterns
- graves book chap 2 has a classification, 
- Goodfellow 10.5 Seq->Tok, 10.9 Tok->Seq (Tok=Initial and/or Tok=>Input), 10.3,4,10,11 SeqN->SeqN, Sec 10.4 S2S.

Models: (hinton)
- memoryless models, bengios language model
- start with a regular mlp converted to rnn like Goodfellow.

## Modules (lstm gru vs)

- motivation: why do mlp rnns have a hard time learning? vanishing gradients relevant according to (DL 10.7)
- lstm/gru: http://colah.github.io/posts/2015-08-Understanding-LSTMs/ (DL 10.10)
- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
- input and output (word Embedding and prediction) layers

## Backpropagation through time

- Hinton 7b
- Unfolding picture

## Practical concerns (minibatching, gclip etc)

- Hinton 7d: why bptt is difficult, back pass linear.
- http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
- Adam and gclip (DL 10.11)
- minibatching
- decoding and generating: greedy, beam, stochastic.

## Code examples

## Advanced

- multilayer (DL 10.5)
- bidirectional
- attention: http://distill.pub/2016/augmented-rnns/
- speech, handwriting, mt
- image captioning, vqa
- ntm, memory networks: (DL 10.12) http://distill.pub/2016/augmented-rnns/
- 2D rnns: graves chap 8. DL end of 10.3.
- recursive nets? (DL 10.6)
- different length input/output sequences: graves a chapter 7 on ctc, chap 6 on hmm hybrids., olah and carter on adaptive computation time. DL 10.4 on s2s.
- comparison to LDS and HMM (Hinton)
- discussion of teacher forcing and its potential problems (DL 10.2.1)
- echo state networks (DL 10.8) just fix the h->h weights.
- skip connections in time, leaky units (DL 10.9)

## References

- [Karpathy 2015.](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) The Unreasonable Effectiveness of Recurrent Neural Networks.
- [Olah 2015.](http://colah.github.io/posts/2015-08-Understanding-LSTMs) Understanding LSTMs.
- [Hinton 2012.](https://d396qusza40orc.cloudfront.net/neuralnets/lecture_slides/lec7.pdf) RNN lecture slides.
- [Olah and Carter 2016.](http://distill.pub/2016/augmented-rnns) Augmented RNNs.
- [Goodfellow 2016.](http://www.deeplearningbook.org/contents/rnn.html) Deep Learning Chapter 10. Sequence modeling: recurrent and recursive nets.
- [Graves 2012.](https://www.cs.toronto.edu/~graves/preprint.pdf), Supervised Sequence Labelling with Recurrent Neural Networks (textbook)
- [Britz 2015.](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns) Recurrent neural networks tutorial.
- [Manning and Socher 2017.](http://cs224n.stanford.edu/) CS224n: Natural Language Processing with Deep Learning.
- [Wikipedia.](https://en.wikipedia.org/wiki/Recurrent_neural_network) Recurrent neural network.
- [Orr 1999.](https://www.willamette.edu/~gorr/classes/cs449/rnn1.html) RNN lecture notes.
- [Le et al. 2015.](https://arxiv.org/abs/1504.00941) A simple way to initialize recurrent networks of rectified linear units
