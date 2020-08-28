export BERT
using Knet.Ops21
using Knet.Layer21

struct BERT

end

function BERT(;
              vocab_size = 30522,
              hidden_size = 768,
              num_hidden_layers = 12,
              num_attention_layers = 12,
              intermediate_size = 3072,
              hidden_act = gelu,
              hidden_dropout_prob = 0.1,
              attention_probs_dropout_prob = 0.1,
              max_position_embeddings = 512,
              type_vocab_size = 2,
              initializer_range = 0.02,
              layer_norm_eps = 1e-12,
              gradient_checkpointing = false,
              )
    

end

