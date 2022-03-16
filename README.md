# A Comparative Study of Neural Machine Translation Models for Turkish Language

This repository provides the codes used for _A Comparative Study of Neural Machine Translation Models for Turkish Language_
article published in [Journal of Intelligent and Fuzzy Systems](https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs211453).

# Abstract
_Machine translation (MT) is an important challenge in the fields of Computational Linguistics. In this study, we conducted neural machine translation (NMT) experiments on two different architectures. First, Sequence to Sequence (Seq2Seq) architecture along with a variation that utilizes attention mechanism is performed on translation task. Second, an architecture that is fully based on the self-attention mechanism, namely Transformer, is employed to perform a comprehensive comparison. Besides, the contribution of employing Byte Pair Encoding (BPE) and Gumbel Softmax distributions are examined for both architectures. The experiments are conducted on two different datasets: TED Talks that is one of the popular benchmark datasets for NMT especially among morphologically rich languages like Turkish and WMT18 News dataset that is provided by The Third Conference on Machine Translation (WMT) for shared tasks on various aspects of machine translation. The evaluation of Turkish-to-English translations’ results demonstrate that the Transformer model with combination of BPE and Gumbel Softmax achieved 22.4 BLEU score on TED Talks and 38.7 BLUE score on WMT18 News dataset. The empirical results support that using Gumbel Softmax distribution improves the quality of translations for both architectures._

# Experimental Results
The best results of the conducted experiments are given below

|          | TED Talks | WMT18 News |
|----------|:-----------:|:-----------:|
| Seq2Seq<sup>1</sup> |    20.0    |    27.2   |
| Seq2Seq + Attention<sup>1</sup>   |    20.1   |    25.8   |
| Transformer<sup>2</sup>   |    22.4   |    38.7   |

<sup>1</sup><sup>_Trained with Gumbel Softmax activation_</sub>

<sup>2</sup><sup>_Trained with BPE tokens and Gumbel Softmax activation_</sub>

For the rest of the experiments, please refer the paper.

# Pretrained networks

Pretrained weights of the networks are shared below

| Model | Link |
|----------|:-----------:|
| Seq2Seq |  [download](https://bilgiedu-my.sharepoint.com/:f:/g/personal/ozgur_ozdemir_bilgiedu_net/EiShqF2eNDxHiWBPXFAsE5sB09DIGMCfUTBMppHxqt9hyg?e=4vllJJ)    |
| Seq2Seq + Attention   |    [download](https://bilgiedu-my.sharepoint.com/:f:/g/personal/ozgur_ozdemir_bilgiedu_net/EsaSUpxwu2tPtOC7jwFVX_4BwQJGPQsWNO5jEDTHfZ9BgQ?e=VxQkqK)   |
| Transformer   |    [download](https://bilgiedu-my.sharepoint.com/:f:/g/personal/ozgur_ozdemir_bilgiedu_net/EpW4zpw6ch1OlajtFoKwI9cBeHT9OG3eVfxs3LjUam4b9Q?e=NXvIn5)   |


# Citation
``` @article{ozdemircomparative,
  title={A comparative study of neural machine translation models for Turkish language},
  author={{\"O}zdemir, {\"O}zg{\"u}r and Ak{\i}n, Emre Salih and Velio{\u{g}}lu, R{\i}za and Dalyan, Tu{\u{g}}ba},
  journal={Journal of Intelligent \& Fuzzy Systems},
  number={Preprint},
  pages={1--11},
  publisher={IOS Press}
}
```
