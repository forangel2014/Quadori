# High-Quality and Diversed Open Rule Induction with Pre-trained Lanuage Models

## Abstract
Rule induction has always been an important task for reasoning. In this paper, we focused on inducing rules expressed in natural languages, i.e. Open Rule Induction. Open rules are more expressive and approach the real world better than traditional rules with pre-defined functions and predicates. However, existing methods inducing open rules suffer from the bias of pre-trained language models, resulting in low quality and diversity. In this paper, we propose Quadori (high-Quality and diverse open rule induction) to induce open rules with higher quality as well as diversity.
In specific, we propose the type-based prompts, samplingbased Supported Beam Search and introduce the Determinantal Point Process to induce rules with higher quality
and diversity. We run experiments on the open rule induction dataset and relation extraction dataset and the results show that Quadori is superior to the existing method in both quality and diversity. Our codes can be accessed at https://anonymous.4open.science/r/Quadori-8131.

## Dependencies

To install requirements:

```
conda env create -f environment.yml
conda activate orion
```

## Download the Two Generator from Orion

We used two BART models pre-trained by Orion to serve as generator for $P(ins|r_p)$ and $P(r_h|ins)$, you could just download them following the steps:

```
mkdir models
cd models
```

1. Download model for $P(ins|r_h)$ (instance generator) from [here](https://drive.google.com/drive/folders/1dgWZS4Cr_QHpGPJ8Rju4Gd_93s340K-v?usp=sharing). You can also find it on [Huggingface Hub](https://huggingface.co/chenxran/orion-instance-generator).

2. Download model for $P(r_p|ins)$ (relation generator) from [here](https://drive.google.com/drive/folders/1syg5b6AmlAT7k2Sx1JpLFXKNX6fOeNoC?usp=sharing). You can also find it on [Huggingface Hub](https://huggingface.co/chenxran/orion-hypothesis-generator).


## Evaluate for Quadori

To evaluate Ouadori's performance, run this command:

```
python evaluation.py --task <task> --inductor rule --mlm_training True --bart_training True --group_beam True --prompt True --ssts True --dpp True
```
