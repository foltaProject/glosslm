# GlossLM
A Massively Multilingual Corpus and Pretrained Model for Interlinear Glossed Text

[📄 Paper](https://arxiv.org/abs/2403.06399)

[📦 Models and data](https://huggingface.co/collections/lecslab/glosslm-66da150854209e910113dd87)

## Background
Interlinear Glossed Text (IGT) is a common format in language documentation projects. It looks something like this:

```
o sey x  -tok    r  -ixoqiil
o sea COM-buscar E3S-esposa
“O sea busca esposa.”
```

The three lines are as follows:
- The **transcription** line is a sentence in the target language (possibly segmented into morphemes)
- The **gloss** line gives an linguistic gloss for each morpheme in the transcription
- The **translation** line is a translation into a higher-resource language

