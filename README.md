# GlossLM
A Massively Multilingual Corpus and Pretrained Model for Interlinear Glossed Text

[📄 Paper](https://arxiv.org/abs/2403.06399) | [📦 Models and data](https://huggingface.co/collections/lecslab/glosslm-66da150854209e910113dd87)


## Usage
```python
import transformers

# Your inputs
transcription = "o sey xtok rixoqiil"
translation = "O sea busca esposa."
lang = "Uspanteco"
metalang = "Spanish"
is_segmented = False

prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
Translation in {metalang}: {translation}\n
Glosses: 
"""

model = transformers.T5ForConditionalGeneration.from_pretrained("lecslab/glosslm")
tokenizer = transformers.ByT5Tokenizer.from_pretrained(
    "google/byt5-base", use_fast=False
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = tokenizer.batch_decode(
    model.generate(**inputs, max_length=1024), skip_special_tokens=True
)
print(outputs[0]) # o sea COM-buscar E3S-esposa
```

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

