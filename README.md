# ARABIC NER TRAINING:

`onto_to_spacy_json.py` reads in a directory of OntoNotes 5 annotations and
creates training data in spaCy's
[JSON](https://spacy.io/api/annotation#json-input) input for training. The
program currently only gets NER tags, ignoring POS (and dependency, which is
not natively in OntoNotes).

Use it like this: (if you pip install the spacy model)

```
python onto_to_spacy_json.py -i "ontonotes-release-5.0/data/arabic/annotations/nw/ann/00" -t "ar_train.json" -e "ar_eval.json" -v 0.1
```

Use it like this to train arabic ner model
```
 python -m spacy train ar ar_test_output_all ar_train_all.json ar_eval_all.json --no-tagger --no-parser
```
In order to load the model and use it take a look at the file:
`test_spacy_model.ipynb`

Use it like this:(if you customozied build the model for me is v2.0.9 , the difference is you need to give thd dir of the output model)

```
python -m spacy train ar /Users/yanliang/arabicNer/data/ar_output_all /Users/yanliang/arabicNer/data/ar_train_all.json /Users/yanliang/arabicNer/data/ar_eval_all.json --no-tagger --no-parser```
```
