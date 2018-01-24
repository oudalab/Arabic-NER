# ARABIC NER TRAINING:

`onto_to_spacy_json.py` reads in a directory of OntoNotes 5 annotations and
creates training data in spaCy's
[JSON](https://spacy.io/api/annotation#json-input) input for training. The
program currently only gets NER tags, ignoring POS (and dependency, which is
not natively in OntoNotes).

Use it like this:

```
python onto_to_spacy_json.py -i "ontonotes-release-5.0/data/arabic/annotations/nw/ann/00" -t "ar_train.json" -e "ar_eval.json" -v 0.1
```
