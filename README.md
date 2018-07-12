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

## Rehearsing OntoNotes to prevent forgetting

`rehearsal.py` is a script that generates a new Prodigy dataset containing both
NER labeled examples from a given dataset, as well as a number of OntoNotes
examples per annotation. Mixing in old gold standard annotations prevents
catastrophic forgetting.

For example, the following will augment the annotations in the `loc_ner_db`
dataset with OntoNotes annotations:

`python rehearsal.py "loc_ner_db" 5`

The augmented data is written to a dataset called `augmented_for_training`,
which should be treated as temporary because the script overwrites it each
time. NER training can then be performed as usual:

```
prodigy ner.batch-train augmented_for_training en_core_web_sm --eval-split 0.2 
```

## Steps using onto_notes data mixed in the prodigy data and use prodigy to train.
First of all, if you don't have prodigy on your local, you need to install it, and create a db (sqlite by default) for where to import your prodigy data:
### create sqlite db through prodigy
```
python3 -m prodigy dataset arabicner "train arabic ner"
```
### import jsonl data that you exported from the prodigy app:
```
python3 -m prodigy db-in arabicner single_arabic_ner.jsonl 
```
### reheasal your dataset with onto_notes, the dir for onto_notes data is hard coded in rehearsal.py, you need to edit from there (5 here means that 5* onto_notes many records will be mixed in prodigy data)
```
python3 rehearsal.py "arabicner" 5
```
### Last train your data with the following command:
```
python3 -m prodigy ner.batch-train augmented_for_training /home/yan/arabicner/Arabic-NER/testmodel/model8 --eval-split 0.2
```

### Remark:
if you want to explore the sqlitedb for prodigy, you need to go to your home directory
and do sqlite3 .prodigy/prodigy.db
it has "dataset", "example", "link" tables, and your data will be under example table.

