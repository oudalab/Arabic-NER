"""
Simple script to convert OntoNotes 5.0 formatted `.name` files to the spaCy
NER-only training format (https://github.com/explosion/spaCy/blob/master/examples/training/train_ner.py#L21)
"""
import json # for tuple support
import plac
import os
import re

def get_ner_files(onto_dir):
    name_files = []
    for dirpath, subdirs, files in os.walk(onto_dir):
        for fname in files:
            if bool(re.search(".name", fname)):
                fn = os.path.join(dirpath, fname)
                name_files.append(fn)
    return name_files

def split(text):
    text = text.strip().split('\n')[1:-1]
    return text

def clean_ent(ent):
    tag = re.findall('TYPE="(.+?)">', ent)[0]
    text = re.findall('>(.+)', ent)[0]
    text = re.sub("\$", "\$", text)
    return (text, tag)

def raw_text(text):
    text = re.sub("<ENAMEX .+?>", "", text)
    text = re.sub("</ENAMEX>", "", text)
    return text

def ent_position(ents, text):
    spacy_ents = []
    for ent in ents:
        ma = re.search(ent[0], text)
        ent_tup = (ma.start(), ma.end(), ent[1])
        spacy_ents.append(ent_tup)
    return spacy_ents

def text_to_spacy(markup):
    ents = re.findall("<ENAMEX(.+?)</ENAMEX>", markup)
    ents = [clean_ent(ent) for ent in ents]
    text = raw_text(markup)
    spacy_ents = ent_position(ents, text)
    final = (text, {"entities" : spacy_ents})
    return final

@plac.annotations(
    onto_dir=("Directory of OntoNotes data to traverse", "option", "i", str),
    output=("File to write spaCy NER JSON out to", "option", "o", str))
def main(onto_dir, output):
    fns = get_ner_files(onto_dir)
    all_annotations = []
    for fn in fns:
        with open(fn, "r") as f:
            markup_doc = f.read()
            markup_list = split(markup_doc)
            for markup in markup_list:
                try:
                    ents = text_to_spacy(markup)
                    all_annotations.append(ents)
                except Exception:
                    print(markup)
    #print(all_annotations)
    with open(output, "w") as f:
        json.dump(all_annotations, f)

if __name__ == "__main__":
    plac.call(main)
