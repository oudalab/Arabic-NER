import spacy
nlp = spacy.blank("xx")
from tqdm import tqdm
import random
import plac
import os
import re
from prodigy import set_hashes
from prodigy.components.db import connect


def get_root_filename(onto_dir):
    name_files = []
    for dirpath, subdirs, files in os.walk(onto_dir):
        for fname in files:
            if bool(re.search(".name", fname)):
                fn = os.path.join(dirpath, fname)
                fn = re.sub("\.name", "", fn)
                name_files.append(fn)
    return name_files


def split_sentence(text):
    text = text.strip().split('\n')[1:-1]
    return text


def split_doc(text):
    text_list = text.strip().split('</DOC>\s<DOC')
    ids = [re.findall('<DOC DOCNO="(.+?)">', t)[0] for t in text_list]
    text_list = [re.sub('<DOC DOCNO=".+?">', "", t).strip() for t in text_list]
    return ids, text_list


def clean_ent(ent):
    tag = re.findall('TYPE="(.+?)">', ent)[0]
    text = re.findall('>(.+)', ent)[0]
    text = re.sub("\$", "\$", text)
    return (text, tag)


def raw_text(text):
    """Remove entity tags"""
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


def onf_to_raw(onf_file):
    """
    Take in a path to a .onf Ontonotes file. Return the raw text (as much as possible).
    The quotes are usually quite messed up, so this is not going to look like real input text.
    """
    with open(onf_file, "r") as f:
        onf = f.read()
    sentences = re.findall("Plain sentence\:\n\-+?\n(.+?)Treebanked sentence", onf, re.DOTALL)
    sentences = [re.sub("\n+?\s*", " ", i).strip() for i in sentences]
    paragraph = ' '.join(sentences)
    return paragraph


def sent_with_offsets(ner_filename):
    """
    Take a .name file and return a sentence list of the kind described here:
    https://github.com/explosion/spacy/blob/master/examples/training/training-data.json
    """
    with open(ner_filename, "r") as f:
        doc = f.read()
    sentences = []
    onto_sents = split_sentence(doc)
    for sent in onto_sents:
        offsets = text_to_spacy(sent)
        sentences.append(offsets)
    return sentences


def dir_to_raw(onto_dir):
    fns = get_root_filename(onto_dir)
    all_annotations = []
    for fn in tqdm(fns):
        ner_filename = fn + ".name"
        onf_filename = fn + ".onf"
        try:
            raw = onf_to_raw(onf_filename)
            sentences = sent_with_offsets(ner_filename)
            final = {"id" : "fake",
                     "paragraphs" : [
                        {"raw" : raw,
                         "sentences" : sentences}]}
            all_annotations.append(final)
        except Exception as e:
            print("Error formatting ", fn, e)
    return all_annotations


def onto_to_prodigy_complete(sent):
    """
    Make an accepted Prodigy task with all NER spans.
    """
    spans = []
    for s in sent[1]['entities']:
        s = {"start" : s[0], "end" : s[1], "label" : s[2]}
        spans.append(s)
    prod = {"answer" : "accept",
            "text" : sent[0],
            "spans" : spans,
            "source" : "OntoNotes_rehearsal"}
    return prod

@plac.annotations(
    dataset=("Name of dataset with Prodigy annotated NER.", "option", "i", str),
    multiplier=("Number of OntoNotes annotation to add per newly collected annotation (5?).", "option", "m", int))
def main(dataset, multiplier, onto_dir="ontonotes-release-5.0/data/english/annotations/"):
    print("Reading OntoNotes")
    raw_annotations = dir_to_raw(onto_dir)
    print("Converting to spaCy spans...")
    all_onto = []
    for i in raw_annotations:
        for s in i['paragraphs'][0]['sentences']:
            all_onto.append(onto_to_prodigy_complete(s))
    all_onto = list(set_hashes(eg) for eg in all_onto)
    random.shuffle(all_onto)
    # get Prodigy annotations
    db = connect()
    annot = db.get_dataset(dataset)
    print("Found {0} annotations in {1}".format(len(annot), dataset))
    # Get the examples to augment
    augment = all_onto[0:multiplier * len(annot)]
    print("Augmenting existing examples with {0} OntoNotes sentences".format(len(augment)))
    both = augment + annot
    random.shuffle(both)
    # use a hardcoded rehearsal dataset because we're dropping and don't want
    # to take user input here. If it exists, drop it so we can refresh it.
    exs = db.get_dataset("augmented_for_training")
    if exs:
        db.drop_dataset("augmented_for_training")
    db.add_examples(both, ["augmented_for_training"])
    print("Wrote examples to 'augmented_for_training'. Use 'ner.batch-train' on that dataset.")

if __name__ == "__main__":
    plac.call(main)
