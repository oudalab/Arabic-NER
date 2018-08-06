from tqdm import tqdm
import random
import plac
import os
import re
import json
import pyarabic.araby as araby
from prodigy import set_hashes
from prodigy.components.db import connect

#update the labels in LDC to our standard used in prodigy labeled data.
from tqdm import tqdm
import random
import plac
import os
import re
import json
from prodigy import set_hashes
from prodigy.components.db import connect

#update the labels in LDC to our standard used in prodigy labeled data.
def update_ldc_labels(data):
    newdata=[]
    for d in data:
        #well there 4122 records with empty span
        if(len(d["spans"])==0):
            continue;
        newdata.append(d)
        label=d["spans"][0]["label"]
        if(label=='PERSON" S_OFF="1'):
            d["spans"][0]["label"]="PERSON"
        elif(label=="PERSON"):
            continue;
        elif(label== 'GPE" S_OFF="1'):
            d["spans"][0]["label"]="GPE"
        elif(label=='ORG" E_OFF="1'):
            d["spans"][0]["label"]="ORG"
        elif(label=='ORG" S_OFF="1'):
            d["spans"][0]["label"]="ORG"
        elif(label=='ORG'):
            continue;
        elif(label=="GPE"):
            continue;
        elif(label=="FAC" or label=="LOC"):
            d["spans"][0]["label"]="GPE"
        else:
            d["spans"][0]["label"]="MISC"
    return newdata;
#validate the labels are what we are looking for
def getlabelset(data):
    labelset={}
    for d in data:
        if(len(d["spans"])>0):
            la=d["spans"][0]["label"]
        else:
            continue
        if(la in labelset):
            labelset[la]=labelset[la]+1
        else:
            labelset[la]=1
    print(labelset)
    return labelset

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
    text = text.replace('{' , 'ا')
    text = text.replace('`' , '')
    text = text.replace('-RRB-' , ')')
    text = text.replace('-LRB-' , '(')
    text = text.replace('-' , '')
    text = araby.strip_tashkeel(text)
    return (text, tag)


def raw_text(text):
    """Remove entity tags"""
    text = re.sub("<ENAMEX .+?>", "", text)
    text = re.sub("</ENAMEX>", "", text)
    text = text.replace('{' , 'ا')
    text = text.replace('`' , '')
    text = text.replace('-RRB-' , ')')
    text = text.replace('-LRB-' , '(')
    text = text.replace('-','')
    text = araby.strip_tashkeel(text)
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
    paragraph = paragraph.replace('-RRB-' , ')')
    paragraph = paragraph.replace('-LRB-' , '(')
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
    dataset=("Name of dataset with Prodigy annotated NER.", "positional", None, str),
    multiplier=("Number of OntoNotes annotation to add per newly collected annotation (5?).","positional", None, int),
    split=("Should 20 percent of annotated and Onto data be pulled off for eval?", "flag", "s"),
    onto_dir=("Location of OntoNotes directory", "positional", None, str))
def main(dataset, multiplier, split, onto_dir="/home/yan/ontonotes-release-5.0/data/files/data/arabic/annotations"):
    """
    Mix in OntoNotes NER annotations with new NER annotations from Prodigy to avoid the catatrophic forgetting problem.

    Given a Prodigy dataset with new NER annotations, create a new dataset ('augmented_for_training') that also
    includes OntoNotes NER sentences mixed in. `prodigy ner.batch-train` can then be called on this dataset
    to learn on the new annotations without forgetting the old.
    See here for more information on the catastrophic forgetting problem: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    """
    print("Reading OntoNotes")
    raw_annotations = dir_to_raw(onto_dir)
    print("Converting to spaCy spans...")
    all_onto = []
    for i in raw_annotations:
        for s in i['paragraphs'][0]['sentences']:
            all_onto.append(onto_to_prodigy_complete(s))
    all_onto = list(set_hashes(eg) for eg in all_onto)
    print("length all_onto")
    print(len(all_onto))
    random.shuffle(all_onto)
    # get Prodigy annotations
    db = connect()
    annot = db.get_dataset(dataset)
    random.shuffle(annot)
    print("Found {0} annotations in {1}".format(len(annot), dataset))
    # Get the examples to augment
    aug_num = multiplier * len(annot)
    print("Augmenting existing examples with {0} OntoNotes sentences".format(aug_num))
    #with open("allonto.json","w") as outfile:
     #   json.dump(all_onto,outfile)
    augment = all_onto[0:aug_num]
    update_ldc_labels(augment)
    print("label set for augmented data:");
    getlabelset(augment)
    if split:
        cutpoint = round(len(annot) * 0.2)
        eval_prod = annot[0:cutpoint]
        annot = annot[cutpoint:]
        eval_onto = all_onto[aug_num:aug_num + 5*cutpoint]
    #both = augment + annot
    both=augment+all_onto
    #both=all_onto
    print("length for both")
    print(len(both))
    random.shuffle(both)
    # use a hardcoded rehearsal dataset because we're dropping and don't want
    # to take user input here. If it exists, drop it so we can refresh it.
    datasetname="augmented_for_training"
    exs = db.get_dataset(datasetname)
    if exs:
        db.drop_dataset(datasetname)
    db.add_examples(both, [datasetname])
    print("Wrote examples to the Prodigy dataset "+datasetname+" . Use 'ner.batch-train' on that dataset.")
    if split:
        eo = db.get_dataset("onto_for_eval")
        if eo:
            db.drop_dataset("onto_for_eval")
        ep = db.get_dataset("prodigy_for_eval")
        if ep:
            db.drop_dataset("prodigy_for_eval")
        db.add_examples(eval_onto, ["onto_for_eval"])
        db.add_examples(eval_prod, ["prodigy_for_eval"])
        print("Wrote eval examples to `prodigy_for_eval` and `onto_for_eval`")

if __name__ == "__main__":
    plac.call(main)

#validate the labels are what we are looking for
def getlabelset(data):
    labelset={}
    for d in data:
        if(len(d["spans"])>0):
            la=d["spans"][0]["label"]
        else:
            continue
        if(la in labelset):
            labelset[la]=labelset[la]+1
        else:
            labelset[la]=1
    return labelset

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
    dataset=("Name of dataset with Prodigy annotated NER.", "positional", None, str),
    multiplier=("Number of OntoNotes annotation to add per newly collected annotation (5?).","positional", None, int),
    split=("Should 20 percent of annotated and Onto data be pulled off for eval?", "flag", "s"),
    onto_dir=("Location of OntoNotes directory", "positional", None, str))
def main(dataset, multiplier, split, onto_dir="/home/yan/ontonotes-release-5.0/data/files/data/arabic/annotations"):
    """
    Mix in OntoNotes NER annotations with new NER annotations from Prodigy to avoid the catatrophic forgetting problem.

    Given a Prodigy dataset with new NER annotations, create a new dataset ('augmented_for_training') that also
    includes OntoNotes NER sentences mixed in. `prodigy ner.batch-train` can then be called on this dataset
    to learn on the new annotations without forgetting the old.
    See here for more information on the catastrophic forgetting problem: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    """
    print("Reading OntoNotes")
    raw_annotations = dir_to_raw(onto_dir)
    print("Converting to spaCy spans...")
    all_onto = []
    for i in raw_annotations:
        for s in i['paragraphs'][0]['sentences']:
            all_onto.append(onto_to_prodigy_complete(s))
    all_onto = list(set_hashes(eg) for eg in all_onto)
    print("length all_onto")
    print(len(all_onto))
    random.shuffle(all_onto)
    # get Prodigy annotations
    db = connect()
    annot = db.get_dataset(dataset)
    random.shuffle(annot)
    print("Found {0} annotations in {1}".format(len(annot), dataset))
    # Get the examples to augment
    aug_num = multiplier * len(annot)
    print("Augmenting existing examples with {0} OntoNotes sentences".format(aug_num))
    #with open("allonto.json","w") as outfile:
     #   json.dump(all_onto,outfile)
    newaugment = all_onto[0:aug_num]

    update_ldc_labels(newaugment)
    print("label set for augmented data:");
    augment=[];
    for data in newaugment:
        if(data['spans'][0]['label']=="" or data['spans'][0]['label']=="MISC"):
            continue
        augment.add(data)
    getlabelset(augment)
    if split:
        cutpoint = round(len(annot) * 0.2)
        eval_prod = annot[0:cutpoint]
        annot = annot[cutpoint:]
        eval_onto = all_onto[aug_num:aug_num + 5*cutpoint]
    #both = augment + annot
    both=augment+annot
    #both=all_onto
    print("length for both")
    print(len(both))
    random.shuffle(both)
    # use a hardcoded rehearsal dataset because we're dropping and don't want
    # to take user input here. If it exists, drop it so we can refresh it.
    datasetname="augmented_for_training_2"
    exs = db.get_dataset(datasetname)
    if exs:
        db.drop_dataset(datasetname)
    db.add_examples(both, [datasetname])
    print("Wrote examples to the Prodigy dataset "+datasetname+" . Use 'ner.batch-train' on that dataset.")
    if split:
        eo = db.get_dataset("onto_for_eval")
        if eo:
            db.drop_dataset("onto_for_eval")
        ep = db.get_dataset("prodigy_for_eval")
        if ep:
            db.drop_dataset("prodigy_for_eval")
        db.add_examples(eval_onto, ["onto_for_eval"])
        db.add_examples(eval_prod, ["prodigy_for_eval"])
        print("Wrote eval examples to `prodigy_for_eval` and `onto_for_eval`")

if __name__ == "__main__":
    plac.call(main)
