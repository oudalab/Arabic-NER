import plac
import spacy
from spacy.language import Language
from tqdm import tqdm
import numpy as np
tqdm.monitor_interval = 0

def make_language(language, n, nlp, input_dir, max_lang_vocab, max_total_vocab):
    """
    Given a language code, a directory with its aligned vectors, and a spaCy model, add aligned word vectors to the model.

    This code was mostly taken from https://github.com/explosion/spacy/blob/master/examples/vectors_fast_text.py
    """
   # vectors_loc = input_dir + "wiki.{0}.vec".format(language)
    vectors_loc ="/home/yan/arabicner/cc.ar.300.vec"
    print("Reading aligned vector {0} and adding to spaCy model.".format(vectors_loc))
    print("previous vocab length: ", len(nlp.vocab))
    prev_vec_len = len(nlp.vocab.vectors)
    print("previous vector length: ", prev_vec_len)
    with open(vectors_loc, 'rb') as file_:
        header = file_.readline()
        nr_row, nr_dim = header.split()
        #nlp.vocab.reset_vectors(width=int(nr_dim)) <-- DUH
        total_to_add = min(max_lang_vocab, int(nr_row))
        i = 0
        for line in tqdm(file_, total = total_to_add):
            if i > max_lang_vocab +1:
                break
            i += 1
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
            # hack to get set_vector to work:
            lex = nlp.vocab[word]
            # add the vectors to the vocab
            nlp.vocab.set_vector(word, vector)

    print("New vocab length, pre-trimming: ", len(nlp.vocab))
    post_vec_len = len(nlp.vocab.vectors)
    print("New vector length, pre-trimming: ", post_vec_len)   
 
    # Prune each new language down to 80% (or actually, prune whole vocab, but by 80% of new words),
    # or, if the vector length is over the max total, trim to that.
    new_words = min(max_lang_vocab, int(nr_row))
    print("Added {0} new words.".format(new_words))
    intermediate_max = int(prev_vec_len) + round(int(new_words) * 0.8)
    intermediate_max = min(intermediate_max, max_total_vocab)
    print("Intermediate pruning to {0} words.".format(intermediate_max))
    nlp.vocab.prune_vectors(intermediate_max) #, batch_size = 20000)
    print("New vocab length, post-trimming: ", len(nlp.vocab))
    post_vec_len = len(nlp.vocab.vectors)
    print("New vector length, post-trimming: ", post_vec_len)   
    return nlp

@plac.annotations(
    languages=("Language codes to use, two letter, comma separated", "option", "l", str),
    version=("Version number for the model (required)", "option", "v", str),
    input_dir=("Directory holding aligned vectors from", "option", "i", str),
    output_dir=("Location to write spaCy model to", "option", "o", str),
    max_lang_vocab=("Maximum number of vocabulary words to add per language", "option", "ml", int),
    max_total_vocab=("Maximum total number of unique vectors after pruning", "option", "mt", int))
def main(languages, version, input_dir = "/tmp/ar_vectors_wiki_lg/", output_dir = "./models/", max_lang_vocab=50000,
         max_total_vocab = 1000):
    """
    /tmp/ar_vectors_wiki_lg
    Example: python create_spacy_model.py -l en,de,hr -i ../data/vectors/ -o ../models/ -m 200000
    """
    print("Adding {0} most frequent words per language.".format(max_lang_vocab))
    nlp = spacy.blank("xx")
    nlp.vocab.reset_vectors(width=int(300))
    lang_list = languages.split(",")

    for n, lang in enumerate(lang_list):
        print("nlp vocab length at next run: ", len(nlp.vocab))
        print("nlp vector length at next run: ", len(nlp.vocab.vectors))
        nlp = make_language(lang, n, nlp, input_dir, max_lang_vocab, max_total_vocab)
    
    doc = nlp("Frau frau frauen Woman woman women")
    print([i.vector[0:5] for i in doc])
    print("Word rows: ", [i.rank for i in doc])
    print("Pruning vocabulary to total size of {0}".format(max_total_vocab))
    nlp.vocab.prune_vectors(max_total_vocab, batch_size = 16384)
    print("Assigning title case words...")
    string_list = [i.text for i in nlp.vocab]
    for string in string_list:
        nlp.vocab.vectors.add(string.title(), row = nlp.vocab[string].rank)
    doc = nlp("Frau frau frauen Woman woman women")
    print([i.vector[0:5] for i in doc])
    print("Word rows: ", [i.rank for i in doc])
    outfile = output_dir + "xx_raw_fasttext_model"
    nlp.meta['lang'] = 'xx'
    nlp.meta['version'] = version
    nlp.meta['author'] = "Andy Halterman"
    desc = 'A vector-only spaCy model with NON-aligned (!!) word embeddings for {0}. Per-lang vocab was limited to {1}, and total vocab to {2}.'.format(languages, max_lang_vocab, max_total_vocab)   
    nlp.meta['description'] = desc
    nlp.to_disk(outfile)
    print("Saved new spaCy model to {0}.".format(outfile))


if __name__ == '__main__':
    plac.call(main)

