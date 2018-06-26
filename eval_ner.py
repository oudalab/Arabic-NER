import plac
from prodigy.models.ner import EntityRecognizer, merge_spans, guess_batch_size
from prodigy.components import printers
from prodigy.components.db import connect
from prodigy.util import get_print
import spacy

def eval_ner(baseline_model, new_model, eval_id, label):
    baseline_model = EntityRecognizer(spacy.load(baseline_model), label=label)
    new_model = EntityRecognizer(spacy.load(new_model), label=label)
    DB = connect()
    print_ = get_print(False)
    evals = DB.get_dataset(eval_id)
    print_("Loaded {} evaluation examples from '{}'".format(len(evals), eval_id))
    baseline = baseline_model.evaluate(evals)
    stats = new_model.evaluate(evals)
    # Baseline
    model_to_bytes = baseline_model.to_bytes()
    print("Baseline model accuracy:")
    print_(printers.ner_result(baseline, baseline['acc'], baseline['acc']))
    print("\n\nNew model accuracy:")
    model_to_bytes = new_model.to_bytes()
    best = (stats['acc'], stats, model_to_bytes)
    best_acc, best_stats, best_model = best
    print_(printers.ner_result(best_stats, best_acc, baseline['acc']))


if __name__ == "__main__":
    plac.call(eval_ner)
