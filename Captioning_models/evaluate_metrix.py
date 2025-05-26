from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def load_textfiles(references, hypothesis):
    #referencesはデータ(画像)数のreferenceが入ったリストが一つのデータ(画像)あたりのreference数個あるリストのリスト
    #hypothesisは一文が一要素のリスト    
    print("The number of references is {}".format(len(references)))
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    #raw_refs = [list(map(str.strip, r)) for r in zip(*references)]
    raw_refs = [list(map(str.strip, r)) for r in references]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs")
    return refs, hypo


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentences)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    #CIDErは10倍されている
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == "__main__":
    dum_references = [["This is a pen.", "He plays soccer in the park.", "A red car is stopping."]]
    dum_hypothesis = ["This is a pen.", "He plays soccer.", "A red car is stopping."]

    refs, hypo = load_textfiles(dum_references, dum_hypothesis)
    final_scores = score(refs, hypo)
    print(final_scores)