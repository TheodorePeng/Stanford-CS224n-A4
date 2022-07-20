from collections import Counter
import re
import numpy as np
import sacrebleu


def n_gram_generator(sent, n=2):
    tokens = sent.split(' ')
    src_len = len(tokens)

    n_gram_list = []
    for i in range(src_len - n + 1):
        n_gram_list.append(' '.join(tokens[i: i + n]))
    
    return Counter(n_gram_list)


def calculate_BP(references, hypothesis):
    len_c = len(hypothesis.split(' '))
    len_r = min([len(ref.split(' ')) for ref in references])
    print('len_c: ', len_c)
    print('len_r: ', len_r)
    if len_c >= len_r:
        return 1
    return np.exp(1 - (float(len_r) / float(len_c)))


def calculate_bleu(references, hypothesis, lambdas=[0.25, 0.25, 0.25, 0.25]):
    p_n = []
    bp = calculate_BP(references, hypothesis)
    print('bp: ', bp)

    for n in range(1, 5):
        hyp_n_gram_list = n_gram_generator(hypothesis, n)
        count_c = sum(hyp_n_gram_list.values())
        n_gram_sum = 0
        ref_n_gram_lists = [n_gram_generator(ref, n) for ref in references]
        for n_gram in hyp_n_gram_list:
            max_count = max([ref_n_gram_list.get(n_gram, 0) for ref_n_gram_list in ref_n_gram_lists])
            n_gram_sum += min([max_count, hyp_n_gram_list.get(n_gram, 0)])
        p_n.append(float(n_gram_sum) / float(count_c))

    print('p_n: ', p_n)

    weighted_scores = [la * np.log(p_i) for (la, p_i) in zip(lambdas, p_n)]

    bleu_score = bp * np.exp(sum(weighted_scores))

    print('my bleu score: ', bleu_score)

    print('sacre_bleu: ', sacrebleu.corpus_bleu([hypothesis], [references], ).score)


if __name__ == '__main__':
    hyp = 'the light shines the darkness has not in the darkness and the trials'
    # hyp = 'and the light shines in the darkness and the darkness can not comprehend'
    refs = [
        'the light shines in the darkness and the darkness has not overcome it',
        # 'and the light shines in the darkness and the darkness did not comprehend it'
    ]

    calculate_bleu(refs, hyp, [0.5, 0.5, 0, 0])


