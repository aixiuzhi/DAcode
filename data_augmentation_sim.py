# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
import gzip
import json
import random
import re

import tensorflow as tf
import tensorflow_hub as hub
import os

from nltk.corpus import stopwords, wordnet
from eda import *

# arguments to be parsed from command line
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="../data/hotpot/train.jsonl.gz", type=str, help="input file of unaugmented data")
ap.add_argument("--output", default="hotpot_train_sim_50.jsonl", type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", default=16, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", default=0.05, type=float,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", default=0.3, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", default=0.3, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", default=0.3, type=float, help="percent of words in each sentence to be deleted")
ap.add_argument("--USE_cache_path", default="../model/3", help="percent of words in each sentence to be deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join

    output = join(dirname(args.input), 'eda_' + basename(args.input))

# number of augmented sentences to generate per original sentence
num_aug = 9  # default
if args.num_aug:
    num_aug = args.num_aug

# how much to replace each word by synonyms
alpha_sr = 0.1  # default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

# how much to insert new words that are synonyms
alpha_ri = 0.1  # default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

# how much to swap words
alpha_rs = 0.1  # default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

# how much to delete words
alpha_rd = 0.1  # default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
    ap.error('At least one alpha should be greater than zero')


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        module_path = cache_path
        self.embed = hub.Module(module_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(),
                       tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

def swap_sentence(parts, answers):
    answer_in_sentence_indexs = []
    for i, sentence in enumerate(parts):
        for answer in answers:
            if answer in sentence:
                answer_in_sentence_indexs.append(i)

    random_index_1 = random.randint(0, len(parts) - 1)
    if len(parts) > 3:
        for answer_in_sentence_index in answer_in_sentence_indexs:
            while random_index_1 == answer_in_sentence_index:
                random_index_1 = random.randint(0, len(parts) - 1)
            random_index_2 = random_index_1

            while random_index_1 == random_index_2:
                random_index_2 = random.randint(0, len(parts) - 1)
                while random_index_2 == answer_in_sentence_index:
                    random_index_2 = random.randint(0, len(parts) - 1)
                while parts[random_index_1] == "<P>" or parts[random_index_2] == "<P>":  # 此处可在之后进行修改
                    random_index_1 = random.randint(0, len(parts) - 1)
                    random_index_2 = random.randint(0, len(parts) - 1)

                parts[random_index_1], parts[random_index_2] = parts[random_index_2], parts[random_index_1]

                return parts
                # new_parts = ""
                # for part in parts:
                #     new_parts.join(part)
                #
                # return new_parts

    else:
        # new_parts = parts
        # return new_parts
        return parts


def del_sentences(sentences, answers):
    random_index = random.randint(0, len(sentences) - 1)
    if len(sentences) > 3:
        for answer in answers:
            count = 0
            while answer in sentences[random_index]:
                random_index = random.randint(0, len(sentences) - 1)
                count += 1
                if count > 2:
                    del sentences[random_index]
                    return sentences
        del sentences[random_index]
        return sentences
    else:
        return sentences


def synonym_replacement(words):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= 1:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) < 3:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0

    if len(new_words) > 1:
        while len(synonyms) < 1:

            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    else:
        return new_words


def merge_passages(titles, title2context, s):
    target_title, target_line = s[0], s[1]

    sindex = 0
    eindex = 0
    answer_indices = None
    passage_text = ""
    line2charindex = []

    answer_text = title2context[target_title][target_line].strip()
    for title in titles:
        lines = [l.strip() for l in title2context[title]]
        # Remove the last line if it is empty
        if lines[-1] == "":
            lines.pop()

        text = " ".join(lines)
        se_indices = []
        for line in lines:
            se_indices.append((sindex, sindex + len(line) - 1))
            sindex += len(line) + 1

        if title == target_title:
            answer_indices = se_indices[target_line]

        line2charindex += se_indices
        passage_text += text + " "
    passage_text = passage_text.strip()
    sindex, eindex = answer_indices

    assert answer_text == passage_text[sindex:eindex + 1]
    assert line2charindex[-1][1] == len(passage_text) - 1

    return line2charindex, passage_text, sindex, eindex, answer_text


def answer_index(input_text, answer):
    start_index = input_text.find(answer)
    end_index = start_index + len(answer) - 1

    return start_index, end_index


def replace_sentence(parts, f_parts, answers):
    i = random.randint(0, len(parts) - 1)
    j = random.randint(0, len(f_parts) - 1)
    for answer in answers:
        while answer in parts[i]:
            i = random.randint(0, len(parts) - 1)

        parts[i] = f_parts[j]

    return parts


def sim_calculation(text, sim_predictor):
    print(text)
    orig_text = text
    sims = []
    for i, each_word in enumerate(text):
        if each_word in stopwords:
            continue
        else:
            text[i] = ""
        sims.append(sim_predictor.semantic_sim(orig_text, text))
    return sorted(sims)[-1]

# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    writer = open(output_file, 'w')
    data = gzip.GzipFile(train_orig, 'r')
    content = data.read().decode('utf-8').strip().split('\n')[1:]
    content = content[:50]
    input_data = [json.loads(line) for line in content]

    sim_predictor = USE(args.USE_cache_path)
    # input_data = []
    # for num in range(int(len(input_data_o) * 0.1)):
    #     input_data.append(input_data_o[num])
    #     # print(input_data)
    #
    print(len(input_data))
    for i, line in enumerate(input_data):
        if i == 0:
            f_data = input_data[i]
        else:
            f_data = input_data[i - 1]
        new_dict = {}
        new_dict_rs = {}
        new_dict_ss = {}
        new_dict_ds = {}
        new_dict_sr = {}
        new_dict_ri = {}
        new_dict_rd = {}
        context = line["context"]
        f_context = f_data["context"]
        qas = line["qas"]

        answers = []
        for qa in qas:
            answer = qa["answers"]
            for each_answer in answer:
                answers.append(each_answer)

        parts = context.split('.')
        f_parts = f_context.split('.')
        aug_sr = ""
        aug_ri = ""
        aug_rd = ""
        aug = ""


        augmented_sentences_sr = []
        augmented_sentences_ri = []
        augmented_sentences_rd = []
        for sentence in parts:
            answer_in_sentence_list = []
            answer_in_sentence = ""

            sentence_sim = sim_calculation(sentence, sim_predictor)

            for answer_d in answers:
                if answer_d in sentence:
                    # answer_in_sentence = sentence
                    # aug_sr += sentence + '.'
                    # aug_ri += sentence + '.'
                    # aug_rd += sentence + '.'
                    aug += sentence + '.'
                    break

                    # answer_in_sentence.split(" ")
                    # answer_in_sentence = [w_d for w_d in sentence if w_d is not '']
                    # for w_a in sentence:
                    #     answer_in_sentence_list.append(w_a)
                    # answer_in_sentence_list.append(answer_in_sentence)

            words = sentence.split(' ')
            words = [word for word in words if word is not '']
            sentence_sim = sim_calculation(words, sim_predictor)
            num_words = len(words)

            rand_num = random.randint(0, 1)
            if rand_num < 0.3:
                a_words_sr = synonym_replacement(words)
                # print(a_words_sr)
                sent_sr = ""
                for w_d in a_words_sr:
                    sent_sr += w_d + " "
                # print(sent_sr)
                aug += sent_sr + '. '
                # print(aug_sr)
            elif rand_num > 0.3 and rand_num < 0.6:
                n_ri = max(1, int(alpha_ri * num_words))
                a_words_ri = random_insertion(words, n_ri)
                # print(a_words_ri)
                sent_ri = ""
                for w_d in a_words_ri:
                    sent_ri += w_d + " "
                # print(sent_ri)
                aug += sent_ri + '. '
                # print(aug_ri)
            else:
                n_rd = max(1, int(alpha_rd * num_words))
                a_words_rd = random_deletion(words, alpha_rd)
                sent_rd = ""
                # print(a_words_rd)
                for w_d in a_words_rd:
                    sent_rd += w_d + " "
                # print(sent_rd)
                aug += sent_rd + '. '
        aug = re.sub(r'\n', '', aug)

        new_parts = swap_sentence(parts, answers)
        # print(new_parts)
        aug_ss = ""
        if new_parts is None:
            new_parts = parts
            for new_part in new_parts:
                aug_ss += new_part + "."
        else:
            for new_part in new_parts:
                aug_ss += new_part + "."
        aug_ss = re.sub(r'\n', '', aug_ss)
        # print(aug_ss)

        parts_del = del_sentences(parts, answers)
        # print(parts_del)
        aug_ds = ""
        for part_del in parts_del:
            aug_ds += part_del + "."

        aug_rs = ""
        parts_rs = replace_sentence(parts, f_parts, answers)
        for part_rs in parts_rs:
            aug_rs += part_rs + "."
        aug_rs = re.sub(r'\n', '', aug_rs)

        qas_ss = []
        qas_ds = []
        qas_rs = []
        qas_sr = []
        qas_ri = []
        qas_rd = []

        for qa in qas:
            qa_ss = {}
            qa_ss.setdefault("question", qa["question"])
            qa_ss.setdefault("answers", qa["answers"])
            qa_ss.setdefault("qid", qa["qid"])
            # qa_ss.setdefault("question_tokens", qa["question_tokens"])
            detected_answer = []
            detected_answers = qa["detected_answers"]
            for each_da in detected_answers:
                detected_answers_dict = {}
                detected_answers_dict.setdefault("text", each_da["text"])
                # print(aug_ss)
                # print("*******")
                # print(each_da["text"])
                ss_start_index, ss_end_index = answer_index(aug_ss, each_da["text"])
                char_spans = []
                char_spans.append([ss_start_index, ss_end_index])
                # print(char_spans)
                detected_answers_dict.setdefault("char_spans", char_spans)
                detected_answer.append(detected_answers_dict)

            qa_ss.setdefault("detected_answers", detected_answer)
            qas_ss.append(qa_ss)

        for qa in qas:
            qa_ds = {}
            qa_ds.setdefault("question", qa["question"])
            qa_ds.setdefault("answers", qa["answers"])
            qa_ds.setdefault("qid", qa["qid"])
            # qa_ds.setdefault("question_tokens", qa["question_tokens"])
            detected_answer = []
            detected_answers = qa["detected_answers"]
            for each_da in detected_answers:
                detected_answers_dict = {}
                detected_answers_dict.setdefault("text", each_da["text"])

                ds_start_index, ds_end_index = answer_index(aug_ds, each_da["text"])
                char_spans = []
                char_spans.append([ds_start_index, ds_end_index])
                detected_answers_dict.setdefault("char_spans", char_spans)
                detected_answer.append(detected_answers_dict)

            qa_ds.setdefault("detected_answers", detected_answer)
            qas_ds.append(qa_ds)

        for qa in qas:
            qa_rs = {}
            qa_rs.setdefault("question", qa["question"])
            qa_rs.setdefault("answers", qa["answers"])
            qa_rs.setdefault("qid", qa["qid"])
            # qa_rs.setdefault("question_tokens", qa["question_tokens"])
            detected_answer = []
            detected_answers = qa["detected_answers"]
            for each_da in detected_answers:
                detected_answers_dict = {}
                detected_answers_dict.setdefault("text", each_da["text"])

                rs_start_index, rs_end_index = answer_index(aug_rs, each_da["text"])
                char_spans = []
                char_spans.append([rs_start_index, rs_end_index])
                detected_answers_dict.setdefault("char_spans", char_spans)
                detected_answer.append(detected_answers_dict)

            qa_rs.setdefault("detected_answers", detected_answer)
            qas_rs.append(qa_rs)

        for qa in qas:
            qa_sr = {}
            qa_sr.setdefault("question", qa["question"])
            qa_sr.setdefault("answers", qa["answers"])
            qa_sr.setdefault("qid", qa["qid"])
            # qa_sr.setdefault("question_tokens", qa["question_tokens"])
            detected_answer = []
            detected_answers = qa["detected_answers"]
            for each_da in detected_answers:
                detected_answers_dict = {}
                detected_answers_dict.setdefault("text", each_da["text"])
                # print(aug)
                sr_start_index, sr_end_index = answer_index(aug, each_da["text"])
                char_spans = [[sr_start_index, sr_end_index]]
                # print(char_spans)
                detected_answers_dict.setdefault("char_spans", char_spans)
                detected_answer.append(detected_answers_dict)

            qa_sr.setdefault("detected_answers", detected_answer)
            qas_sr.append(qa_sr)

        # new_dict.setdefault('context', context)
        new_dict.setdefault("context", context)
        new_dict.setdefault("qas", qas)
        # new_dict.setdefault("context_tokens", context_tokens)
        # print(new_dict)

        new_dict_ss.setdefault("context", aug_ss[:-1])
        new_dict_ss.setdefault("qas", qas_ss)
        # new_dict_ss.setdefault("context_tokens", context_tokens)
        # print(new_dict_ss)

        new_dict_ds.setdefault("context", aug_ds[:-1])
        new_dict_ds.setdefault("qas", qas_ds)
        # new_dict_ds.setdefault("context_tokens", context_tokens)

        new_dict_rs.setdefault("context", aug_rs[:-1])
        new_dict_rs.setdefault("qas", qas_rs)
        # print(aug)
        new_dict_sr.setdefault("context", aug[:-1])
        new_dict_sr.setdefault("qas", qas_sr)
        # new_dict_sr.setdefault("context_tokens", context_tokens)
        # print(new_dict_sr)

        new_dicts = []
        new_dicts.append(new_dict)
        new_dicts.append(new_dict_ss)
        # new_dicts.append(new_dict_ds)
        new_dicts.append(new_dict_rs)
        new_dicts.append(new_dict_sr)
        # new_dicts.append(new_dict_ri)
        # new_dicts.append(new_dict_rd)

        for each in new_dicts:
            writer.write(json.dumps(each) + '\n')

    # writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(
        num_aug))


# main function
if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd,
            num_aug=num_aug)