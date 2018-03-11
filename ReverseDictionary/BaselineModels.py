import tensorflow as tf
import utils
import sys
import numpy as np
import scipy.spatial.distance as dist

tf.app.flags.DEFINE_string("data_dir", "data/definitions/", "Directory for finding training data and dumping processed data.")
tf.app.flags.DEFINE_string("train_file", "train.definitions.ids100000", "File with dictionary definitions for training.")
tf.app.flags.DEFINE_string("dev_file", "'dev.definitions.ids100000", "File with dictionary definitions for dev testing.")
tf.app.flags.DEFINE_string("embeddings_path","embeddings/GoogleWord2Vec.clean.normed.pkl","Path to pre-trained (.pkl) word embeddings.")

tf.app.flags.DEFINE_boolean("limitedVocab",False,"Whether to limit the vocabulary")
tf.app.flags.DEFINE_integer("vocab_size", 100000, "Nunber of words the model knows and stores representations for")
tf.app.flags.DEFINE_integer("max_seq_len", 20, "Maximum length (in words) of a definition processed by the model")

FLAGS = tf.app.flags.FLAGS

def get_Candidates_Answers(base_rep, pre_emb_for_all_vocab, top, rev_vocab):
    sims_base = 1 - np.squeeze(dist.cdist(base_rep, pre_emb_for_all_vocab, metric="cosine"))
    sims_base = np.nan_to_num(sims_base)
    candidate_ids_base = sims_base.argsort()[::-1][:top]
    candidates_base_mean = [rev_vocab[idx] for idx in candidate_ids_base]
    return candidates_base_mean

def queryBaseline(pre_emb_for_all_vocab, vocab, rev_vocab):
    while True:
        sys.stdout.write("Type a definition: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sys.stdout.write("Number of candidates: ")
        sys.stdout.flush()
        top = int(sys.stdin.readline())
        token_ids = utils.sentence_to_token_ids(sentence, vocab)

        base_rep_mean = np.asarray([np.mean(pre_emb_for_all_vocab[token_ids], axis=0)])
        print("Top %s baseline candidates from W2V mean/add model:" % top)
        for ii, cand in enumerate(get_Candidates_Answers(base_rep_mean, pre_emb_for_all_vocab, top, rev_vocab)):
            print("%s: %s" % (ii + 1, cand))

        # base_rep_add = np.asarray([np.sum(pre_emb_for_all_vocab[token_ids], axis=0)])
        # print("Top %s baseline candidates from W2V add model:" % top)
        # for ii, cand in enumerate(get_Candidates_Answers(base_rep_add, pre_emb_for_all_vocab, top, rev_vocab)):
        #     print("%s: %s" % (ii + 1, cand))

        base_rep_mult = np.asarray([np.prod(pre_emb_for_all_vocab[token_ids], axis=0)])
        print("Top %s baseline candidates from W2V mult model:" % top)
        for ii, cand in enumerate(get_Candidates_Answers(base_rep_mult, pre_emb_for_all_vocab, top, rev_vocab)):
            print("%s: %s" % (ii + 1, cand))

def queryBaselineWithConecptDesc(pre_emb_for_all_vocab, vocab, rev_vocab):
    with tf.gfile.GFile("data/definitions/concept_descriptions.tok", mode="r") as data_file:
        with tf.gfile.GFile("data/output/concept_Baseline.txt", mode="w") as output_file:
            for line in data_file:
                top = 10
                token_ids = utils.sentence_to_token_ids(line, vocab)
                base_rep_mean = np.asarray([np.mean(pre_emb_for_all_vocab[token_ids[1:]], axis=0)])
                print("Top %s baseline candidates from W2V mean/add model:" % top)
                for ii, cand in enumerate(get_Candidates_Answers(base_rep_mean, pre_emb_for_all_vocab, top, rev_vocab)):
                    output_file.write(cand + " ")
                    print(cand + " ")
                output_file.write("\n")
                output_file.flush()
                print("\n")

def main(unused_argv):
    utils.prepareVocabFilesAndGetDataForBaseline(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
    vocab, rev_vocab = utils.getVocabulary(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
    embs_dict = utils.load_pretrained_target_embeddings(FLAGS.embeddings_path)
    pre_emb_dim = 300

    pre_embs_for_all_vocab = utils.get_embedding_matrix(embs_dict, vocab, pre_emb_dim)
    # queryBaseline(pre_embs_for_all_vocab, vocab, rev_vocab)
    queryBaselineWithConecptDesc(pre_embs_for_all_vocab, vocab, rev_vocab)

if __name__ == "__main__":
    tf.app.run()