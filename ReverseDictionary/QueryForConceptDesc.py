from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utils
import sys
import numpy as np
import scipy.spatial.distance as dist
import os

tf.app.flags.DEFINE_string("data_dir", "data/definitions/", "Directory for finding training data and dumping processed data.")
tf.app.flags.DEFINE_string("train_file", "definitions.tok", "File with dictionary definitions for training.")
tf.app.flags.DEFINE_string("dev_file", "concept_descriptions.tok", "File with dictionary definitions for dev testing.")
tf.app.flags.DEFINE_string("embeddings_path","embeddings/glove.6B.300d.txt","Path to pre-trained (.pkl) word embeddings.")

tf.app.flags.DEFINE_boolean("limitedTrainData",True,"Whether to limit the training data")
tf.app.flags.DEFINE_boolean("limitedVocab",True,"Whether to limit the vocabulary")
tf.app.flags.DEFINE_integer("vocab_size", 100000, "Nunber of words the model knows and stores representations for")
tf.app.flags.DEFINE_integer("max_seq_len", 20, "Maximum length (in words) of a definition processed by the model")
tf.app.flags.DEFINE_boolean("restore", False, "Restore a trained model instead of training one.")
tf.app.flags.DEFINE_integer("input_embedding_size", 500, "Number of units in word representation.")

tf.app.flags.DEFINE_integer("num_epochs", 2, "Train for this number of sweeps through the training set")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate applied in TF optimiser")
tf.app.flags.DEFINE_string("save_dir", "/tmp/", "Directory for saving model. If using restore=True, directory to restore from.")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")

FLAGS = tf.app.flags.FLAGS

def restore_model(sess, save_dir, vocab_file):
    model_path = tf.train.latest_checkpoint(save_dir)
    saver = tf.train.import_meta_graph(model_path + ".meta")
    saver.restore(sess, model_path)
    graph = tf.get_default_graph()

    input_node = graph.get_tensor_by_name("input_placeholder:0")
    target_node = graph.get_tensor_by_name("labels_placeholder:0")
    predictions = graph.get_tensor_by_name("fully_connected/Tanh:0")
    loss = graph.get_tensor_by_name("total_loss:0")

    return input_node, target_node, predictions, loss

def query_model(sess, input_node, predictions, vocab, rev_vocab, max_seq_len, output_embs_for_all_vocab):
    with tf.gfile.GFile("data/definitions/concept_descriptions.tok", mode="r") as data_file:
        with tf.gfile.GFile("data/output/concept_BOW.txt", mode="w") as output_file:
            for line in data_file:
                top = 10
                token_ids = utils.sentence_to_token_ids(line, vocab)
                padded_ids = np.asarray(utils.pad_sequence(token_ids[1:], max_seq_len))

                input_data = np.asarray([padded_ids])
                model_preds = sess.run(predictions, feed_dict={input_node: input_data})
                sims = 1 - np.squeeze(dist.cdist(model_preds, output_embs_for_all_vocab, metric="cosine"))
                sims = np.nan_to_num(sims)
                candidate_ids = sims.argsort()[::-1][:top]
                candidates = [rev_vocab[idx] for idx in candidate_ids]
                for ii, cand in enumerate(candidates):
                    output_file.write(cand + " ")
                    print(cand + " ")
                output_file.write("\n")
                output_file.flush()
                print("\n")

def main(unused_argv):
    vocab_file = os.path.join(FLAGS.data_dir, "definitions_%s.vocab" % FLAGS.vocab_size)
    output_embs_dict = utils.load_pretrained_target_embeddings_from_file(FLAGS.embeddings_path)
    output_emb_dim = 300
    # vocab, rev_vocab = utils.getVocabularyIncludingEmbeddings(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
    vocabExtended, rev_vocab, vocab  = utils.getVocabularyIncludingEmbeddings(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
    # output_embs_for_all_vocab = utils.get_embedding_matrix(output_embs_dict, vocab, output_emb_dim)
    output_embs_for_all_vocab = utils.get_embedding_matrix(output_embs_dict, vocabExtended, output_emb_dim)

    with tf.device("/cpu:0"):
        with tf.Session() as sess:
            (input_node, target_node, predictions, loss) = restore_model(sess, FLAGS.save_dir, vocab_file)
            query_model(sess, input_node, predictions, vocab, rev_vocab, FLAGS.max_seq_len, output_embs_for_all_vocab)

if __name__ == "__main__":
    tf.app.run()