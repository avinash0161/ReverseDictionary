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
tf.app.flags.DEFINE_string("embeddings_path","embeddings/GoogleWord2Vec.clean.normed.pkl","Path to pre-trained (.pkl) word embeddings.")

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

def build_model(max_seq_len, vocab_size, input_embedding_size, learning_rate, output_embs_for_all_vocab):
    with tf.device("/device:GPU:0"):
        tf.reset_default_graph()
        gloss_in = tf.placeholder(tf.int32, [None, max_seq_len], name="input_placeholder")
        head_in = tf.placeholder(tf.int32, [None], name="labels_placeholder")
        with tf.variable_scope("embeddings"):
            input_embedding_matrix = tf.get_variable(name="inp_embs_for_all_vocab", shape=[vocab_size, input_embedding_size])
        embs_for_gloss_in = tf.nn.embedding_lookup(input_embedding_matrix, gloss_in)

        output_emb_size = output_embs_for_all_vocab.shape[-1]

        core_out = tf.reduce_mean(embs_for_gloss_in, axis=1)
        output_form = "cosine"
        out_emb_matrix = tf.get_variable(name="out_embs_for_all_vocab",shape=[vocab_size, output_emb_size],
            initializer=tf.constant_initializer(output_embs_for_all_vocab), trainable=False)
        embs_for_head = tf.nn.embedding_lookup(out_emb_matrix, head_in)

        core_out = tf.contrib.layers.fully_connected(core_out, output_emb_size, activation_fn=tf.tanh)

        losses = tf.losses.cosine_distance(tf.nn.l2_normalize(embs_for_head, 1),tf.nn.l2_normalize(core_out, 1),dim=1)
        total_loss = tf.reduce_mean(losses, name="total_loss")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        return gloss_in, head_in, total_loss, train_step, output_form

def read_Data(data_dir, vocab_size, limitedTrainingData, phase):
    glosses, heads = [], []
    if limitedTrainingData:
        gloss_path = os.path.join(data_dir, "%s.definitions.ids%s.gloss" % (phase, vocab_size))
        head_path = os.path.join(data_dir, "%s.definitions.ids%s.head" % (phase, vocab_size))
    else:
        gloss_path = os.path.join(data_dir, "%s.definitions.idsUNLIMITED.gloss" % phase)
        head_path = os.path.join(data_dir, "%s.definitions.idsUNLIMITED.head" % (phase, vocab_size))

    with tf.gfile.GFile(gloss_path, mode="r") as gloss_file:
        with tf.gfile.GFile(head_path, mode="r") as head_file:
            gloss, head = gloss_file.readline(), head_file.readline()
            while gloss and head:
                gloss_ids = np.array([int(x) for x in gloss.split()], dtype=np.int32)
                glosses.append(gloss_ids)
                heads.append(int(head))
                gloss, head = gloss_file.readline(), head_file.readline()
    return np.asarray(glosses), np.array(heads, dtype=np.int32)

def gen_batch(training_glossAndHeads, batch_size):
  gloss, head = training_glossAndHeads
  data_length = len(gloss)
  num_batches = data_length // batch_size
  data_x, data_y = [], []
  for i in range(num_batches):
    data_x = gloss[batch_size * i:batch_size * (i + 1)]
    data_y = head[batch_size * i:batch_size * (i + 1)]
    yield (data_x, data_y)

def gen_epochs(data_dir, total_epochs, batch_size, vocab_size, limitedTrainingData, phase):
    training_glossAndHeads = read_Data(data_dir, vocab_size, limitedTrainingData, phase)
    for _ in range(total_epochs):
        yield gen_batch(training_glossAndHeads, batch_size)


def train_network(model, num_epochs, batch_size, data_dir, save_dir, vocab_size, limitedTrainData):
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        gloss_in, head_in, total_loss, train_step, _ = model
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(data_dir, num_epochs, batch_size, vocab_size, limitedTrainData, "train")):
            print("\nEPOCH", idx)
            training_loss = 0
            for step, (gloss, head) in enumerate(epoch):
                training_loss_, _ = sess.run([total_loss, train_step],feed_dict={gloss_in: gloss, head_in: head})
                training_loss += training_loss_
                if step % 500 == 0 and step > 0:
                    loss_ = training_loss / 500  ###### Seems contentious
                    print("Average loss step %s, for last 500 steps: %s" % (step, loss_))
                    training_losses.append(training_loss / 500)
                    training_loss = 0
            save_path = os.path.join(save_dir, "recurrent_%s.ckpt" % idx)
            save_path = saver.save(sess, save_path)
            print("Model saved in file: %s after epoch: %s" % (save_path, idx))
        return save_dir, saver

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
    while True:
        sys.stdout.write("Type a definition: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sys.stdout.write("Number of candidates: ")
        sys.stdout.flush()
        top = int(sys.stdin.readline())
        token_ids = utils.sentence_to_token_ids(sentence, vocab)
        padded_ids = np.asarray(utils.pad_sequence(token_ids, max_seq_len))

        input_data = np.asarray([padded_ids])
        model_preds = sess.run(predictions, feed_dict={input_node: input_data})
        sims = 1 - np.squeeze(dist.cdist(model_preds, output_embs_for_all_vocab, metric="cosine"))
        sims = np.nan_to_num(sims)
        candidate_ids = sims.argsort()[::-1][:top]
        candidates = [rev_vocab[idx] for idx in candidate_ids]

        print("\n Top %s candidates from the RNN model:" % top)
        for ii, cand in enumerate(candidates):
            print("%s: %s" % (ii + 1, cand))

        sys.stdout.flush()
        sentence = sys.stdin.readline()

def main(unused_argv):
    vocab_file = os.path.join(FLAGS.data_dir,"definitions_%s.vocab" % FLAGS.vocab_size)
    if not FLAGS.restore:
        input_emb_size = FLAGS.input_embedding_size #500
        output_embs_dict = utils.load_pretrained_target_embeddings(FLAGS.embeddings_path)
        output_emb_dim = 300
        utils.prepare_and_save_data(FLAGS.data_dir, FLAGS.train_file, FLAGS.dev_file, FLAGS.limitedVocab, FLAGS.vocab_size, FLAGS.limitedTrainData, FLAGS.max_seq_len)
        vocab, rev_vocab = utils.getVocabulary(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
        output_embs_for_all_vocab = utils.get_embedding_matrix(output_embs_dict, vocab, output_emb_dim)

        model = build_model(FLAGS.max_seq_len, FLAGS.vocab_size,FLAGS.input_embedding_size,FLAGS.learning_rate,output_embs_for_all_vocab)

        save_path, saver = train_network(model,FLAGS.num_epochs,FLAGS.batch_size,FLAGS.data_dir,FLAGS.save_dir,FLAGS.vocab_size, FLAGS.limitedTrainData)

    else:
        output_embs_dict = utils.load_pretrained_target_embeddings(FLAGS.embeddings_path)
        output_emb_dim = 300
        vocab, rev_vocab = utils.getVocabulary(FLAGS.data_dir, FLAGS.limitedVocab, FLAGS.vocab_size)
        output_embs_for_all_vocab = utils.get_embedding_matrix(output_embs_dict, vocab, output_emb_dim)

        with tf.device("/cpu:0"):
            with tf.Session() as sess:
                (input_node, target_node, predictions, loss) = restore_model(sess, FLAGS.save_dir, vocab_file)
                query_model(sess, input_node, predictions, vocab, rev_vocab, FLAGS.max_seq_len, output_embs_for_all_vocab)


if __name__ == "__main__":
    tf.app.run()
