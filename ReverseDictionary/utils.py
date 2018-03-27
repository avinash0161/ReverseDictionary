import collections
import tensorflow as tf
import re
import os
import pickle
import numpy as np
from nltk.stem import PorterStemmer

_splitExpression = re.compile("([.,!?\"':;)(])")
_digitsExpression = re.compile(r"\d")
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

def basic_tokenizer(sentence):
    tokens = []
    for space_split_tokens in sentence.strip().split():
        tokens.extend(_splitExpression.split(space_split_tokens))
    return [t.lower() for t in tokens if t]

def createVocabFile(vocabFileStem, dataPath, limitedVocab = False, vocabLimit = 100000):
    if not tf.gfile.Exists(vocabFileStem + ".vocab"):
        print("Creating vocabulary %s from data %s" % (vocabFileStem, dataPath))
        head = collections.defaultdict(int)
        words = collections.defaultdict(int)
        with tf.gfile.GFile(dataPath, mode="r") as f:
            for line in f:
                tokens = basic_tokenizer(line)
                words[tokens[0]] += 1
                head[tokens[0]] += 1
                for word in tokens[1:]:
                    if word != tokens[0]:
                        words[word] += 1
        all_words = _START_VOCAB + sorted(words, key=words.get, reverse=True)
        head_vocab = sorted(head, key=head.get, reverse=True)
        with tf.gfile.GFile(vocabFileStem + "_all_head_words.txt", mode="w") as head_file:
            for w in head_vocab:
              head_file.write(w + "\n")
        with tf.gfile.GFile(vocabFileStem + ".vocab", mode="w") as vocab_file:
            if limitedVocab:
                for w in all_words[:vocabLimit]:
                    vocab_file.write(w + "\n")
            else:
                for w in all_words:
                    vocab_file.write(w + "\n")

def getVocabulary(dataDir, limitedVocab = False, vocabLimit = 100000):
    if limitedVocab:
        vocabPathStem = os.path.join(dataDir, "definitions_%d" % vocabLimit)
    else:
        vocabPathStem = os.path.join(dataDir, "definitions_UNLIMITED")

    vocabFilePath = vocabPathStem + ".vocab"
    if tf.gfile.Exists(vocabFilePath):
        rev_vocab = []
        with tf.gfile.GFile(vocabFilePath, mode="r") as f:
            rev_vocab.extend(f.readlines())
        # rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = {x: y for (y, x) in enumerate(rev_vocab)}
        rev_vocab = {y: x for x, y in vocab.items()}
        return vocab, rev_vocab

def getVocabularyIncludingEmbeddings(dataDir, limitedVocab = False, vocabLimit = 100000):
    if limitedVocab:
        vocabPathStem = os.path.join(dataDir, "definitions_%d" % vocabLimit)
    else:
        vocabPathStem = os.path.join(dataDir, "definitions_UNLIMITED")

    vocabFilePath = vocabPathStem + ".vocab"
    if tf.gfile.Exists(vocabFilePath):
        rev_vocab = []
        with tf.gfile.GFile(vocabFilePath, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = {x: y for (y, x) in enumerate(rev_vocab)}

        # rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        with open('embeddings/glove.6B.300d.txt', "r") as input_file:
            for line in input_file:
                vals = line.rstrip().split(' ')
                rev_vocab.append(vals[0].strip())

        rev_vocab_dict = {}
        for word in rev_vocab:
            if word not in rev_vocab_dict:
                rev_vocab_dict[word] = 1
        rev_vocab = []
        for word in rev_vocab_dict.keys():
            rev_vocab.append(word)
        vocabExtended = {x: y for (y, x) in enumerate(rev_vocab)}
        rev_vocab = {y: x for x, y in vocabExtended.items()}
        return vocabExtended, rev_vocab, vocab

def prepareVocabFilesAndGetDataForBaseline(dataDir, limitedVocab = False, vocabLimit = 100000):
    if limitedVocab:
        vocabPathStem = os.path.join(dataDir, "definitions_%d" % vocabLimit)
    else:
        vocabPathStem = os.path.join(dataDir, "definitions_UNLIMITED")
    dataPath = os.path.join(dataDir, "definitions.tok")
    createVocabFile(vocabPathStem, dataPath, limitedVocab, vocabLimit)

def load_pretrained_target_embeddings(embeddingsFile):
    print("Loading pretrained embeddings from %s" % embeddingsFile)
    with open(embeddingsFile, "rb") as input_file:
        pre_embs_dict = pickle.load(input_file)
    return pre_embs_dict

def load_pretrained_target_embeddings_from_file(embeddingsFile):
    print("Loading pretrained embeddings from %s" % embeddingsFile)
    pre_embs_dict = {}
    with open(embeddingsFile, "r") as input_file:
        for line in input_file:
            vals = line.rstrip().split(' ')
            pre_embs_dict[vals[0]] = [float(x) for x in vals[1:]]
    return pre_embs_dict

def sentence_to_token_ids(sentence, vocabulary):
    ps = PorterStemmer()
    tokens = basic_tokenizer(sentence)
    tokenList = []
    count1 = 0
    count2 = 0
    count3 = 0

    for w in tokens:
        if w in vocabulary:
            tokenList.append(vocabulary.get(_digitsExpression.sub(b"0", w), UNK_ID))
            count1 += 1
        elif ps.stem(w) in vocabulary:
            tokenList.append(vocabulary.get(_digitsExpression.sub(b"0", ps.stem(w)), UNK_ID))
            count2 += 1
        else:
            tokenList.append(UNK_ID)
            count3 += 1
    print(sentence + " had these counts " + str(count1) + " " + str(count2) + " "+ str(count3))
    return tokenList
    # return [vocabulary.get(_digitsExpression.sub(b"0", w), UNK_ID) for w in tokens]

def get_embedding_matrix(embedding_dict, vocab, emb_dim):
    emb_matrix = np.zeros([len(vocab), emb_dim])
    x = len(vocab)
    for word,idx in vocab.items():
        if word in embedding_dict:
            emb_matrix[idx] = embedding_dict[word]
        else:
            print("Out of vocab word %s - will put 0 in place" % word)
    return np.asarray(emb_matrix)

def pad_sequence(sequence, max_seq_len):
  padding_required = max_seq_len - len(sequence)
  # Sentence too long, so truncate.
  if padding_required < 0:
    padded = sequence[:max_seq_len]
  # Sentence too short, so pad.
  else:
    padded = sequence + ([PAD_ID] * padding_required)
  return padded

def makeVocabTrainingAndDevFiles(dataPath, targetPath, vocab_dict, maxSeqLen):
    if not (tf.gfile.Exists(targetPath + ".gloss") and tf.gfile.Exists(targetPath + ".head")):
        print("Encoding data into token-ids in %s" % targetPath)
        with tf.gfile.GFile(dataPath, mode="r") as data_file:
            with tf.gfile.GFile(targetPath + ".gloss", mode="w") as glosses_file:
                with tf.gfile.GFile(targetPath + ".head", mode="w") as heads_file:
                    for line in data_file:
                        token_ids = sentence_to_token_ids(line, vocab_dict)
                        heads_file.write(str(token_ids[0]) + "\n")
                        clean_gloss = [w for w in token_ids[1:] if w != token_ids[0]]
                        glosses_ids = pad_sequence(clean_gloss, maxSeqLen)
                        glosses_file.write(" ".join([str(t) for t in glosses_ids]) + "\n")


def prepare_and_save_data(dataDir, train_file, dev_file, limitedVocab, vocabulary_size, limitedTrainData, max_seq_len):
    train_file_path = os.path.join(dataDir, train_file)
    dev_file_path = os.path.join(dataDir, dev_file)

    if limitedVocab:
        vocabPathStem = os.path.join(dataDir, "definitions_%d" % vocabulary_size)
    else:
        vocabPathStem = os.path.join(dataDir, "definitions_UNLIMITED")

    if limitedTrainData:
        train_target_ids = os.path.join(dataDir, "train.definitions.ids%d" % vocabulary_size)
        dev_target_ids = os.path.join(dataDir, "dev.definitions.ids%d" % vocabulary_size)
    else:
        train_target_ids = os.path.join(dataDir, "train.definitions.idsUNLIMITED")
        dev_target_ids = os.path.join(dataDir, "dev.definitions.idsUNLIMITED")

    dataPath = os.path.join(dataDir, "definitions.tok")
    createVocabFile(vocabPathStem, dataPath, limitedVocab, vocabulary_size)
    vocab, rev_vocab = getVocabulary(dataDir, limitedVocab, vocabulary_size)

    makeVocabTrainingAndDevFiles(train_file_path, train_target_ids, vocab, max_seq_len)
    makeVocabTrainingAndDevFiles(dev_file_path, dev_target_ids, vocab, max_seq_len)



