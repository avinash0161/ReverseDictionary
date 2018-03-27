import tensorflow as tf
from collections import defaultdict

def pickleInfo():
    import pickle
    # 54722 words * 300 dimensions
    word2vec = pickle.load(open('embeddings/GoogleWord2Vec.clean.normed.pkl', 'rb'))
    print len(word2vec.keys())
    # with tf.gfile.GFile("data/definitions/concept_descriptions.tok", mode="r") as data_file:
    #     for line in data_file:
    #         words = line.split()
    #         if words[0] not in word2vec:
    #             print(words[0])

    notInEmbeddingsBig = defaultdict(lambda: 0)
    with tf.gfile.GFile("data/definitions/definitions.tok", mode="r") as data_file:
        for line in data_file:
            words = line.split()
            if words[0] not in word2vec:
                if words[0] not in notInEmbeddingsBig:
                    notInEmbeddingsBig[words[0]] = 1
                else:
                    notInEmbeddingsBig[words[0]] += 1

    print(len(notInEmbeddingsBig.keys()))
    print(sum(notInEmbeddingsBig.values()))

def txtInfo():
    pre_embs_dict = {}
    with open('embeddings/glove.6B.300d.txt', "r") as input_file:
        for line in input_file:
            vals = line.rstrip().split(' ')
            pre_embs_dict[vals[0]] = [float(x) for x in vals[1:]]
            if(len(pre_embs_dict[vals[0]])) != 300:
                print line + " not 300"
    notInEmbeddingsBig = defaultdict(lambda: 0)
    with tf.gfile.GFile("data/definitions/definitions.tok", mode="r") as data_file:
        for line in data_file:
            words = line.split()
            if words[0] not in pre_embs_dict:
                if words[0] not in notInEmbeddingsBig:
                    notInEmbeddingsBig[words[0]] = 1
                else:
                    notInEmbeddingsBig[words[0]] += 1

    print(len(notInEmbeddingsBig.keys()))
    print(sum(notInEmbeddingsBig.values()))

if __name__ == "__main__":
    txtInfo()