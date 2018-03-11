import tensorflow as tf
from collections import defaultdict

if __name__ == "__main__":
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
