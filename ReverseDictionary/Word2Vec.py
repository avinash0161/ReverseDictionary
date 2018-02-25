

if __name__ == "__main__":
    import pickle
    # 54722 words * 300 dimensions
    word2vec = pickle.load(open('embeddings/GoogleWord2Vec.clean.normed.pkl', 'rb'))
    print len(word2vec.keys())
    for value in word2vec.values():
        print len(value)
