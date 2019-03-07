import tf_glove

corpus = [["this", "is", "a", "comment", "."], ["this", "is", "another", "comment", "."]]

model = tf_glove.GloVeModel(embedding_size=300,context_size=3)
model.fit_to_corpus(corpus)
model.train(num_epochs=100)
