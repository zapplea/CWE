from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf
import numpy as np


class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    def __init__(self, embedding_size, char_embedding_size, context_size, max_vocab_size=8000000, min_occurrences=1,
                 scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05,max_word_len=11):
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_word_len = max_word_len
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None
        self.__char_to_id = None

    def fit_to_corpus(self, corpus):
        print('fit corpus...')
        self.__fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        print('Done!')
        print('build graph...')
        self.__build_graph()
        print('Done!')

    def __fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        # generate word vocabulary
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        print('word vocab size: ',len(self.__word_to_id))
        # generate char vocabulary
        self.__chars = set()
        for word in self.__words:
            if word == '#OTHER#':
                continue
            char_ls = list(word)
            for char in char_ls:
                self.__chars.add(char)
        self.__chars = list(self.__chars)
        # Obedient: the padding char must be 0 because of embedding table when build the model.
        self.__chars.insert(0, '#PAD#')
        self.__char_to_id = {char: i for i, char in enumerate(self.__chars)}
        print('char vocab size: ',len(self.__char_to_id))

        # prepare input matrix
        # convert word_txt in cooccurrence_counts to word id. This is the batch.
        self.__cooccurrence_matrix = {
            (self.__word_to_id[words[0]], self.__word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id}
        self.__cooccurrence_matrix = {}
        for words,count in cooccurrence_counts.items():
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id:
                words_id_pair = (self.__word_to_id[words[0]], self.__word_to_id[words[1]])
                word0_char_ls = []
                if words[0] == '#OTHER#':
                    word0_char_ls.append(self.__char_to_id['#PAD#'])
                else:
                    for char in list(words[0]):
                        word0_char_ls.append(self.__char_to_id[char])
                if len(word0_char_ls)<self.max_word_len:
                    word0_char_ls.extend(np.zeros(shape=(self.max_word_len-len(word0_char_ls),),dtype='int32').tolist())

                word1_char_ls = []
                if words[1] == '#OTHER#':
                    word1_char_ls.append(self.__char_to_id['#PAD#'])
                else:
                    for char in list(words[1]):
                        word1_char_ls.append(self.__char_to_id[char])
                if len(word1_char_ls)<self.max_word_len:
                    word1_char_ls.extend((np.ones(shape=(self.max_word_len-len(word1_char_ls),),dtype='int32')*self.padding_char_id).tolist())
                chars_id_pair = (tuple(word0_char_ls),tuple(word1_char_ls))
                self.__cooccurrence_matrix[(words_id_pair,chars_id_pair)] = count

    def __char_compress(self,chars_embeddings):
        shape = chars_embeddings.get_shape()

    def __padding_char_mask(self,char_ids):
        padding = tf.ones_like(char_ids)*self.padding_char_id
        condition = tf.equal(padding,char_ids)
        # (batch size, max word len)
        mask = tf.where(condition,tf.zeros_like(char_ids),tf.ones_like(char_ids))
        mask = tf.tile(tf.expand_dims(mask,axis=2),multiples=[1,1,self.char_embedding_size])
        return mask

    def __char_seq_len(self,char_ids):
        padding = tf.ones_like(char_ids) * self.padding_char_id
        condition = tf.equal(padding, char_ids)
        # (batch size, max word len)
        temp = tf.where(condition,tf.zeros_like(char_ids),tf.ones_like(char_ids))
        # (batch size, )
        char_seq_len = tf.reduce_sum(temp,axis=1)
        # TODO: some char like #OTHER#, its full padded, so, the length is 0
        condition = tf.equal(char_seq_len,tf.zeros_like(char_seq_len))
        # (batch size, )
        char_seq_len = tf.where(condition,tf.ones_like(char_seq_len),char_seq_len)
        return char_seq_len


    def __char_enhance(self):
        """

        :return: (batch size, char dim)
        """
        # Done: add char input placeholder
        self.__focal_chars_input = tf.placeholder(tf.int32, shape=(None, self.max_word_len), name='focal_chars')
        self.__context_chars_input = tf.placeholder(tf.int32, shape=(None, self.max_word_len), name='context_chars')
        # Fixed: the #PAD# should be [0, 0, 0,....]
        char_embeddings =  tf.Variable(tf.random_uniform([self.char_vocab_size-1, self.char_embedding_size], 1.0, -1.0),
                                       name="char_embeddings")
        padding_char_embedding = tf.Variable(tf.random_uniform([1,self.char_embedding_size],1.0,-1.0),
                                             name="padding_char_embeddings")
        self.__char_embeddings = tf.concat([padding_char_embedding,char_embeddings],axis=0)
        # Done: mask the padded word

        # (batch size, max word len, char dim)
        focal_chars_mask = self.__padding_char_mask(self.__focal_chars_input)
        focal_chars_embeddings = tf.nn.embedding_lookup([self.__focal_chars_input],self.__char_embeddings)*focal_chars_mask
        # (batch size,)
        focal_chars_seq_len = self.__char_seq_len(self.__focal_chars_input)
        # (batch size, char dim)
        focal_chars_denominator = tf.tile(tf.expand_dims(focal_chars_seq_len, axis=1), multiples=[1, self.char_embedding_size])
        # (batch size, char dim)
        focal_chars_embedding = tf.truediv(tf.reduce_sum(focal_chars_embeddings,axis=1),focal_chars_denominator)

        # (batch size, max word len, char dim)
        context_chars_mask = self.__padding_char_mask(self.__context_chars_input)
        context_chars_embeddings = tf.nn.embedding_lookup([self.__context_chars_input],self.__char_embeddings)*context_chars_mask
        # (batch size,)
        context_chars_seq_len = self.__char_seq_len(self.__context_chars_input)
        # (batch size, char dim)
        context_chars_denominator = tf.tile(tf.expand_dims(context_chars_seq_len, axis=1),
                                          multiples=[1, self.char_embedding_size])
        # (batch size, char dim)
        context_chars_embedding = tf.truediv(tf.reduce_sum(context_chars_embeddings,axis=1),context_chars_denominator)

        return focal_chars_embedding,context_chars_embedding

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")
            # FIXED: eliminate influenced of self.batch_size
            # self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
            #                                     name="focal_words")
            # self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
            #                                       name="context_words")
            # self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
            #                                            name="cooccurrence_count")

            self.__focal_input = tf.placeholder(tf.int32, shape=[None],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[None],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[None],
                                                       name="cooccurrence_count")


            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")
            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")
            #(batch size, word dim)
            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)

            # (batch size, char dim)
            focal_chars_embedding, context_chars_embedding = self.__char_enhance()
            enhanced_focal_embedding = tf.concat([focal_embedding,focal_chars_embedding],axis=1)
            enhanced_context_embedding = tf.concat([context_embedding, context_chars_embedding], axis=1)

            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(enhanced_focal_embedding, enhanced_context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        batches = self.__prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.__graph) as session:
            if should_write_summaries:
                print("Writing TensorBoard summaries to {}".format(log_dir))
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    # TODO:feed i\j_chars to the model
                    i_s, j_s,i_chars,j_chars, counts = batch
                    # FIXED:this condition should be eliminated, otherwise several data in the tail will be wasted.
                    # if len(counts) != self.batch_size:
                    #     continue
                    print('feed_dict...')
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__focal_chars_input:i_chars,
                        self.__context_chars_input:j_chars,
                        self.__cooccurrence_count: counts}
                    print('feed Done')
                    session.run([self.__optimizer], feed_dict=feed_dict)
                    if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)
                    total_steps += 1
                if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                    current_embeddings = self.__combined_embeddings.eval()
                    output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                    self.generate_tsne(output_path, embeddings=current_embeddings)
            self.__embeddings = self.__combined_embeddings.eval()
            self.__char_embeddings_mat = self.__char_embeddings.eval()
            if should_write_summaries:
                summary_writer.close()

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1],char_ids[0],char_ids[1], count)
                         for (word_ids,char_ids), count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, i_chars, j_chars, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices,i_chars,j_chars, counts))

    @property
    def char_vocab_size(self):
        return len(self.__char_to_id)

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def padding_char_id(self):
        return self.__char_to_id['#PAD#']

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings
    @property
    def char_embeddings(self):
        return self.__char_embeddings_mat
    @property
    def char_to_id(self):
        return self.__char_to_id
    @property
    def word_to_id(self):
        return self.__word_to_id

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
