from collections import defaultdict
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
import jieba

def cntext_to_word_sequence(text, filters, min_len=1, char_level=False):
    """
    Transform each text in Chinese texts to a sequences of Chinese words
    """    
    #translate_dict = dict((c, ' ') for c in filters)
    #translate_map = str.maketrans(translate_dict)
    #text = text.translate(translate_map)
    text = text.strip()
    if isinstance(filters, set):
        filters_set = filters
    elif isinstance(filters, list):
        filters_set = set(filters)
    elif isinstance(filters, str):
        filters_lst = ' '.join(filters).split(' ')
        filters_set  = set(filters_lst)
    else:
        filters_set = set(' '.join('！"#$%&()（）*+，。；,-./:;《<=>》?@【[\\]】^_`——{|}~\t\n').split() )

    if char_level:
        words = text.split(' ')
    else:
        words = jieba.cut(text)

    # using set() for faster search
    #  & (len(w)>=min_len)
    seq = [w for w in words if (w not in filters_set) & (len(w)>=min_len)]
    return seq

    
def cntexts_to_sequences(texts, num_words, oov_token, document_count, filters, word_index ):
    """Transforms each text in Chinese texts to a sequence of integers.
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    # Arguments
        texts: A list of texts (strings).
    # Returns
        A list of sequences.
    """
    return list(cntexts_to_sequences_generator(texts, num_words, oov_token, document_count, filters, word_index))    


def cntexts_to_sequences_generator(texts, num_words, oov_token, document_count, filters, word_index, char_level=False ):
    """Transforms each Chinese text in `texts` to a sequence of integers.
    Each item in texts can also be a list,
    in which case we assume each item of that list to be a token.
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    # Arguments
        texts: A list of texts (strings).
    # Yields
        Yields individual sequences.
    """
    num_words = num_words
    oov_token_index = word_index.get(oov_token)
    for text in texts:
        document_count += 1
        #print(text)

        if char_level or isinstance(text, list):
            longtext = ' '.join(text)           
            text = longtext
        seq = cntext_to_word_sequence(text, filters, char_level)
        #print(seq, '\n')
                
        vect = []
        temp = []
        for w in seq:
            i = word_index.get(w)
            temp.append(i)
            if i is not None:
                if num_words and i >= num_words:
                    if oov_token_index is not None:
                        vect.append(oov_token_index)
                else:
                    vect.append(i)
            elif oov_token is not None:
                vect.append(oov_token_index)
        #print(temp)
        #print('-------------')
        yield vect       

        
class cnTokenizer(Tokenizer):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...
    # Arguments
        num_words: the maximum number of words to keep, based
                   on word frequency. Only the most common `num_words-1` words will
                   be kept.
        filters: a string where each element is a character that will be
                 filtered from the texts. The default is all punctuation, plus
                 tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: str. Separator for word splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
                   replace out-of-vocabulary words during text_to_sequence calls
    By default, all punctuation is removed, turning the texts into
    space-separated sequences of words
    (words maybe include the `'` character). These sequences are then
    split into lists of tokens. They will then be indexed or vectorized.
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, num_words=None,
                 filters='!！"#$%&()（）*+，。；,-./:;《<=>》?@【[\\]】^_`——{|}~\t\n',
                 lower=False,
                 min_len=1,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 document_count=0,
                 **kwargs):
        # Legacy support
        if 'nb_words' in kwargs:
            warnings.warn('The `nb_words` argument in `Tokenizer` '
                          'has been renamed `num_words`.')
            num_words = kwargs.pop('nb_words')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.filters = filters
        self.min_len = min_len
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)
        self.word_index = dict()
        self.index_word = dict()
            
    def fit_on_cntexts(self, texts):
        '''
        假设输入是一组中文列表
        '''
        for text in texts:
            self.document_count += 1

            if self.char_level or isinstance(text, list):
                longtext = ' '.join(text)           
                text = longtext
            seq = cntext_to_word_sequence(text, filters=self.filters, min_len=self.min_len, char_level=self.char_level)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
            # In how many documents each word occurs
               self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists        
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
                list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
        )

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    
    def cntexts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.
        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Returns
            A list of sequences.
        """
        return list(self.cntexts_to_sequences_generator(texts))    

    def cntexts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.
        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.
        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            self.document_count += 1

            if self.char_level or isinstance(text, list):
                longtext = ' '.join(text)           
                text = longtext
            seq = cntext_to_word_sequence(text, filters = self.filters, min_len=self.min_len, char_level=self.char_level)
                    
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect         

    def cntexts_to_matrix(self, texts, mode='binary'):
        """Convert a list of texts to a Numpy matrix.

        # Arguments
            texts: list of strings.
            mode: one of "binary", "count", "tfidf", "freq".

        # Returns
            A Numpy matrix.
        """
        sequences = self.cntexts_to_sequences(texts)
        return self.cnsequences_to_matrix(sequences, mode=mode)

    def cnsequences_to_matrix(self, sequences, mode='binary'):
        """Converts a list of sequences into a Numpy matrix.

        # Arguments
            sequences: list of sequences
                (a sequence is a list of integer word indices).
            mode: one of "binary", "count", "tfidf", "freq"

        # Returns
            A Numpy matrix.

        # Raises
            ValueError: In case of invalid `mode` argument,
                or if the Tokenizer requires to be fit to sample data.
        """
        if not self.num_words:
            if self.word_index:
                num_words = len(self.word_index) + 1
            else:
                raise ValueError('Specify a dimension (`num_words` argument), '
                                 'or fit on some text data first.')
        else:
            num_words = self.num_words

        if mode == 'tfidf' and not self.document_count:
            raise ValueError('Fit the Tokenizer on some data '
                             'before using tfidf mode.')

        x = np.zeros((len(sequences), num_words))
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            counts = defaultdict(int)
            for j in seq:
                if j >= num_words:
                    continue
                counts[j] += 1
            for j, c in list(counts.items()):
                if mode == 'count':
                    x[i][j] = c
                elif mode == 'freq':
                    x[i][j] = c / len(seq)
                elif mode == 'binary':
                    x[i][j] = 1
                elif mode == 'tfidf':
                    # Use weighting scheme 2 in
                    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.document_count /
                                 (1 + self.index_docs.get(j, 0)))
                    x[i][j] = tf * idf
                else:
                    raise ValueError('Unknown vectorization mode:', mode)
        return x
			
