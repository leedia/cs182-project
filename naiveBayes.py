import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from math import log

class TextClassifier:

    def train(self, text):
        # Loop through file and fill dict and nrated
        self.dict = {}
        self.nrated = [0] * 5
        count = 0
        rating = int(line[0])
        for word in text:
            if word not in self.dict:
                self.dict[word] = count
                count += 1
        self.nrated[rating] += 1

        # Fill counts
        self.counts = [[0] * len(self.dict) for _ in range(5)]
        f = open(infile)
        for line in f:
            rating = int(line[0])
            for word in line[2:].split():
                self.counts[rating][self.dict[word]] += 1

    def fit(self, alpha=1):
        self.F = [[0] * len(self.dict) for _ in range(5)]
        for rating in range(5):
            for word in range(len(self.dict)):
                p = float(alpha + self.counts[rating][word])/(sum(self.counts[rating]) + alpha * len(self.counts[rating]))
                if p <= 0:
                    self.F[rating][word] = 0
                else:
                    self.F[rating][word] = -log(p)

    def test(self, infile):
        """
        Test time! The infile has the same format as it did before. For each review,
        predict the rating. Ignore words that don't appear in your dictionary.
        Are there any factors that won't affect your prediction?
        You'll report both the list of predicted ratings in order and the accuracy.
        """
        file = open(infile)
        accurate, total = 0.,0.
        pred = []
        for line in file:
            rating = int(line[0])
            ratings = [0 for _ in range(5)]
            for word in line[2:].split():
                if word in self.dict:
                    for i in range(5):
                        ratings[i] += self.F[i][self.dict[word]]
            p = ratings.index(min(ratings))
            pred.append(p)
            if p == rating:
                accurate += 1
            total += 1
        return (pred, accurate/total)

    def improve_alpha(self, infile):
        alpha, acc, prev_acc = 0., 0., -0.01
        while prev_acc < acc and alpha <= 1:
            alpha += 0.01
            prev_acc = acc
            self.q5(alpha=alpha)
            acc = self.q6(infile)[1]
        return alpha

if __name__ == '__main__':
    c = TextClassifier()
    print "Processing training set..."
    c.train('mini.train')
    print len(c.dict), "words in dictionary"
    print "Fitting model..."
    c.fit()
    print "Accuracy on validation set:", c.test('mini.valid')[1]
    print "Good alpha:", c.improve_alpha('mini.valid')


#started cleaning algorithm
for filename in os.listdir("files"):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
