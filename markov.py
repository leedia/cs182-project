# Markov 

import numpy as np
import pandas as pd 
from collections import defaultdict
import random

def get_pairs(text):
    for i in range(len(text)-1):
        yield (text[i], text[i+1])

if __name__ == '__main__':
    df = pd.read_csv('out6.csv', quotechar='|')
    fake = df[df.label == 'FAKE']
    real = df[df.label == 'REAL']
    fakes = fake['text']
    reals = real['text']
    
    pairs = []
    fd = defaultdict(list)
    for f in fakes:
        pairs = get_pairs(f.split())
        for w1, w2 in pairs:
            fd[w1].append(w2)

    rd = defaultdict(list)
    for r in reals:
        pairs = get_pairs(r.split())
        for w1, w2 in pairs:
            rd[w1].append(w2)

    # start with common word, same number of words
    #w0 = "trump" 

    # start with random word
    min_words = 300
    farr = ""
    rarr = ""
    samples = 10

    for _ in range(samples): 
        fw0 = random.choice(fd.keys())
        rw0 = random.choice(rd.keys())

        fchain = [fw0]
        rchain = [rw0]

        for i in range(min_words-1):
            fchain.append(random.choice(fd[fchain[-1]]))
            rchain.append(random.choice(rd[rchain[-1]]))

        while fchain[-1] != '.':
            fchain.append(random.choice(fd[fchain[-1]]))

        while rchain[-1] != '.':
            rchain.append(random.choice(rd[rchain[-1]]))

        farr += " " + ' '.join(fchain)
        rarr += " " + ' '.join(rchain)

    print('Fake: ' + farr)
    print('Real: ' + rarr)

    # calculate difference 
    fprop = defaultdict(float)
    for article in fakes:
        for word in article:
            fprop[word] += 1
    # normalize
    norm = sum(fprop.itervalues())
    for k, v in fprop.items():
        fprop[k] = float(v)/norm

    loss = 0
    len_text = len(farr)
    fsprop = defaultdict(float)
    for word in farr[i]:
        fsprop[word] += 1
    #normalize
    for k, v in fsprop.items():
        fsprop[k] = float(v)/len_text

    #calc
    loss = 0
    for k, v in fsprop.items():
        loss += (fprop[k]-v)**2
    print('Fake loss: ' + str(loss))

    #repeat for real
    rprop = defaultdict(float)
    for article in reals:
        for word in article:
            rprop[word] += 1
    # normalize
    norm = sum(rprop.itervalues())
    for k, v in rprop.items():
        rprop[k] = float(v)/norm

    loss = 0
    len_text = len(rarr)
    rsprop = defaultdict(float)
    for word in rarr[i]:
        rsprop[word] += 1
    #normalize
    for k, v in rsprop.items():
        rsprop[k] = float(v)/len_text

    #calc
    loss = 0
    for k, v in rsprop.items():
        loss += (rprop[k]-v)**2
    print('Real loss: ' + str(loss))
