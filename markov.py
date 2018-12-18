# Markov Text Generation

import numpy as np
import pandas as pd 
from collections import defaultdict
import random
import time

# lazily return key, value for k-gram dictionary
def get_k_pairs(text, k):
    for i in range(len(text)-k-1):
        yield (text[i:i+k], text[i+k])

if __name__ == '__main__':
    # read and sort data
    df = pd.read_csv('output.csv', quotechar='|')
    fake = df[df.label == 'FAKE']
    real = df[df.label == 'REAL']
    fakes = fake['text']
    reals = real['text']

    ks = [1, 2, 3, 10, 100] # iterate k's for k-grams
    
    for k in ks: 
        t0 = time.time() # track time
        pairs = []
        fd = defaultdict(list)
        for f in fakes:
            pairs = get_k_pairs(f.split(), k)
            for w1, w2 in pairs:
                fd[' '.join(w1)].append(w2)

        rd = defaultdict(list)
        for r in reals:
            pairs = get_k_pairs(r.split(), k)
            for w1, w2 in pairs:
                rd[' '.join(w1)].append(w2)

        # start with random word
        samples = 10 # test 10 x
        fakeacc = [] # track fake accuracy over samples
        realacc = [] # track real accuracy over samples
        rand_restarts = 0 # how many times a random key had to be generated

        for _ in range(samples): 
            num_words = 300-k+1 # each article is approx 300 words
            farr = "" # fake news generation
            rarr = "" # real news generation
            # first randomly chosen k-grams
            fw0 = random.choice(fd.keys())
            rw0 = random.choice(rd.keys())

            # fake and real generations as list representation
            fchain = fw0.split()
            rchain = rw0.split()

            while num_words > 0:
                try:
                    fchain.append(random.choice(fd[' '.join(fchain[-k:])]))
                    rchain.append(random.choice(rd[' '.join(rchain[-k:])]))
                    num_words -= 1
                except: # case of k-gram input at very end of article, where there's no value in the dict
                    fchain.append(random.choice(fd.keys()))
                    rchain.append(random.choice(rd.keys()))
                    num_words -= k
                    rand_restarts += 1

            farr = ' '.join(fchain)
            rarr = ' '.join(rchain)

            # calculate accuracy difference for fake news generation
            fprop = defaultdict(float) # build dict to store corpus acc
            for article in fakes:
                for word in article:
                    fprop[word] += 1
            norm = sum(fprop.itervalues()) # normalize
            for key, v in fprop.items():
                fprop[key] = float(v)/norm
            len_text = len(farr)
            fsprop = defaultdict(float) # build dict to store generated acc
            for word in farr.split():
                fsprop[word] += 1
            for key, v in fsprop.items(): # normalize
                fsprop[key] = float(v)/len_text 
            loss = 0 # calculate loss
            for key, v in fsprop.items():
                loss += (fprop[key]-v)**2 # use L2 loss
            fakeacc.append(loss)

            # repeat calculations for real news generation
            rprop = defaultdict(float)
            for article in reals:
                for word in article:
                    rprop[word] += 1
            norm = sum(rprop.itervalues())
            for key, v in rprop.items():
                rprop[key] = float(v)/norm
            len_text = len(rarr)
            rsprop = defaultdict(float)
            for word in rarr.split():
                rsprop[word] += 1
            for key, v in rsprop.items():
                rsprop[key] = float(v)/len_text
            loss = 0
            for key, v in rsprop.items():
                loss += (rprop[key]-v)**2
            realacc.append(loss)

        t1 = time.time()

        # print last output 
        print('NEW ITERATION k: ' + str(k))
        print("Code Duration: " + str(t1-t0) + " seconds")
        print("Avg Num of Random Restarts: " + str(float(rand_restarts)/samples))
        print('Fake Generated Text: ' + farr + '\n')
        print('Real Generated Text: ' + rarr)
        print('Fake Loss: ' + str(sum(fakeacc)/float(samples))) # avg accuracy
        print('Real Loss: ' + str(sum(realacc)/float(samples)))
        print('\n\n')