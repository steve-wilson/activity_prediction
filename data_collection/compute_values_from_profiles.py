
import os
import sys

import numpy as np
from gensim.models import FastText
from scipy.spatial.distance import cosine

# average embeddings of all words in each category to get 50 vectors
# for each profile in user_profiles, average embeddings of all words in the profile
# get 50 consines (1 for each vector), round, and write to file (associate with the user id)

def vector_average(vectors):
    return np.average(vectors,0)

def get_lexicon_vectors(embeddings, lexicon_path):

    lexicon_vectors = {}

    with open(lexicon_path) as lexicon:
        for entry in lexicon:
            item_id, category, words = entry.strip('\n').split(' ',2)
            embedding = vector_average([embeddings.wv[word.lower()] for word in words.split()])
            lexicon_vectors[category] = embedding

    return lexicon_vectors

def load_embeddings(embeddings_path):
    return FastText.load_fasttext_format(embeddings_path)

def sim(v1,v2):
#    print("Check:")
#    print(v1)
#    print(v2)
    return 1 - cosine(np.array(v1,dtype=np.float),np.array(v2,dtype=np.float))

def get_lexicon_scores(profiles_path, lexicon_path, embeddings_path):

    print("loading embeddings...")
    sys.stdout.flush()
    emb = load_embeddings(embeddings_path)
    print("done.\ncomputing lexicon vectors...")
    sys.stdout.flush()
    lexicon = get_lexicon_vectors(emb, lexicon_path)
    print("done.\ncomputing all lexicon scores...")
    sys.stdout.flush()

    with open(profiles_path) as profiles_file:
        for line in profiles_file:
            if line.count('\t') > 1:
                userid, location, profile = line.strip('\n').split('\t',2)
                profile_vectors = []
                for word in profile.split():
                    try:
                        profile_vectors.append(emb.wv[word.lower()])
                    except:
                        sys.stderr.write("No vectors for " + word + '\n')
                if profile_vectors:
                    profile_embedding = vector_average(profile_vectors)
                    scores = []
                    for category, lex_emb in sorted(lexicon.items()):
                        scores.append(sim(profile_embedding, lex_emb))
                    print(userid,' '.join([str(round(s,5)) for s in scores]))
#                else:
#                    sys.stderr.write("No matching words in profile for user: " + str(userid) + '\n')
            else:
                sys.stderr.write("no profile found for this line: " + line)

if __name__ == "__main__":
    get_lexicon_scores(sys.argv[1], sys.argv[2], sys.argv[3]) # profiles.out lexicon.txt fasttext_embeddings_path
