
# Added by Steve Wilson
# Summer 2018

import csv
import os
import sys
import re

import nltk
from pattern.en import conjugate

def convert(act):

    act = re.sub(r"\bmy\b","one's",act)
    act = re.sub(r"\bmyself\b","oneself",act)
    act = re.sub(r"\bme\b","one",act)
    act = re.sub(r"\bmine\b","one's",act)
    act = re.sub(r"\bwe\b","one",act)

    act = act.strip('.')
    words = nltk.word_tokenize(act)
    
    if words[0].lower() != "i":
        words = ["i"] + words
#    print(words)
    vb = words[1]
    present_tense_verb = conjugate(vb, tense="present",
                                    person=1, number="singular")
    words[1] = present_tense_verb
    words = [word for word in words if word != "i"]

    return " ".join(words)

if __name__ == "__main__":
    indir,outdir = sys.argv[1],sys.argv[2]
    for f in os.listdir(indir):
        path = indir.rstrip(os.sep) + os.sep + f
        outpath = outdir.rstrip(os.sep) + os.sep + f
        with open(path) as thefile:
            with open(outpath,'w') as outfile:
                for line in thefile:
                    line = line.strip()
                    if line.lower() != "i" and line != "":
                        try:
                            converted = convert(line)
                            outfile.write(converted + '\n')
                        except:
                            print("Error processing:",line)
