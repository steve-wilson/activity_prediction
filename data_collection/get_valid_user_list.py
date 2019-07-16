
import os
import sys

# Find set of users who meet the following criteria:
# -- profile not blank
# -- profile contains at least 1 word that is in glove-twitter vocabulary
# -- has at least 4 addtl acts (i.e. 5 total)
# -- has at least 10 non-act tweets (i.e. #tweets - #acts > 10)

def create_valid_user_list(profiles_path, tweets_path, acts_path, glove_path, activity_list_path):

    # load GloVe vocab into a set object
    vocab = set([])
    with open(glove_path) as glove_file:
        for line in glove_file:
            word, _ = line.split(" ",1)
            vocab.add(word)
    # load activity list userids into a set object
    al_userids = set([])
    with open(activity_list_path) as activity_list_file:
        al_header = activity_list_file.readline()
        for al_line in activity_list_file:
            al_parts = al_line.split(',')
            al_userid = al_parts[1]
            al_userids.add(al_userid.strip())

    # for each user in profiles file
    with open(profiles_path) as profiles_file:
        for profile_line in profiles_file:
            if '\t' not in profile_line:
                sys.stderr.write('no profile: ' + profile_line + '\n')
                continue
            parts = profile_line.split('\t')
            user = parts[0]
            profile = parts[-1]
            profile_words = [word for word in profile.split() if word.lower().strip("""#()[]{}-=~.,?!:;"'""") in vocab]

            # important: make sure it is in the activity list! otherwise we don't have a cluster for them
            if user not in al_userids:
                sys.stderr.write("no valid query activities: " + profile + '\n')
                continue

            # make sure num glove words >= 1
            if not profile_words:
                sys.stderr.write("no glove words: " + profile + '\n')
                continue

            # count num tweets (num lines in tweets file)
            tweet_file_path = tweets_path.rstrip(os.sep) + os.sep + user + '.tweets'
            if os.path.exists(tweet_file_path):
                with open(tweet_file_path) as tweets_file:
                    num_tweets = len(tweets_file.readlines())
                    # check that num tweets >= 25
                    if num_tweets < 25:
                        sys.stderr.write("not enough tweets: " + str(num_tweets) + '\n')
                        continue
            else:
                sys.stderr.write("no tweets file: " + tweet_file_path + '\n')
                continue

            # count num acts (num lines in acts file)
            acts_file_path = acts_path.rstrip(os.sep) + os.sep + user + '.acts'
            if os.path.exists(acts_file_path):
                with open(acts_file_path) as acts_file:
                    num_acts = len(acts_file.readlines())
                    # check that num acts >= 4
                    if num_acts < 4:
                        sys.stderr.write("not enough acts: " + str(num_acts) + '\n')
                        continue
            else:
                sys.stderr.write("no acts file: " + acts_file_path + '\n')
                continue

            # if user is valid, print their id and continue the loop
            print user

if __name__ == "__main__":

    create_valid_user_list(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]) #profiles_path tweets_path acts_path glove_path activity_list_path
