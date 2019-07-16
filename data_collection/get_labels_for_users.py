
import collections
import os
import sys

# take a list of user ids and and clustering assignment
# assign labels to each user
# also need the user <--> activity mapping

def get_labels_for_users(user_list_path, activity_list_path, clusters_file_paths, out_dir, addtl_act_ids_path=None):

    out_dir = out_dir.rstrip(os.sep) + os.sep

    # keep a count of the number of users that are given each cluster label
    cluster2user_count = collections.defaultdict(int)

    # load the user list
    users = []
    with open(user_list_path) as user_file:
        users = [user.strip() for user in user_file.readlines()]

    # load the cluster assignment
    # build a dictionary {activity_id:cluster_id}
    aid2cluster = {}
    for clusters_file_path in clusters_file_paths:
        with open(clusters_file_path) as clusters_file:
            for line in clusters_file:
                aid,cluster = line.strip().split()
                aid2cluster[aid] = cluster

    # get all activities for each user in the list
    # for each activity, get the cluster_id that it is in
#    user2aids = collections.defaultdict(list)
    user2clusters = collections.defaultdict(list)
    with open(activity_list_path) as activity_list_file:
        header = activity_list_file.readline()
        for line in activity_list_file.readlines():
            parts = line.split(',',3)
            userid = parts[1]
            aid = parts[0]
#            user2aids[userid].append(aid)
            cluster_id = aid2cluster.get(aid,None)
            # NOTE: there may be users who don't have any activities because the activities were filtered out 
            #   ("I thought" or wildcard match not in correct order). Don't create a file for these users.
            if cluster_id:
                user2clusters[userid].append(cluster_id)
                cluster2user_count[cluster_id] += 1

    # in the output directory, save a file per user that contains the list of cluster_ids
    # the first cluster_id should be the one from the query. This will allow us to do multiple different
    #   setups-- some where we train on all valid clusters and some where we only train on the query cluster.
    #   at test time, we will only evaluate on the prediction of the query activity (which is not in the history)
    for user,clusters in user2clusters.items():
        print user,' '.join(list(set(clusters)))
#    with open(out_dir + user,'w') as out_file:
#            out_file.write(' '.join(list(set(clusters))))

def usage():
    print("usage: python " + os.path.basename(__file__) + " user_list_path activity_list_path cluster_file_path out_dir")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        usage()
    user_list_path = sys.argv[1]
    activity_list_path = sys.argv[2]
    cluster_file_path_1 = sys.argv[3]
    out_dir = sys.argv[4]

    get_labels_for_users(user_list_path, activity_list_path, [cluster_file_path_1], out_dir)
