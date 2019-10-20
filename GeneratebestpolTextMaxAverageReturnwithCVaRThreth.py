import os
import re
import statistics

def find_all_key_files_path(directory, keyfile_name):
    fn = re.compile(".*"+keyfile_name + ".*")
    path=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fn.match(file) is not None:
                #print(file)
                path.append(os.path.join(root, file))
    return path

if __name__ == '__main__':
    # experiment setup params.
    root_path = "./"
    epoch = 0
    beta = 0.0 

    # find result files by recursively opening directories. 
    result_files = find_all_key_files_path(root_path, ".csv")
    print(result_files)
    print()

    # open files
    results = []

    for result_file in result_files:
        # find the best one
        f_learning_log = open(result_file, "r")

        i = 0
        bestpol_id = -1
        bestpol_score = -99999.0
        scores=[]
        for line in f_learning_log.readlines():
            if (i % 5) == 0 and (i!=0):
                scores.append(float(line.split(",")[-1]))
                id = int(line.split(",")[0])
                print(scores)
                print(id)
                if bestpol_score < statistics.mean(scores) and (float(line.split(",")[-2]) > beta):
                    bestpol_id = id
                    bestpol_score = statistics.mean(scores)
                scores = []
            elif i != 0:
                scores.append(float(line.split(",")[-1]))
            i += 1
        print(bestpol_id)
        print(bestpol_score)

        #
        dirname = result_file.split("_results")[0] + "saves/"
        print(dirname)
        f_saves = open(dirname+"bestpol-cvar.txt", "w")
        f_saves.write(str(bestpol_id))
        f_saves.close()
        f_learning_log.close()