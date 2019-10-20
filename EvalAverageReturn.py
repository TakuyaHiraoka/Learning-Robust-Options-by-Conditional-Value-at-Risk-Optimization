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

    # find test files by recursively opening directory
    result_files = find_all_key_files_path(root_path, ".csv")
    print(result_files)
    print()
    # find test files by recursively opening directory
    optfiles = find_all_key_files_path(root_path, "bestpol-cvar.txt")
    print(optfiles)

    # open test files
    results = [] # list of dictionary representing the score {param 1 : its val., param 2 : its val, ..., score:val}

    for result_file in result_files:
        # find bestcvar files in corresponding
        for optfile in optfiles:
            if optfile.split("/")[-2].split("_2opts")[0] == result_file.split("/")[-1].split("_2opts")[0]:
                print("matdched")
                print(optfile)
                print(result_file)
                #
                f_opt = open(optfile,"r")
                polid = int(f_opt.readlines()[0])
                if polid != -1:
                    f_result = open(result_file, "r")
                    CVaR = float(f_result.readlines()[polid*1].split(",")[-1])
                    print(polid)
                    print(CVaR)
                    results.append(CVaR)
                    f_result.close()
                f_opt.close()
                #print(CVaR)
    # calculate average (and std dev)
    print("Epoch number, num data, average, std dev")
    if len(results) > 1:
        print(str(epoch) + ", " + str(len(results)) + ", " + str(statistics.mean(results)) + ", " + str(statistics.stdev(results)))
    else:
        print("No feasible policies")
    #print(str(statistics.mean(results)) + " (" + str(statistics.stdev(results)) + ") &") # for tex style
