import os
import re
import statistics

def find_all_key_files_path(directory, keyfile_name):
    fn = re.compile(".*"+keyfile_name+".*txt")
    path=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fn.match(file) is not None:
                #print(file)
                path.append(os.path.join(root, file))
    return path

if __name__ == '__main__':
    root_path = "./4test"
    root_path = "./"

    # find test files by recursively opening directory
    result_files = find_all_key_files_path(root_path, "-test")
    print(result_files)
    # open test files
    results = [] # list of dictionary representing the score {param 1 : its val., param 2 : its val, ..., score:val}
    result_format = re.compile("^return: +(.[0-9]+.[0-9]+),.*"
                               + "'torso_mass': +([0-9]+.[0-9]+),.*"
                               + "'ground_friction': +([0-9]+.[0-9]+),.*"
                               + "'joint_damping': +([0-9]+.[0-9]+),.*"
                               + "'armature': +([0-9]+.[0-9]+).*"
                               )

    for result_file in result_files:
        f = open(result_file, "r")
        dic_result = {}
        # parse file into array
        for line in f.readlines():
            m =result_format.search(line)
            if m is not None:
                #print(m.group(0))
                dic_result = {"score": float(m.group(1)),
                              "torso_mass": float(m.group(2)),
                              "ground_friction": float(m.group(3)),
                              "joint_damping": float(m.group(4)),
                              "armature": float(m.group(5)),
                              }
                #print(dic_result)
                results.append(dic_result)
        f.close()
    # calculate average (and std dev)
    print("")
    for target_param in ["torso_mass", "ground_friction", "joint_damping", "armature"]:
        # enumerate
        value_dic = {}
        for result in results:
            if result[target_param] not in value_dic.keys():
                value_dic[result[target_param]]=[]
        #print(target_param + str(value_dic))
        for result in results:
            value_dic[result[target_param]].append(result["score"])
        #print(value_dic)

        # print our result
        for k in value_dic.keys():
            num_samples = len(result_files) * len(value_dic[k])
            break
        print(target_param + "(" + str(num_samples) + "/" + str(len(result_files)) + "), ", end="")
        for key in value_dic.keys():
            print(str(key)+", ", end="")
        print("")
        print("mean, ", end="")
        for key in value_dic.keys():
            print(str(statistics.mean(value_dic[key]))+", ", end="")
        print("")

        print("stdev, ", end="")
        for key in value_dic.keys():
            print(str(statistics.stdev(value_dic[key]))+", ", end="")

        print("")

    # for certain case analysis
    print("")
    scores = []
    for result in results:
        if result["torso_mass"] == 9.0 and result["ground_friction"] == 2.5:
            #print(result)
            scores.append(result["score"])
    mean = statistics.mean(scores)
    std = statistics.stdev(scores)
    print("mean, stdev, num")
    print(str(mean) + ", " + str(std) + ", " + str(len(scores)))
