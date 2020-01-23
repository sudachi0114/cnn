
import os
import re
import time
import pprint
import argparse

def formatter(SOURCE_FILE):

    pdir, fname = os.path.split(SOURCE_FILE)
    bodyname, ext = os.path.splitext(fname)

    f = open(SOURCE_FILE)
    l = f.readlines()

    # 空白文字 0回以上 の後, 数字から始まる行を検索する正規表現
    regex = re.compile(r"^\s*[0-9]")

    matched_row = []
    result = []
    for i, line in enumerate(l):
        mo = regex.search(line)

        if mo is not None:
            # 間違って引っかかってしまうものを除外
            if "epoch" in l[i]:
                pass
            elif "steps" in l[i]:
                pass
            else:
                matched_row.append(i)
                target = l[i]

                tmp = target.split("/", 1)
                # print(tmp)

                if len(tmp) < 2:
                    pass
                else:
                    left_num, rest = tmp[0], tmp[1]
                    # print(tmp[1].split("[", 1))
                    right_num, rest = tmp[1].split("[", 1)

                left_num = int(left_num)
                right_num = int(right_num)

                if left_num == right_num:
                    result.append(l[i])
        else:
            result.append(l[i])


    result_file = os.path.join(pdir, "{}.formated.log".format(bodyname))
    rf = open(result_file, "w")
    rf.writelines(result)
    rf.close()

    f.close()

    print("Done.")
    print("  export formated file: \n", result_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is program for Log Formatting.")

    parser.add_argument("-a", "--all", action="store_true",
                        help="process all file in current directory.")

    args = parser.parse_args()

    print("Log Fomater started.")
    if args.all:
        print("  process all log file in current directory.")

        start = time.time()
        cwd = os.getcwd()
        cwd_list = os.listdir(cwd)

        for f in cwd_list:
            # log file であり、かつ、すでに処理されているものではない log file
            if ("log" in f) and ("format" not in f):
                formatter(f)

        elapsed_time = time.time() - start
        print("All process has done, elapsed_time: {} [sec]".format(elapsed_time))
    else:
        print("  please enter log file path below:")
        SOURCE_FILE = input(">>> ")

        start = time.time()
        formatter(SOURCE_FILE)
        elapsed_time = time.time() - start
        print("All process has done, elapsed_time: {} [sec]".format(elapsed_time))
