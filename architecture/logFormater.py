
#  tee でとった log を整形するプログラム

import os

cwd = os.getcwd()

log_dir = os.path.join(cwd, "log")
child_log_list = os.listdir(log_dir)

print("Please chose log below -----")
for i, log_name in enumerate(child_log_list):
    print("[", i, "]", log_name)

selected_log = int(input(">>> "))

child_log_dir = os.path.join(log_dir, child_log_list[selected_log])
target_file = os.path.join(child_log_dir, "20191010.log")  # TODO: ここは選択(対話的)にする
file_name, _ = os.path.splitext(os.path.basename(target_file))
#print(file_name)


with open(target_file) as f:
    lines = f.readlines()
    print("この log file は ", len(lines), " 行です")

    
for i in range(len(lines)):
    if "[steps / epoch]" in lines[i]:
        sep = i
print(sep, " 行目以前は全て保存の対象です。")

print("log の Formatting を開始..")

fmt = []
for i in range(len(lines)):
    if i < sep or i == sep or i == sep+1:
        fmt.append(lines[i])
    else:
        if "Epoch" in lines[i]:
            fmt.append(lines[i])
        elif "val_accuracy" in lines[i]:
            fmt.append(lines[i])

fmt.append(lines[len(lines)-1])
#print(fmt)

fmt_log = '\n'.join(fmt)
print("log の Formatting が終了しました。")

fmt_target = os.path.join(child_log_dir, "{}_formatted.log".format(file_name))
with open(fmt_target, 'w') as f:
    f.write(fmt_log)
    print("整形後のlog を {} 書き込みに成功しました。".format(fmt_target))
