
#  tee でとった log を整形するプログラム

import os

cwd = os.getcwd()
log_dir =os.path.join(cwd, 'log')


# child log 選択 -----

child_log_list = os.listdir(log_dir)

print("Please chose log below by number -----")

for i, child_log in enumerate(child_log_list):
    print(i, "|", child_log)

selected_idx = int(input(">>> "))

child_log_dir =  os.path.join(log_dir, child_log_list[selected_idx])


# log file 選別 -----

target_list = os.listdir(child_log_dir)

sieve_target_list = []
for i, file_name in enumerate(target_list):
    if "formatted" in file_name:
        pass
    elif "log" in file_name:
        sieve_target_list.append(file_name)


# log file 選択 -----

if len(sieve_target_list) == 1:
    target_file = os.path.join(child_log_dir, sieve_target_list[0])
elif len(sieve_target_list) > 1:
    print("Please chose log below by number -----")

    for i, file_name in enumerate(sieve_target_list):
        print(i, "|", file_name)

    selected_idx = int(input(">>> "))
    target_file = os.path.join(child_log_dir, sieve_target_list[selected_idx])


# 処理開始 -----
file_name, _ = os.path.splitext(os.path.basename(target_file))

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

fmt_log = ''.join(fmt)
print("log の Formatting が終了しました。")

fmt_target = "{}_formatted.log".format(file_name)
save_location = os.path.join(child_log_dir, fmt_target)
with open(save_location, 'w') as f:
    f.write(fmt_log)
    print("整形後のlog を {} に保存しました。".format(save_location))
