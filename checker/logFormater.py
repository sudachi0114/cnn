
#  tee でとった log を整形するプログラム

import os

cwd = os.getcwd()
#log_dir = 
target_file = os.path.join(log_dir, "20191006.log")  # TODO: ここは選択(対話的)にする
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

fmt_target = "{}_formatted.log".format(file_name)
with open(fmt_target, 'w') as f:
    f.write(fmt_log)
    print("整形後のlog を {} 書き込みに成功しました。".format(fmt_target))
