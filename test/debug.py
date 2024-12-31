import json

file_path = '/home/SENSETIME/zengwang/myprojects/task_define_service/data/shot2story/134k_full_train.json'

with open(file_path, 'r') as f:
    data = json.load(f)

item = data[0]

t = 0