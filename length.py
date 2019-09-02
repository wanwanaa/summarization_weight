import numpy as np

filename = 'DATA/seg_data/src-train.txt'
# l = []
# # with open(filename, 'r', encoding='utf-8') as f:
# #     for line in f:
# #         l.append(len(list(line)))
# #
# # mean = sum(l) / len(l)
# # var = np.std(l)
# # result = mean + var
# # result = int(round(result))
# # # print('max:', max(l))      # 141
# # # print('min', min(l))       # 2
# # # print('mean:', mean)       # 31
# # # print('mean+var:', result) # 57
# # print(l)

num = []
n = 0
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        n += 1
        if line == '\n':
            num.append(n-1)
            n = 0
# print(num[0])
# print(num[1])
# print(num)
a = 0
for i in num:
    if i > 4:
        a += 1
print(a)
print(a/len(num))
mean = sum(num) / len(num)
var = np.std(num)
result = mean + var
result = int(round(result))
print('max:', max(num))      # 141
print('min', min(num))       # 2
print('mean:', mean)       # 31
print('mean+var:', result) # 57