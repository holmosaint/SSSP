result_file = open("./result.NY.txt", "r")
gt_file = open("./script/ch9-1.1/results/USA-road-d.ss.res", "r")

result = []
for line in result_file.readlines():
    if line[0:2] != "ss":
        continue
    line = line.split()
    num = int(line[1])
    result.append(num)

gt = []
target = False
for line in gt_file.readlines():
    line = line[:-1].split()
    if line[0] == 'f':
        if line[-1].endswith("NY.ss"):
            target = True
        else:
            target = False
    if (not target) or line[0] != "d":
        continue
    num = int(line[1])
    gt.append(num)

print("Result length: {}".format(len(result)))
print("gt length: {}".format(len(gt)))
assert len(gt) == len(result)
for i in range(len(gt)):
    assert gt[i] == result[i]

result_file.close()
gt_file.close()
print("Test True!")
