with open('correspondences.txt', 'r') as f:
    correspondences = [eval(l[:-1]) for l in f.readlines()]

print(correspondences)


with open('mapper.txt', 'r') as f:
    mapper = eval(f.read())

print(mapper)

buf = []

for cor in correspondences:
    buf.append(f'{cor[0]},{cor[1]},{mapper[cor[0]][cor[1]][cor[2]]},{cor[3]},{cor[4]},{mapper[cor[3]][cor[4]][cor[5]]}')

with open('correspondences_mapped.txt', 'w') as f:
    print('\n'.join(buf), file=f)

with open('correspondences_mapped.txt', 'r') as f:
    pred = [l for l in f.readlines()]
    print(pred)