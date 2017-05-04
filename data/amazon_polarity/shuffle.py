from random import shuffle

with open("./orig_train.csv") as train:
    lines = train.readlines()

shuffle(lines)

with open("./train.csv", 'w') as new_train:
    new_train.writelines(lines)

