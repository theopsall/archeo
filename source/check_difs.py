"""
Checking differences between the trackss accross the TRAIN, VAL and TEST dirs
"""
import os

TRAIN = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/TRAIN/"
TEST = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/TEST/"
VAL = "/home/theo/Desktop/ISMAP_2020/bovw/WAVES/VAL/"



TRAIN_SET = set()
VAL_SET = set()
TEST_SET = set()

for label in os.listdir(TRAIN):
    for filename in os.listdir(os.path.join(TRAIN, label)):
        TRAIN_SET.add(filename.split("_")[1])


for label in os.listdir(VAL):
    for filename in os.listdir(os.path.join( VAL, label)):
        VAL_SET.add(filename.split("_")[1])

for label in os.listdir(TEST):
    for filename in os.listdir(os.path.join(TEST, label)):
        TEST_SET.add(filename.split("_")[1])

print("TRAIN",TRAIN_SET)
print("TEST",TEST_SET)
print("VAL",VAL_SET)
print('Same tracks on train, val and test sets', \
    TRAIN_SET.intersection(VAL_SET).intersection(TEST_SET))
print('Same tracks on val and test sets', VAL_SET.intersection(TEST_SET))
