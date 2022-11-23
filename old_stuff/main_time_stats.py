import json
import os
from datetime import datetime, timedelta

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

day_stats = [0]*121
hour_stats = [0]*24
min_stats = [0]*60

structures = dict()
os.chdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d)+"/source-tweet")
    with open(str(d)+".json", 'r') as f:
      file = json.load(f)
      father_time_str = file["created_at"]
      father_time = datetime.strptime(father_time_str, '%a %b %d %X %z %Y')

    os.chdir("../..")

    os.chdir("replies")
    sons = os.listdir("..")
    if(len(sons)>0):
      great_id = 0
      for s in sons:
        with open(s, 'r') as f:
          file_son = json.load(f)
          son_time_str = file_son["created_at"]
          son_time = datetime.strptime(son_time_str, '%a %b %d %X %z %Y')
          diff = son_time - father_time
          day_stats[int(diff.days)] = day_stats[int(diff.days)] + 1
          if diff.days==0:
            hour_stats[int(diff.seconds/3600)] = hour_stats[int(diff.seconds/3600)] + 1
            if int(diff.seconds/3600) == 0:
              min_stats[int(diff.seconds / 60)] = min_stats[int(diff.seconds / 60)] + 1

    os.chdir("../..")
    os.chdir("../..")


  os.chdir("../..")


os.chdir("../..")
os.chdir("../..")

day_stats_test = [0]*121
hour_stats_test = [0]*24
min_stats_test = [0]*60
sub_dirs = os.listdir("rumoureval-2019-test-data/twitter-en-test-data/")
os.chdir("rumoureval-2019-test-data/twitter-en-test-data/")

for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d)+"/source-tweet")
    with open(str(d)+".json", 'r') as f:
      file = json.load(f)
      father_time_str = file["created_at"]
      father_time = datetime.strptime(father_time_str, '%a %b %d %X %z %Y')

    os.chdir("../..")

    os.chdir("replies")
    sons = os.listdir("..")
    if(len(sons)>0):
      great_id = 0
      for s in sons:
        with open(s, 'r') as f:
          file_son = json.load(f)
          son_time_str = file_son["created_at"]
          son_time = datetime.strptime(son_time_str, '%a %b %d %X %z %Y')
          diff = son_time - father_time
          day_stats_test[int(diff.days)] = day_stats_test[int(diff.days)] + 1
          if diff.days==0:
            hour_stats_test[int(diff.seconds/3600)] = hour_stats_test[int(diff.seconds/3600)] + 1
            if int(diff.seconds/3600) == 0:
              min_stats_test[int(diff.seconds / 60)] = min_stats_test[int(diff.seconds / 60)] + 1

    os.chdir("../..")
    os.chdir("../..")


  os.chdir("../..")

os.chdir("../..")
os.chdir("../..")
print("STATS")
print(day_stats)
print(hour_stats)
print(min_stats)
print("TEST_STATS")
print(day_stats_test)
print(hour_stats_test)
print(min_stats_test)
'''
STATS
[5081, 73, 44, 15, 6, 5, 10, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
[3659, 564, 196, 90, 70, 86, 69, 35, 35, 43, 44, 29, 24, 16, 11, 13, 10, 9, 9, 26, 16, 13, 10, 4]
[143, 272, 238, 207, 208, 200, 151, 160, 105, 109, 84, 101, 77, 87, 69, 63, 62, 62, 68, 55, 36, 34, 41, 32, 56, 42, 34, 42, 45, 48, 32, 35, 33, 30, 25, 34, 28, 29, 24, 29, 26, 17, 30, 18, 20, 19, 24, 20, 27, 24, 20, 30, 25, 25, 18, 21, 11, 21, 16, 17]
TEST_STATS
[910, 45, 19, 7, 5, 6, 0, 4, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[429, 118, 54, 37, 35, 21, 19, 13, 22, 13, 15, 13, 21, 18, 14, 16, 8, 2, 9, 7, 5, 8, 6, 7]
[8, 23, 15, 24, 29, 17, 15, 12, 14, 8, 15, 8, 13, 8, 8, 9, 6, 9, 5, 6, 11, 3, 6, 8, 3, 1, 4, 3, 3, 6, 4, 4, 6, 3, 2, 3, 11, 2, 3, 4, 3, 7, 4, 7, 3, 3, 6, 5, 3, 3, 8, 3, 1, 3, 9, 2, 4, 6, 3, 4]
'''