import json
import os
from datetime import datetime, timedelta

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

day_stats = [0]*121

structures = dict()
os.chdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")
greatest_diff = timedelta()
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
        id = int(s[:-5])
        if id > great_id:
          great_id = id

      with open(str(great_id)+".json", 'r') as f:
        file_son = json.load(f)
        son_time_str = file_son["created_at"]
        son_time = datetime.strptime(son_time_str, '%a %b %d %X %z %Y')



      diff = son_time - father_time

      if diff > greatest_diff:
        greatest_diff = diff

    os.chdir("../..")
    os.chdir("../..")


  os.chdir("../..")

print("GREATEST DIFFERENCE")
print(greatest_diff)#120 days, 2:32:06
os.chdir("../..")
os.chdir("../..")

sub_dirs = os.listdir("rumoureval-2019-test-data/twitter-en-test-data/")
os.chdir("rumoureval-2019-test-data/twitter-en-test-data/")
greatest_diff = timedelta()
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
        id = int(s[:-5])
        if id > great_id:
          great_id = id

      with open(str(great_id)+".json", 'r') as f:
        file_son = json.load(f)
        son_time_str = file_son["created_at"]
        son_time = datetime.strptime(son_time_str, '%a %b %d %X %z %Y')

      diff = son_time - father_time

      if diff > greatest_diff:
        greatest_diff = diff

    os.chdir("../..")
    os.chdir("../..")


  os.chdir("../..")

print("GREATEST DIFFERENCE")
print(greatest_diff) #63 days, 19:02:13
os.chdir("../..")
os.chdir("../..")