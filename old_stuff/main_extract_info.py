import json
import os
from datetime import datetime, timedelta

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

posts = dict()
users = dict()
os.chdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d)+"/source-tweet")
    with open(str(d)+".json", 'r') as f:
      post_info = dict()
      file = json.load(f)
      post_info["text"] = file["text"]
      post_info["user_id"] = file["user"]["id"]
      post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
      post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %X %z %Y')

      posts[d] = post_info

      if file["user"]["id"] not in users.keys():
        user_info = dict()
        user_info["description"] = file["user"]["description"]
        user_info["description"] = file["user"]["name"]
        user_info["description"] = file["user"]["screen_name"]
        user_info["description"] = file["user"]["created_at"]
        users[file["user"]["id"]] = user_info
    os.chdir("../..")

    os.chdir("replies")
    sons = os.listdir("..")
    if(len(sons)>0):
      for s in sons:
        with open(s, 'r') as f:
          post_info = dict()
          file = json.load(f)
          post_info["text"] = file["text"]

          post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
          post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %X %z %Y')
          post_info["user_id"] = file["user"]["id"]
          posts[int(s[:-5])] = post_info

          if file["user"]["id"] not in users.keys():
            user_info = dict()
            user_info["description"] = file["user"]["description"]
            user_info["name"] = file["user"]["name"]
            user_info["screen_name"] = file["user"]["screen_name"]
            user_info["created_at"] = datetime.strptime(file["user"]["created_at"], '%a %b %d %X %z %Y')
            users[file["user"]["id"]] = user_info

    os.chdir("../..")
    os.chdir("../..")



  os.chdir("../..")


os.chdir("../..")
os.chdir("../..")

posts_test = dict()
users_test = dict()

sub_dirs = os.listdir("rumoureval-2019-test-data/twitter-en-test-data/")
os.chdir("rumoureval-2019-test-data/twitter-en-test-data/")

for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d) + "/source-tweet")
    with open(str(d) + ".json", 'r') as f:
      post_info = dict()
      file = json.load(f)
      post_info["text"] = file["text"]
      post_info["user_id"] = file["user"]["id"]
      post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
      post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %X %z %Y')

      posts_test[d] = post_info

      if file["user"]["id"] not in users.keys():
        user_info = dict()
        user_info["description"] = file["user"]["description"]
        user_info["description"] = file["user"]["name"]
        user_info["description"] = file["user"]["screen_name"]
        user_info["description"] = file["user"]["created_at"]
        users_test[file["user"]["id"]] = user_info
    os.chdir("../..")

    os.chdir("replies")
    sons = os.listdir("..")
    if (len(sons) > 0):
      for s in sons:
        with open(s, 'r') as f:
          post_info = dict()
          file = json.load(f)
          post_info["text"] = file["text"]

          post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
          post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %X %z %Y')
          post_info["user_id"] = file["user"]["id"]
          posts_test[int(s[:-5])] = post_info

          if file["user"]["id"] not in users.keys():
            user_info = dict()
            user_info["description"] = file["user"]["description"]
            user_info["name"] = file["user"]["name"]
            user_info["screen_name"] = file["user"]["screen_name"]
            user_info["created_at"] = datetime.strptime(file["user"]["created_at"], '%a %b %d %X %z %Y')
            users_test[file["user"]["id"]] = user_info

    os.chdir("../..")
    os.chdir("../..")

  os.chdir("../..")

os.chdir("../..")
os.chdir("../..")


print(len(posts))
print(len(users))
print(len(posts_test))
print(len(users_test))