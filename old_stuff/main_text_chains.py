import os

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

chains = dict()
os.chdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")


for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d))
    acc_chain = list()
    acc_chain.append(int(d))
    chains[d] = acc_chain.copy()
    os.chdir("replies")
    sons = os.listdir("..")
    if(len(sons)>0):
      sons.sort()
      for s in sons:
        acc_chain.append(int(s[:-5]))
        chains[int(s[:-5])] = acc_chain.copy()

    os.chdir("../..")
    os.chdir("../..")

  os.chdir("../..")

os.chdir("../..")
os.chdir("../..")

chains_test = dict()

sub_dirs = os.listdir("rumoureval-2019-test-data/twitter-en-test-data/")
os.chdir("rumoureval-2019-test-data/twitter-en-test-data/")

for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d))
    acc_chain = list()
    acc_chain.append(int(d))
    chains_test[d] = acc_chain.copy()
    os.chdir("replies")
    sons = os.listdir("..")
    if(len(sons)>0):
      sons.sort()
      for s in sons:
        acc_chain.append(int(s[:-5]))
        chains_test[int(s[:-5])] = acc_chain.copy()


    os.chdir("../..")
    os.chdir("../..")


  os.chdir("../..")

os.chdir("../..")
os.chdir("../..")

print(len(chains))
print(len(chains_test))