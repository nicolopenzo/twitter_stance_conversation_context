import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import pad_chains, extract_thread
import json
from datetime import datetime, timedelta

class TweetConversationsBERT(Dataset):

    def __init__(self, posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                 tokenizer, max_len_bert, labels):

        self.id = list()
        chains_list = list()
        trees_list = list()
        parents_list = list()
        graph_source_list = list()
        graph_dest_list = list()
        graph_time_list = list()
        discrete_graphs_list = list()
        thread_list = list()
        labels_list = list()

        dummy_post = dict()
        dummy_user = dict()

        dummy_post["text"] = " "
        dummy_post["user_id"] = -1
        dummy_post["in_reply_to_user_id"] = -1
        dummy_post["created_at"] = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')

        posts[-1] = dummy_post.copy()

        dummy_user["description"] = " "
        dummy_user["name"] = " "
        dummy_user["screen_name"] = " "
        dummy_user["created_at"] = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')

        users[-1] = dummy_user.copy()

        self.max_len_chain = 0
        self.max_len_thread = 0

        for key in labels.keys():
            self.id.append(key)
            chains_list.append(chains[key])
            trees_list.append(trees[key])
            parents_list.append(parents[key])
            graph_source_list.append(graph_source[key])
            graph_dest_list.append(graph_dest[key])
            graph_time_list.append(graph_time[key])
            discrete_graphs_list.append(discrete_graphs[key])
            thread_list.append(extract_thread(trees[key], parents[key]))
            labels_list.append(labels[key])

        pad_vector = list()
        pad_vector_thread = list()

        for ch in chains_list:
            self.max_len_chain = max(self.max_len_chain, len(ch))

        for th in thread_list:
            self.max_len_thread = max(self.max_len_thread, len(th))

        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

        pad_chains(chains_list, thread_list, trees_list, parents_list, graph_source_list, graph_dest_list,
                   graph_time_list, pad_vector, pad_vector_thread, self.max_len_chain, self.max_len_thread)

        for idx in range(len(self.id)):
            info = dict()
            info["chains"] = chains_list[idx]
            info["trees"] = trees_list[idx]
            info["parents"] = parents_list[idx]
            #info["graph_source"] = graph_source_list[idx]
            #info["graph_dest"] = graph_dest_list[idx]
            #info["graph_time"] = graph_time_list[idx]
            #info["discrete_graphs"] = discrete_graphs_list[idx]
            info["thread"] = thread_list[idx]
            info["labels"] = labels_list[idx]
            info["pad_vector"] = pad_vector[idx]
            info["pad_vector_thread"] = pad_vector_thread[idx]
            with open("structures/"+str(self.id[idx])+'.json', 'w') as fp:
                json.dump(info, fp)
                fp.close()
        offset = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')
        for key in posts.keys():
            date = posts[key]["created_at"]
            if type(date) is not float:
                date = (date - offset).total_seconds()
            posts[key]["created_at"] = date
            if key == -1:
                filename = "posts/dummy.json"
            else:
                filename = "posts/"+str(key)+".json"
            with open(filename, 'w') as fp:
                json.dump(posts[key], fp)
                fp.close()

        for key in users.keys():
            date = users[key]["created_at"]
            if type(date) is not float:
                date = (date - offset).total_seconds()
            users[key]["created_at"] = date
            if key == -1:
                filename = "users/dummy.json"
            else:
                filename = "users/"+str(key)+".json"
            with open(filename, 'w') as fp:
                json.dump(users[key], fp)
                fp.close()



    def __len__(self):
            return len(self.id)

    def __getitem__(self, idx):

        post_id = self.id[idx]
        with open("posts/"+str(self.id[idx])+".json", 'r') as f:
            file = json.load(f)
            text_post = file["text"]
            user_id = file["user_id"]
            f.close()
            del file

        with open("structures/"+str(self.id[idx])+".json", 'r') as f:
            file = json.load(f)
            labels = file["labels"]
            f.close()
            del file

        encoding = self.tokenizer.encode_plus(
            text_post,
            max_length=self.max_len_bert,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True
        )

        return {
            'post_text': text_post,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.Tensor([int(labels)])
        }


def create_data_loaderBERT(posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                       tokenizer, max_len_bert, labels, batch_size, shuffle=True):
    ds = TweetConversationsBERT(
        posts=posts,
        users=users,
        chains=chains,
        trees=trees,
        parents=parents,
        graph_source=graph_source,
        graph_dest=graph_dest,
        graph_time=graph_time,
        discrete_graphs=discrete_graphs,
        tokenizer=tokenizer,
        max_len_bert=max_len_bert,
        labels=labels
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=1, shuffle=shuffle)

class TweetConversationsLstmBERT(Dataset):

    def __init__(self, posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                 tokenizer, max_len_bert, labels, thread):
        self.id = list()
        chains_list = list()
        trees_list = list()
        parents_list = list()
        graph_source_list = list()
        graph_dest_list = list()
        graph_time_list = list()
        discrete_graphs_list = list()
        thread_list = list()
        labels_list = list()
        thread_size_list = list()
        chain_size_list = list()

        self.thread = thread

        dummy_post = dict()
        dummy_user = dict()

        dummy_post["text"] = "fake"
        dummy_post["user_id"] = -1
        dummy_post["in_reply_to_user_id"] = -1
        dummy_post["created_at"] = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')

        posts[-1] = dummy_post.copy()

        dummy_user["description"] = ""
        dummy_user["name"] = ""
        dummy_user["screen_name"] = ""
        dummy_user["created_at"] = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')

        users[-1] = dummy_user.copy()

        self.max_len_chain = 0
        self.max_len_thread = 0

        for key in labels.keys():
            self.id.append(key)
            chains_list.append(chains[key])
            trees_list.append(trees[key])
            parents_list.append(parents[key])
            graph_source_list.append(graph_source[key])
            graph_dest_list.append(graph_dest[key])
            graph_time_list.append(graph_time[key])
            discrete_graphs_list.append(discrete_graphs[key])
            thread_list.append(extract_thread(trees[key], parents[key]))
            labels_list.append(labels[key])

        pad_vector = list()
        pad_vector_thread = list()

        for ch in chains_list:
            self.max_len_chain = max(self.max_len_chain, len(ch))
            chain_size_list.append(int(len(ch)-1))

        for th in thread_list:
            self.max_len_thread = max(self.max_len_thread, len(th))
            thread_size_list.append(int(len(th) - 1))

        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

        pad_chains(chains_list, thread_list, trees_list, parents_list, graph_source_list, graph_dest_list,
                   graph_time_list, pad_vector, pad_vector_thread, self.max_len_chain, self.max_len_thread)

        for idx in range(len(self.id)):
            info = dict()
            info["chains"] = chains_list[idx]
            info["trees"] = trees_list[idx]
            info["parents"] = parents_list[idx]
            # info["graph_source"] = graph_source_list[idx]
            # info["graph_dest"] = graph_dest_list[idx]
            # info["graph_time"] = graph_time_list[idx]
            # info["discrete_graphs"] = discrete_graphs_list[idx]
            info["thread"] = thread_list[idx]
            info["labels"] = labels_list[idx]
            info["pad_vector"] = pad_vector[idx]
            info["pad_vector_thread"] = pad_vector_thread[idx]
            info["thread_size"] = thread_size_list[idx]
            info["chain_size"] = chain_size_list[idx]
            with open("structures/" + str(self.id[idx]) + '.json', 'w') as fp:
                json.dump(info, fp)
                fp.close()

        offset = datetime.strptime("Thu Jan 01 00:00:00 +0000 1970", '%a %b %d %H:%M:%S %z %Y')
        for key in posts.keys():
            date = posts[key]["created_at"]
            if type(date) is not float:
                date = (date - offset).total_seconds()
            posts[key]["created_at"] = date
            if key == -1:
                filename = "posts/dummy.json"
            else:
                filename = "posts/"+str(key)+".json"
            with open(filename, 'w') as fp:
                json.dump(posts[key], fp)
                fp.close()

        for key in users.keys():
            date = users[key]["created_at"]
            if type(date) is not float:
                date = (date - offset).total_seconds()
            users[key]["created_at"] = date
            if key == -1:
                filename = "users/dummy.json"
            else:
                filename = "users/"+str(key)+".json"
            with open(filename, 'w') as fp:
                json.dump(users[key], fp)
                fp.close()


    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):

        with open("structures/" + str(self.id[idx]) + ".json", 'r') as f:
            file = json.load(f)
            if self.thread:
                seq = file["thread"]
                pad_seq = file["pad_vector_thread"]
                seq_size = file["thread_size"]
                length = self.max_len_thread
            else:
                seq = file["chains"]
                pad_seq = file["pad_vector"]
                seq_size = file["chain_size"]
                length = self.max_len_chain
            labels = file["labels"]
            f.close()
            del file

        input_ids_seq = torch.zeros(length, self.max_len_bert, dtype=torch.int32)
        attention_mask_seq = torch.zeros(length, self.max_len_bert, dtype=torch.int32)
        for idx_t in range(length):
            if seq[idx_t] == -1:
                filename = "dummy"
            else:
                filename = str(seq[idx_t])

            with open("posts/" + filename + ".json", 'r') as f:
                file = json.load(f)
                text_post = file["text"]

                f.close()
                del file

            encoding = self.tokenizer.encode_plus(
                text_post,
                max_length=self.max_len_bert,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                truncation=True
            )
            input_ids_seq[idx_t, :] = encoding["input_ids"].flatten()
            attention_mask_seq[idx_t, :] = encoding["attention_mask"].flatten()


        return {
            'input_ids': input_ids_seq,
            'attention_mask': attention_mask_seq,
            'len_seq': int(self.max_len_seq),
            'pad_vector_seq': torch.Tensor(pad_seq),
            'seq_size': torch.Tensor([seq_size]),
            'labels': torch.Tensor([int(labels)])
        }

def create_data_loaderLstmBERT(posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                       tokenizer, max_len_bert, labels, batch_size, shuffle=True, thread=True):
    ds = TweetConversationsLstmBERT(
        posts=posts,
        users=users,
        chains=chains,
        trees=trees,
        parents=parents,
        graph_source=graph_source,
        graph_dest=graph_dest,
        graph_time=graph_time,
        discrete_graphs=discrete_graphs,
        tokenizer=tokenizer,
        max_len_bert=max_len_bert,
        thread=thread,
        labels=labels
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=shuffle)