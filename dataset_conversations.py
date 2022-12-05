import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import pad_chains, extract_thread


class TweetConversationsBERT(Dataset):

    def __init__(self, posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                 tokenizer, max_len_bert, labels):
        self.posts = posts
        self.users = users
        self.id = list()
        self.chains = list()
        self.trees = list()
        self.parents = list()
        self.graph_source = list()
        self.graph_dest = list()
        self.graph_time = list()
        self.discrete_graphs = list()
        self.thread = list()
        self.labels = list()

        dummy_post = dict()
        dummy_user = dict()

        dummy_post["text"] = " "
        dummy_post["user_id"] = -1
        dummy_post["in_reply_to_user_id"] = -1
        dummy_post["created_at"] = 0

        self.posts[-1] = dummy_post.copy()

        dummy_user["description"] = " "
        dummy_user["name"] = " "
        dummy_user["screen_name"] = " "
        dummy_user["created_at"] = 0

        self.users[-1] = dummy_user.copy()

        self.max_len_chain = 0
        self.max_len_thread = 0

        self.pad_vector = list()
        self.pad_vector_thread = list()

        for key in labels.keys():
            self.id.append(key)
            self.chains.append(chains[key])
            self.trees.append(trees[key])
            self.parents.append(parents[key])
            self.graph_source.append(graph_source[key])
            self.graph_dest.append(graph_dest[key])
            self.graph_time.append(graph_time[key])
            self.discrete_graphs.append(discrete_graphs[key])
            self.thread.append(extract_thread(trees[key], parents[key]))
            self.labels.append(labels[key])

        for ch in self.chains:
            self.max_len_chain = max(self.max_len_chain, len(ch))

        for th in self.thread:
            self.max_len_thread = max(self.max_len_thread, len(th))

        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

        pad_chains(self.chains, self.thread, self.trees, self.parents, self.graph_source, self.graph_dest,
                   self.graph_time,
                   self.pad_vector, self.pad_vector_thread, self.max_len_chain, self.max_len_thread)


    def __len__(self):
            return len(self.chains)

    def __getitem__(self, idx):

        post_id = self.id[idx]
        post = self.posts[post_id]
        user = self.users[post["user_id"]]

        text_post = post["text"]

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
            'labels': torch.Tensor([int(self.labels[idx])])
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
                 tokenizer, max_len_bert, labels):
        self.posts = posts
        self.users = users
        self.id = list()
        self.chains = list()
        self.trees = list()
        self.parents = list()
        self.graph_source = list()
        self.graph_dest = list()
        self.graph_time = list()
        self.discrete_graphs = list()
        self.thread = list()
        self.labels = list()

        dummy_post = dict()
        dummy_user = dict()

        dummy_post["text"] = ""
        dummy_post["user_id"] = -1
        dummy_post["in_reply_to_user_id"] = -1
        dummy_post["created_at"] = 0

        self.posts[-1] = dummy_post.copy()

        dummy_user["description"] = ""
        dummy_user["name"] = ""
        dummy_user["screen_name"] = ""
        dummy_user["created_at"] = 0

        self.users[-1] = dummy_user.copy()

        del dummy_post
        del dummy_user

        self.max_len_chain = 0
        self.max_len_thread = 0

        self.pad_vector = list()
        self.pad_vector_thread = list()

        for key in labels.keys():
            self.id.append(key)
            self.chains.append(chains[key])
            self.trees.append(trees[key])
            self.parents.append(parents[key])
            self.graph_source.append(graph_source[key])
            self.graph_dest.append(graph_dest[key])
            self.graph_time.append(graph_time[key])
            self.discrete_graphs.append(discrete_graphs[key])
            self.thread.append(extract_thread(trees[key], parents[key]))
            self.labels.append(labels[key])

        for ch in self.chains:
            self.max_len_chain = max(self.max_len_chain, len(ch))

        self.thread_size = list()

        for th in self.thread:
            self.max_len_thread = max(self.max_len_thread, len(th))
            self.thread_size.append(int(len(th)-1))

        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

        pad_chains(self.chains, self.thread, self.trees, self.parents, self.graph_source, self.graph_dest, self.graph_time,
                   self.pad_vector, self.pad_vector_thread, self.max_len_chain, self.max_len_thread)


    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):

        thread = self.thread[idx]

        input_ids_thread = torch.zeros(self.max_len_thread, self.max_len_bert, dtype=torch.int32)
        attention_mask_threads = torch.zeros(self.max_len_thread, self.max_len_bert, dtype=torch.int32)

        for idx_t in range(self.max_len_thread):
            post = self.posts[thread[idx_t]]
            text_post = post["text"]
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
            input_ids_thread[idx_t, :] = encoding["input_ids"].flatten()
            attention_mask_threads[idx_t, :] = encoding["attention_mask"].flatten()


        return {
            'input_ids': input_ids_thread,
            'attention_mask': attention_mask_threads,
            'len_thread': self.max_len_thread,
            'pad_vector_thread': self.pad_vector_thread,
            'thread_size': torch.Tensor([self.thread_size[idx]]),
            'labels': torch.Tensor([int(self.labels[idx])])
        }


def create_data_loaderLstmBERT(posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                       tokenizer, max_len_bert, labels, batch_size, shuffle=True):
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
        labels=labels
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=1, shuffle=shuffle)