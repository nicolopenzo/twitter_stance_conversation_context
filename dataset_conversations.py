import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import pad_chains


class TweetConversations(Dataset):

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
        self.labels = list()

        max_len_chain = 0

        self.pad_vector = list()
        for key in labels.keys():
            self.id.append(key)
            self.chains.append(chains[key])
            self.trees.append(trees[key])
            self.parents.append(parents[key])
            self.graph_source.append(graph_source[key])
            self.graph_dest.append(graph_dest[key])
            self.graph_time.append(graph_time[key])
            self.discrete_graphs.append(discrete_graphs[key])
            self.labels.append(labels[key])

        for ch in self.chains:
            max_len_chain = max(max_len_chain, len(ch))
        self.tokenizer = tokenizer
        self.max_len_bert = max_len_bert

        pad_chains(self.chains, self.trees, self.parents, self.graph_source, self.graph_dest, self.graph_time,
                   self.pad_vector, max_len_chain)


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
            'input_ids': torch.Tensor(encoding['input_ids']),
            'attention_mask': torch.Tensor(encoding['attention_mask']),
            'labels': torch.Tensor([int(self.labels[idx])])
        }


def create_data_loader(posts, users, chains, trees, parents, graph_source, graph_dest, graph_time, discrete_graphs,
                       tokenizer, max_len_bert, labels, batch_size, shuffle=True):
    ds = TweetConversations(
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