import torch.nn as nn
from transformers import AutoModel
import torch as torch
import torch.nn.functional as F
torch.set_printoptions(threshold=100_000)
torch.set_printoptions(profile="full")

class TweetConversationModel(nn.Module):

    def __init__(self,
                 num_classes,
                 device
                 ):

        super(TweetConversationModel, self).__init__()
        #BERT-based structure

        # TreeLSTM structure
        size = 768
        self.size = size  # size of input embeddings
        self.device = device  # cuda or cpu
        # final linear transformation, before softmax (our MLP)
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1).to(self.device)

        self.BERT = AutoModel.from_pretrained("vinai/bertweet-base")
        #for param in self.BERT.parameters():
        #    param.requires_grad = False
        self.linear1 = nn.Linear(self.BERT.config.hidden_size, 100)
        self.drop = nn.Dropout(p=0.5)
        self.drop_mlp = nn.Dropout(p=0.1)
        self.act_mlp = nn.ReLU()
        self.act_mlp2 = nn.Tanh()
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 4)
        self.softmax = nn.Softmax(dim=4).to(self.device)


    def forward(self, input_ids, attention_mask):
        bert_vector = self.BERT(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.drop(bert_vector.last_hidden_state[:, 0, :])

        output1 = self.drop(self.act_mlp(self.linear1(output)))
        output2 = self.drop(self.act_mlp(self.linear2(output1)))
        logits = self.act_mlp2(self.linear3(output2))

        return self.softmax(logits)