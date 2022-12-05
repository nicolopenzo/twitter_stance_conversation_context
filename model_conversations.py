import torch.nn as nn
from transformers import AutoModel
import torch as torch
import torch.nn.functional as F
torch.set_printoptions(threshold=100_000)
torch.set_printoptions(profile="full")

class TweetConversationBERT(nn.Module):

    def __init__(self,
                 num_classes,
                 device
                 ):

        super(TweetConversationBERT, self).__init__()
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
        self.softmax = nn.Softmax(dim=1).to(self.device)


    def forward(self, input_ids, attention_mask):
        print(input_ids.size())

        bert_vector = self.BERT(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_vector.last_hidden_state[:, 0, :])

        output1 = self.drop(self.act_mlp(self.linear1(output)))
        output2 = self.drop(self.act_mlp(self.linear2(output1)))
        logits = self.act_mlp2(self.linear3(output2))

        return self.softmax(logits)

class TweetConversationLstmBERT(nn.Module):

    def __init__(self,
                 num_classes,
                 device
                 ):

        super(TweetConversationLstmBERT, self).__init__()
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

        #LSTM
        self.lstm = torch.nn.LSTMCell(768, 768, bias=False).to(self.device)

        self.linear1 = nn.Linear(self.BERT.config.hidden_size, 100)
        self.drop = nn.Dropout(p=0.5)
        self.drop_mlp = nn.Dropout(p=0.1)
        self.act_mlp = nn.ReLU()
        self.act_mlp2 = nn.Tanh()
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 4)
        self.softmax = nn.Softmax(dim=1).to(self.device)


    def forward(self, input_ids, attention_mask, len_seq, pad_vector_seq, thread_real_seq):
        h = torch.Tensor.zero_(torch.Tensor(input_ids.size(dim=0), 768)).to(self.device)
        c = torch.Tensor.zero_(torch.Tensor(input_ids.size(dim=0), 768)).to(self.device)
        diff= torch.ones(input_ids.size(dim=0), 768).to(self.device)
        fin = torch.Tensor.zero_(torch.Tensor(input_ids.size(dim=0), 768)).to(self.device)
        # LSTM
        for i in range(int(len_seq[0])):
            input = self.BERT(
                input_ids=input_ids[:, i, :],
                attention_mask=attention_mask[:, i, :]
                )
            input2 = self.drop(input.last_hidden_state[:, 0, :])
            h, c = self.lstm(input2, (h, c))


            diff.size()
            p = pad_vector_seq[:, i].repeat(768,1).transpose(1,0)
            # if it is padding value, drop to zero h and c
            fin = fin * (diff-p) + h * p
            del input
            del input2


        output1 = self.drop(self.act_mlp(self.linear1(fin)))
        output2 = self.drop(self.act_mlp(self.linear2(output1)))
        logits = self.act_mlp2(self.linear3(output2))

        return self.softmax(logits)