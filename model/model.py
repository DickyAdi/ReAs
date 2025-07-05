import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import re

import utils

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):        
        # Calculate the attention scores
        scores = self.attention_weights(lstm_output)  # Shape: (batch_size, seq_len, hidden_size * 2)
        scores = torch.tanh(scores)  # Apply non-linearity

        # Calculate attention weights
        scores = self.context_vector(scores)  # Shape: (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # Shape: (batch_size, seq_len)
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len)

        # Calculate the context vector as the weighted sum of the LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # Shape: (batch_size, 1, hidden_size * 2)
        context = context.squeeze(1)  # Shape: (batch_size, hidden_size * 2)

        return context, attention_weights

class biLSTM_sentiment(nn.Module):
    def __init__(self, embedding_matrix, output_dim, n_layers, hidden_dim=512, use_attention=True, bidirectional=True, dropout=0, freeze_embedding=False) -> None:
        super(biLSTM_sentiment, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=freeze_embedding, padding_idx=0)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim//2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = attention(hidden_dim)
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        text_lengths = text_lengths.to('cpu')

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.use_attention:
            attention_context, attention_weights = self.attention(lstm_out)
            attention_context = attention_context.unsqueeze(1)
        else:
            attention_context = lstm_out

        lstm2_out, _ = self.lstm2(attention_context)

        if self.bidirectional:
            h = torch.cat((lstm2_out[:, -1, :self.hidden_dim], lstm2_out[:, 0, :self.hidden_dim]), dim=1)
        else:
            h = h[-1,:,:]

        out = self.fc(self.dropout(h))
        return out, attention_weights

class biLSTM_Attention(L.LightningModule):
    def __init__(self, lr, num_classes, embedding_matrix, hidden_dim=512, dropout=0, use_attention=True, bidirectional=True,optim_decay=0, class_weight=None, isMPS = True, freeze_embedding=False):
        super().__init__()
        self.freeze_embedding = freeze_embedding
        self.model = biLSTM_sentiment(embedding_matrix, 3, 1, dropout=dropout, freeze_embedding=freeze_embedding, use_attention=use_attention, bidirectional=bidirectional)
        self.isMPS = isMPS
        if not isMPS:
            self.accuracy = MulticlassAccuracy(num_classes)
            self.F1Score = MulticlassF1Score(num_classes)
        if class_weight is not None and len(class_weight) > 0:
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.num_classes = num_classes
        self.optim_decay = optim_decay
        self.use_attention = use_attention
        self.save_hyperparameters()
    def forward(self, text, text_lengths):
        # text = text.to('cpu')
        # text_lengths 
        return self.model(text, text_lengths)
    def training_step(self, batch, batch_idx):
        text, text_lengths, label = batch
        if self.use_attention:
            logits, weights = self.model(text, text_lengths)
        else:
            logits = self.model(text, text_lengths)
        loss = self.criterion(logits, label)
        _, preds = torch.max(logits, dim=1)
        if self.isMPS:
            temp_preds = preds.to('cpu')
            temp_label = label.to('cpu')
            log_values = {'train_acc' : multiclass_accuracy(temp_preds, temp_label, num_classes=self.num_classes), 'train_F1Score' : multiclass_f1_score(temp_preds, temp_label, num_classes=self.num_classes), 'train_loss' : loss}
            del temp_preds, temp_label
        else:
            log_values = {'train_acc' : self.accuracy(preds, label), 'train_F1Score' : self.F1Score(preds, label), 'train_loss' : loss}
        self.log_dict(log_values, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        text, text_lengths, label = batch
        if self.use_attention:
            logits, weights = self.model(text, text_lengths)
        else:
            logits = self.model(text, text_lengths)
        loss = self.criterion(logits, label)
        _, preds = torch.max(logits, dim=1)
        if self.isMPS:
            temp_preds = preds.to('cpu')
            temp_label = label.to('cpu')
            log_values = {'val_acc' : multiclass_accuracy(temp_preds, temp_label, num_classes=self.num_classes), 'val_F1Score' : multiclass_f1_score(temp_preds, temp_label, num_classes=self.num_classes), 'val_loss' : loss}
            del temp_label, temp_preds
        else:
            log_values = {'val_acc' : self.accuracy(preds, label), 'val_F1Score' : self.F1Score(preds, label), 'val_loss' : loss}
        self.log_dict(log_values, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.optim_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min', min_lr=0.0000001)
        # return {
        #     'optimizer' : optimizer,
        #     'lr_scheduler' : {
        #         'scheduler' : scheduler,
        #         'monitor' : 'val_loss',
        #         'frequency' : 1
        #     }
        # }
        return optimizer

class inference_model:
    def __init__(self, model, vocab):
        # self.model = model.load_from_checkpoint(weights_path).to('cpu').eval()
        self.model = model
        self.MAX_TOKEN = 256
        self.vocab = vocab
        self.class2idx = {
            'Positive' : 0,
            'Negative' : 1,
            'Neutral' : 2
        }
        self.idx2class = {
            0 : 'Positive',
            1 : 'Negative',
            2 : 'Neutral'
        }
    def __remove_emoticons(self,text):
        return re.sub(r'[^\w\s,.]', '', text)
    def __predict_prepare_data(self, sentence):
        # sentence = self.__remove_emoticons(sentence)
        text = torch.LongTensor(self.__sentence2idx(sentence)).unsqueeze(0)
        text_lengths = torch.tensor([len(self.__sentence2idx(sentence))])
        return text, text_lengths
    def __sentence2idx(self, sentence):
        sentenceidx = []
        for word in word_tokenize(sentence):
            if re.match(r'^[a-zA-Z+$]', word):
                word = word.lower()
            if word in self.vocab:
                sentenceidx.append(self.vocab[word])
            else:
                sentenceidx.append(0)
        return sentenceidx
    def predict(self, text):
        if not isinstance(text, str) or not text or text.strip() == "":
            raise TypeError(f"Expected type string. Got {type(text)}")
        prep_text, prep_len_text = self.__predict_prepare_data(text)
        # prep_len_text = prep_len_text.to('cpu')
        # prep_text = prep_text.to('cpu')
        with torch.no_grad():
            pred, context_weights = self.model(prep_text, prep_len_text)
        pred = self.idx2class[int(torch.argmax(pred, 1))]
        return pred, context_weights
    def visualize_attention(self, text, title=True):
        prediction, attention_weights = self.predict(text)
        sentence_list = word_tokenize(text)
        print(prediction)
        plt.figure(figsize=(15,6))
        sns.heatmap(attention_weights.cpu().numpy(), annot=False, cmap='Blues', xticklabels=sentence_list)
        plt.xticks(rotation=90)
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weights')
        if title:
            plt.title(text)
        plt.show()