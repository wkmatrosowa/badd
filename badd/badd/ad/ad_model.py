from torch import nn

class AdModel(nn.Module):

    def __init__(self, embeddings, num_classes, embedding_dim, hidden_dim, num_lstm_layers,
                 padding_idx=0):
        super().__init__()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx, freeze=False)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_lstm_layers,
                                  batch_first=True,
                                  bidirectional=False)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(in_features=hidden_dim,
                                            out_features=num_classes,
                                            bias=False)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.relu(x)
        x = self.output_layer(x)

        return x