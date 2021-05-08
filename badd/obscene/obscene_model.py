from torch import nn


class ObsceneModel(nn.Module):

    def __init__(self,
                 embeddings,
                 num_classes,
                 linear_size_1,
                 linear_size_2):
        super().__init__()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.batch_norm = nn.BatchNorm1d(num_features=embeddings.shape[-1])

        self.linear1 = nn.Linear(in_features=embeddings.shape[-1],
                                 out_features=linear_size_1,
                                 bias=False)
        self.linear2 = nn.Linear(in_features=linear_size_1,
                                 out_features=linear_size_2,
                                 bias=False)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(in_features=linear_size_2,
                                            out_features=num_classes,
                                            bias=False)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.batch_norm(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x