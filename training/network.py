import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        hidden_dim,
        activation=nn.ReLU(),
        normalization=None,
        out_transform=None,
    ):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        if normalization == "batch":
            self.normalization = nn.BatchNorm1d(hidden_dim)
        elif normalization == "layer":
            self.normalization = nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None
        self.out_transform = out_transform

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        x = self.output_layer(x)
        if self.out_transform is not None:
            return self.out_transform(x)
        return x
