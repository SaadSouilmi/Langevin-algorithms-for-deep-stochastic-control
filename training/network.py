import torch.nn as nn
import torch


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        hidden_dim: int,
        activation: nn.Module = nn.ReLU(),
        normalization: str = None,
        out_transform: nn.Module = None,
        dropout: float = None,
        initialization: str = "normal",
        seed: int = 42,
        sigma: float = 0.1,
    ):
        super(MLP, self).__init__()

        # Initialize random generator
        g = torch.Generator()
        g.manual_seed(seed)

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

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # Initialize weights
        with torch.no_grad():
            if initialization == "normal":
                self.input_layer.weight.normal_(mean=0, std=sigma, generator=g)
                for child in self.hidden_layers.children():
                    child.weight.normal_(mean=0, std=sigma, generator=g)
                self.output_layer.weight.normal_(mean=0, std=sigma, generator=g)
            elif initialization == "zero":
                self.input_layer.weight.fill_(0)
                for child in self.hidden_layers.children():
                    child.weight.fill_(0)
                self.output_layer.weight.fill_(0)
            else:
                raise ValueError(
                    "Invalid value of initialization, expected 'normal' or 'zero'."
                )

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.activation(x)
            if self.normalization is not None:
                x = self.normalization(x)
        x = self.output_layer(x)
        if self.out_transform is not None:
            return self.out_transform(x)
        return x
