from needle import init
from needle.autograd import Tensor
from needle.nn.nn_basic import Module, Parameter


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        config = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }

        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=self.in_features,
                fan_out=self.out_features,
                **config,
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    fan_in=self.out_features,
                    fan_out=1,
                    **config,
                ).reshape((1, self.out_features))
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_weights = X @ self.weight
        if self.bias:
            return X_weights + self.bias.broadcast_to(X_weights.shape)
        return X_weights
