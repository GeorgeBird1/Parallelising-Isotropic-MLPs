import torch
from torch import nn
import math
from collections.abc import Callable
from abc import ABC, abstractmethod
import warnings


# Defining a general IsotropicActivation function, such that users can create specific inherited isotropic functions and the neuroadaption will still recognise them
class IsotropicActivation(nn.Module, ABC):
    def __init__(
        self,
        trainable_intrinsic_length: bool = False,
        positive_intrinsic_length: bool = False,
        init_intrinsic_length: float = 1e-6,
        epsilon: float = 1e-3,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.epsilon = epsilon
        self.positive = positive_intrinsic_length
        self.trainable = trainable_intrinsic_length

        if self.trainable:
            if self.positive:
                if init_intrinsic_length <= 0:
                    raise ValueError(
                        "A positive trainable intrinsic length must be initialised above zero"
                    )

                self.intrinsic_length = nn.Parameter(
                    torch.tensor(
                        math.log(init_intrinsic_length),
                        **factory_kwargs,
                    )
                )
            else:
                self.intrinsic_length = nn.Parameter(
                    torch.tensor(
                        init_intrinsic_length,
                        **factory_kwargs,
                    )
                )
        else:
            self.register_buffer(
                "intrinsic_length",
                torch.tensor(
                    init_intrinsic_length,
                    **factory_kwargs,
                ),
            )

    @property
    def o(self) -> torch.Tensor:
        if self.trainable and self.positive:
            return torch.exp(self.intrinsic_length)

        return self.intrinsic_length

    @o.setter
    @torch.no_grad()
    def o(self, value: float | torch.Tensor) -> None:
        if value < 0 and self.positive: raise ValueError(f"Value must be positive, but is {value=:.4f}<0!")
        value = torch.as_tensor(
            value,
            device=self.intrinsic_length.device,
            dtype=self.intrinsic_length.dtype,
        )

        if value.numel() != 1:
            raise ValueError("Intrinsic length must be a scalar")

        if self.trainable and self.positive:
            if value <= 0:
                raise ValueError("Intrinsic length must be positive")

            self.intrinsic_length.copy_(value.log())
        else:
            self.intrinsic_length.copy_(value)

    @abstractmethod
    def radial_function(self, magnitude: torch.Tensor) -> torch.Tensor:
        r"""
        Computes f(magnitude), for F(\vec{x}) = f(\left\|\vec{x}\right\|_2) \hat{x}.
        """
        raise NotImplementedError

    @abstractmethod
    def origin_scale(self, magnitude: torch.Tensor) -> torch.Tensor:
        r"""
        Returns the limiting value of f(r) / r as r approaches zero.
        """
        raise NotImplementedError

    def _radial_scale(self, x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        squared_radius = torch.sum(
            torch.square(x),
            dim=dims,
            keepdim=True,
        )

        magnitude = torch.sqrt(
            torch.abs(self.o + squared_radius)
        )

        return torch.where(
            magnitude > self.epsilon,
            self.radial_function(magnitude) / magnitude.clamp_min(self.epsilon),
            self.origin_scale(magnitude),
        )

    def forward(self, x: torch.Tensor, dims: tuple[int, ...] = (-1,),) -> torch.Tensor:
        radial_scale = self._radial_scale(x, dims)
        return radial_scale * x

    def forward_with_scale(self, x: torch.Tensor, dims: tuple[int, ...] = (-1,),) -> tuple[torch.Tensor, torch.Tensor]:
        radial_scale = self._radial_scale(x, dims)
        return radial_scale * x, radial_scale

class IsotropicTanh(IsotropicActivation):
    def radial_function(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.tanh(magnitude)

    def origin_scale(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(magnitude)