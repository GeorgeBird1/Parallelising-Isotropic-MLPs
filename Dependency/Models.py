import torch
import torch.nn as nn
import math

from .RewrittenLayers import IsotropicActivation

class ElementwiseTanh(nn.Module):
    """I've recoded this simply to have it on a fair even playing field against isotropic-tanh, no behind the scenes acceleration"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)

class NormalLinear(nn.Module):
    """
    This is just a standard isotropic dense network as you'd construct normally, with all operations sequential
    """

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3: raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width_structure[i], width_structure[i + 1]))
             for i in range(len(width_structure) - 1)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(width_structure[i + 1]))
             for i in range(len(width_structure) - 1)]
        )
        self.activations = nn.ModuleList([activation_class() for _ in range(len(width_structure) - 2)])

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)
        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    def forward(self, x):
        for i in range(self.layers - 2):
            x = self.activations[i](x @ self.weights[i] + self.biases[i])

        return (x @ self.weights[-1] + self.biases[-1])


class FactorisedParallelLinear_IndependentChains(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions (preserving the original matrix structure)"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3: raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width_structure[i], width_structure[i + 1]))
             for i in range(len(width_structure) - 1)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(width_structure[i + 1]))
             for i in range(len(width_structure) - 1)]
        )
        self.activations = nn.ModuleList([activation_class() for _ in range(len(width_structure) - 2)])

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)
        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    @staticmethod
    def multiply_chain(matrices: list[torch.Tensor | nn.Parameter] | nn.ParameterList):
        match len(matrices):
            case 0: return None
            case 1: return matrices[0]
            case _:
                chain_length = len(matrices)
                prefix = matrices[chain_length - 1]
                for i in range(chain_length - 1):
                    prefix = prefix @ matrices[chain_length - 2 - i]
                return prefix
    @staticmethod
    def multiply_scalars(scalars: list[torch.Tensor | nn.Parameter]):
        match len(scalars):
            case 0: return 1.0
            case 1: return scalars[0]
            case _:
                a = scalars[0]
                for i in range(1, len(scalars)):
                    a = a * scalars[i]
                return a

    def forward(self, x):
        # These form W_3 @ W_2 @ W_1 @ x, W_2 @ W_1 @ x, W_1 @ x etc.
        temp_weights = [x @ self.multiply_chain(self.weights[:i + 1][::-1]) for i in range(len(self.weights))]
        # These form [[W_3 W_2 W_1 b_0, W_2 W_1 b_0, W_1 b_0, b_0], [W_3 W_2 b_1, W_2 b_1, b_1], [W_3 b_2, b_2], [b_3]] etc.
        temp_biases = [
            [bias,]+[bias @ self.multiply_chain(self.weights[i + 1 : j + 1][::-1]) for j in range(i+1, len(self.weights))] for
            i, bias in enumerate(self.biases)
        ]
        reordered_biases = [[row[k - i] for i, row in enumerate(temp_biases) if 0 <= k - i < len(row)] for k in range(len(temp_biases))]

        s_factors = []

        for i in range(self.layers-2):
            s_factors.append(
                self.activations[i]._radial_scale(
                    self.multiply_scalars(s_factors)*(temp_weights[i] + reordered_biases[i][0]) + sum([self.multiply_scalars(s_factors[j+1:]) * bias for j, bias in enumerate(reordered_biases[i][1:])]), dims=1
                )
            )

        final = self.multiply_scalars(s_factors)*(temp_weights[-1] + reordered_biases[-1][0]) + sum([self.multiply_scalars(s_factors[j+1:]) * bias for j, bias in enumerate(reordered_biases[-1][1:])])
        return final

    @torch.no_grad()
    def prep_inference_time(self):
        # Precompute all weight chains
        parallel_weights = [
            self.multiply_chain(self.weights[:i + 1][::-1])
            for i in range(len(self.weights))
        ]

        self.parallel_widths = self.width_structure[1:]
        self.W_parallel = torch.hstack(parallel_weights)

        # Precompute all weight-bias contractions
        temp_biases = [
            [bias,] + [
                bias @ self.multiply_chain(self.weights[i + 1:j + 1][::-1])
                for j in range(i + 1, len(self.weights))
            ]
            for i, bias in enumerate(self.biases)
        ]

        self.parallel_biases = [
            [row[k - i] for i, row in enumerate(temp_biases) if 0 <= k - i < len(row)]
            for k in range(len(temp_biases))
        ]

    def forward_inference_time(self, x):
        if not hasattr(self, "W_parallel") or not hasattr(self, "parallel_biases"):
            raise RuntimeError("Must call prep_inference_time() before forward_inference_time()")

        # Compute all weight-input products simultaneously
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            s_factors.append(
                self.activations[i]._radial_scale(
                    self.multiply_scalars(s_factors) * (temp_weights[i] + self.parallel_biases[i][0])
                    + sum([
                        self.multiply_scalars(s_factors[j + 1:]) * bias
                        for j, bias in enumerate(self.parallel_biases[i][1:])
                    ]),
                    dims=1
                )
            )

        final = self.multiply_scalars(s_factors) * (temp_weights[-1] + self.parallel_biases[-1][0]) \
                + sum([
                    self.multiply_scalars(s_factors[j + 1:]) * bias
                    for j, bias in enumerate(self.parallel_biases[-1][1:])
                ])

        return final


class ContractedParallelLinear(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions - this stores the weights as already precontracted terms, altering the gradient descent dynamics"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3: raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width_structure[0], width_structure[i]))
             for i in range(1, len(width_structure))]
        )
        self.biases = nn.ParameterList(
            [
                nn.ParameterList([nn.Parameter(torch.empty(width_structure[i])) for _ in range(i)])
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList([activation_class() for _ in range(len(width_structure) - 2)])

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)
        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    @staticmethod
    def multiply_scalars(scalars: list[torch.Tensor | nn.Parameter]):
        match len(scalars):
            case 0:
                return 1.0
            case 1:
                return scalars[0]
            case _:
                a = scalars[0]
                for i in range(1, len(scalars)):
                    a = a * scalars[i]
                return a

    def forward(self, x):
        s_factors = []

        for i in range(self.layers - 2):
            s_factors.append(
                self.activations[i]._radial_scale(
                    self.multiply_scalars(s_factors) * x @ self.weights[i] + sum([self.multiply_scalars(s_factors[j:]) * b for j, b in enumerate(self.biases[i])]), dims=1
                )
            )

        final = self.multiply_scalars(s_factors) * x @ self.weights[-1] + sum([self.multiply_scalars(s_factors[j:]) * b for j, b in enumerate(self.biases[-1])])
        return final

class ReparameterisedParallelLinear_VectorBiases(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions - this stores the weights as already precontracted terms, altering the gradient descent dynamics"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3: raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.parallel_widths = width_structure[1:]

        self.W_parallel = nn.Parameter(
            torch.empty(width_structure[0], sum(self.parallel_widths))
        )

        self.biases = nn.ParameterList(
            [
                nn.ParameterList([nn.Parameter(torch.empty(width_structure[i])) for _ in range(i)])
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList([activation_class() for _ in range(len(width_structure) - 2)])

    def reset_parameters(self):
        start = 0
        for width in self.parallel_widths:
            nn.init.orthogonal_(self.W_parallel[:, start:start + width])
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    def reset_parameters_xavier(self):
        start = 0
        for width in self.parallel_widths:
            nn.init.xavier_normal_(self.W_parallel[:, start:start + width])
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    @staticmethod
    def multiply_scalars(scalars: list[torch.Tensor | nn.Parameter]):
        match len(scalars):
            case 0:
                return 1.0
            case 1:
                return scalars[0]
            case _:
                a = scalars[0]
                for i in range(1, len(scalars)):
                    a = a * scalars[i]
                return a

    def forward(self, x):
        # Compute all independent weight-input products simultaneously
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            s_factors.append(
                self.activations[i]._radial_scale(
                    self.multiply_scalars(s_factors) * temp_weights[i]
                    + sum([self.multiply_scalars(s_factors[j:]) * b for j, b in enumerate(self.biases[i])]),
                    dims=1
                )
            )

        final = self.multiply_scalars(s_factors) * temp_weights[-1] \
                + sum([self.multiply_scalars(s_factors[j:]) * b for j, b in enumerate(self.biases[-1])])

        return final


class FactorisedParallelLinear_RecursiveChains(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions (preserving the original matrix structure)"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i], width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    @staticmethod
    def combine_bias_terms(s_factors, biases):
        coefficient = 1.0
        bias_term = biases[-1]

        for j in range(len(s_factors) - 1, -1, -1):
            coefficient = coefficient * s_factors[j]
            bias_term = bias_term + coefficient * biases[j]

        return coefficient, bias_term

    def forward(self, x):
        # Build W_0, W_0 W_1, W_0 W_1 W_2, ... once each.
        parallel_weights = [self.weights[0]]
        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ torch.hstack(parallel_weights),
            self.width_structure[1:],
            dim=1
        )

        # Build b_i, b_i W_{i+1}, b_i W_{i+1} W_{i+2}, ... once each.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        reordered_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

        s_factors = []

        for i in range(self.layers - 2):
            coefficient, bias_term = self.combine_bias_terms(
                s_factors,
                reordered_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    coefficient * temp_weights[i] + bias_term,
                    dims=1
                )
            )

        coefficient, bias_term = self.combine_bias_terms(
            s_factors,
            reordered_biases[-1]
        )

        final = coefficient * temp_weights[-1] + bias_term
        return final

    @torch.no_grad()
    def prep_inference_time(self):
        # Precompute W_0, W_0 W_1, W_0 W_1 W_2, ...
        parallel_weights = [self.weights[0]]

        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        self.parallel_widths = self.width_structure[1:]
        self.W_parallel = torch.hstack(parallel_weights)

        # Precompute all weight-bias contractions.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        self.parallel_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

    def forward_inference_time(self, x):
        if not hasattr(self, "W_parallel") or not hasattr(self, "parallel_biases"):
            raise RuntimeError(
                "Must call prep_inference_time() before forward_inference_time()"
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            coefficient, bias_term = self.combine_bias_terms(
                s_factors,
                self.parallel_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    coefficient * temp_weights[i] + bias_term,
                    dims=1
                )
            )

        coefficient, bias_term = self.combine_bias_terms(
            s_factors,
            self.parallel_biases[-1]
        )

        final = coefficient * temp_weights[-1] + bias_term
        return final

class ReparameterisedParallelLinear_RecursiveBiases(nn.Module):
    """Stores the independent reparameterised weight matrices in one parallel matrix, altering the gradient descent dynamics"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.parallel_widths = width_structure[1:]

        self.W_parallel = nn.Parameter(
            torch.empty(
                width_structure[0],
                sum(self.parallel_widths)
            )
        )

        self.biases = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(width_structure[i]))
                        for _ in range(i)
                    ]
                )
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.orthogonal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    def reset_parameters_xavier(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.xavier_normal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    @staticmethod
    def combine_bias_terms(s_factors, biases):
        coefficient = 1.0
        bias_term = biases[-1]

        for j in range(len(s_factors) - 1, -1, -1):
            coefficient = coefficient * s_factors[j]
            bias_term = bias_term + coefficient * biases[j]

        return coefficient, bias_term

    def forward(self, x):
        # Compute all independent weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            coefficient, bias_term = self.combine_bias_terms(
                s_factors,
                self.biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    coefficient * temp_weights[i] + bias_term,
                    dims=1
                )
            )

        coefficient, bias_term = self.combine_bias_terms(
            s_factors,
            self.biases[-1]
        )

        final = coefficient * temp_weights[-1] + bias_term
        return final

class FactorisedParallelLinear_GroupedScalars(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions (preserving the original matrix structure)"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i], width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    @staticmethod
    def combine_bias_terms(s_factors, temp_weight, biases):
        if len(s_factors) == 0:
            return temp_weight + biases[0]

        coefficient = 1.0
        bias_term = biases[-1]

        for j in range(len(s_factors) - 1, 0, -1):
            coefficient = coefficient * s_factors[j]
            bias_term = bias_term + coefficient * biases[j]

        coefficient = coefficient * s_factors[0]

        return coefficient * (temp_weight + biases[0]) + bias_term

    def forward(self, x):
        # Build W_0, W_0 W_1, W_0 W_1 W_2, ... once each.
        parallel_weights = [self.weights[0]]

        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ torch.hstack(parallel_weights),
            self.width_structure[1:],
            dim=1
        )

        # Build b_i, b_i W_{i+1}, b_i W_{i+1} W_{i+2}, ... once each.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        reordered_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                reordered_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            reordered_biases[-1]
        )

        return final

    @torch.no_grad()
    def prep_inference_time(self):
        # Precompute W_0, W_0 W_1, W_0 W_1 W_2, ...
        parallel_weights = [self.weights[0]]

        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        self.parallel_widths = self.width_structure[1:]
        self.W_parallel = torch.hstack(parallel_weights)

        # Precompute all weight-bias contractions.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        self.parallel_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

    def forward_inference_time(self, x):
        if not hasattr(self, "W_parallel") or not hasattr(self, "parallel_biases"):
            raise RuntimeError(
                "Must call prep_inference_time() before forward_inference_time()"
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                self.parallel_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            self.parallel_biases[-1]
        )

        return final

class ReparameterisedParallelLinear_MatrixBiases(nn.Module):
    """Stores the independent reparameterised weight matrices in one parallel matrix, altering the gradient descent dynamics"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.parallel_widths = width_structure[1:]

        self.W_parallel = nn.Parameter(
            torch.empty(
                width_structure[0],
                sum(self.parallel_widths)
            )
        )

        self.biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(i, width_structure[i])
                )
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.orthogonal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.xavier_normal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            nn.init.zeros_(b)

    @staticmethod
    def combine_bias_terms(s_factors, temp_weight, biases):
        if len(s_factors) == 0:
            return temp_weight + biases[0]

        coefficient = 1.0
        bias_term = biases[-1]
        bias_coefficients = []

        for j in range(len(s_factors) - 1, 0, -1):
            coefficient = coefficient * s_factors[j]
            bias_coefficients.append(coefficient)

        if len(bias_coefficients) > 0:
            bias_coefficients = torch.cat(
                bias_coefficients[::-1],
                dim=1
            )

            bias_term = bias_term + bias_coefficients @ biases[1:-1]

        coefficient = coefficient * s_factors[0]

        return coefficient * (temp_weight + biases[0]) + bias_term

    def forward(self, x):
        # Compute all independent weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                self.biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            self.biases[-1]
        )

        return final

class FactorisedParallelLinear_FinalOptimised(nn.Module):
    """This form does compute the standard isotropic dense network, but with precomputation of terms via contractions (preserving the original matrix structure)"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i], width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width_structure[i + 1]))
                for i in range(len(width_structure) - 1)
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        for W in self.weights:
            nn.init.orthogonal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        for W in self.weights:
            nn.init.xavier_normal_(W)

        for b in self.biases:
            nn.init.zeros_(b)

    @staticmethod
    def combine_bias_terms(s_factors, temp_weight, biases):
        if len(s_factors) == 0:
            return temp_weight + biases[0]

        coefficient = 1.0
        bias_term = biases[-1]

        for j in range(len(s_factors) - 1, 0, -1):
            coefficient = coefficient * s_factors[j]
            bias_term = bias_term + coefficient * biases[j]

        coefficient = coefficient * s_factors[0]

        return coefficient * (temp_weight + biases[0]) + bias_term

    def forward(self, x):
        # Build W_0, W_0 W_1, W_0 W_1 W_2, ... once each.
        parallel_weights = [self.weights[0]]

        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ torch.hstack(parallel_weights),
            self.width_structure[1:],
            dim=1
        )

        # Build b_i, b_i W_{i+1}, b_i W_{i+1} W_{i+2}, ... once each.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        reordered_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                reordered_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            reordered_biases[-1]
        )

        return final

    @torch.no_grad()
    def prep_inference_time(self):
        # Precompute W_0, W_0 W_1, W_0 W_1 W_2, ...
        parallel_weights = [self.weights[0]]

        for i in range(1, len(self.weights)):
            parallel_weights.append(
                parallel_weights[-1] @ self.weights[i]
            )

        self.parallel_widths = self.width_structure[1:]
        self.W_parallel = torch.hstack(parallel_weights)

        # Precompute all weight-bias contractions.
        temp_biases = []

        for i, bias in enumerate(self.biases):
            row = [bias]

            for j in range(i + 1, len(self.weights)):
                row.append(
                    row[-1] @ self.weights[j]
                )

            temp_biases.append(row)

        self.parallel_biases = [
            [
                row[k - i]
                for i, row in enumerate(temp_biases)
                if 0 <= k - i < len(row)
            ]
            for k in range(len(temp_biases))
        ]

    def forward_inference_time(self, x):
        if not hasattr(self, "W_parallel") or not hasattr(self, "parallel_biases"):
            raise RuntimeError(
                "Must call prep_inference_time() before forward_inference_time()"
            )

        # Compute all weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                self.parallel_biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            self.parallel_biases[-1]
        )

        return final

class ReparameterisedParallelLinear_GroupedScalars(nn.Module):
    """Stores the independent reparameterised weight matrices in one parallel matrix, altering the gradient descent dynamics"""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.parallel_widths = width_structure[1:]

        self.W_parallel = nn.Parameter(
            torch.empty(
                width_structure[0],
                sum(self.parallel_widths)
            )
        )

        self.biases = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(width_structure[i]))
                        for _ in range(i)
                    ]
                )
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.orthogonal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    def reset_parameters_xavier(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.xavier_normal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            for a in b:
                nn.init.zeros_(a)

    @staticmethod
    def combine_bias_terms(s_factors, temp_weight, biases):
        if len(s_factors) == 0:
            return temp_weight + biases[0]

        coefficient = 1.0
        bias_term = biases[-1]

        for j in range(len(s_factors) - 1, 0, -1):
            coefficient = coefficient * s_factors[j]
            bias_term = bias_term + coefficient * biases[j]

        coefficient = coefficient * s_factors[0]

        return coefficient * (temp_weight + biases[0]) + bias_term

    def forward(self, x):
        # Compute all independent weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        s_factors = []

        for i in range(self.layers - 2):
            z = self.combine_bias_terms(
                s_factors,
                temp_weights[i],
                self.biases[i]
            )

            s_factors.append(
                self.activations[i]._radial_scale(
                    z,
                    dims=1
                )
            )

        final = self.combine_bias_terms(
            s_factors,
            temp_weights[-1],
            self.biases[-1]
        )

        return final

class ReparameterisedParallelLinear_CoefficientMatrix(nn.Module):
    """Stores the independent reparameterised weight matrices in one parallel matrix, with bias-like terms contracted in parallel."""

    def __init__(self, width_structure: list[int], activation_class: IsotropicActivation):
        if len(width_structure) < 3:
            raise ValueError("Please ensure at least one hidden layer!")

        super().__init__()
        self.width_structure = width_structure
        self.layers = len(width_structure)

        self.parallel_widths = width_structure[1:]

        self.W_parallel = nn.Parameter(
            torch.empty(
                width_structure[0],
                sum(self.parallel_widths)
            )
        )

        # Each row is one independent bias-like term.
        self.biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(i, width_structure[i])
                )
                for i in range(1, len(width_structure))
            ]
        )

        self.activations = nn.ModuleList(
            [activation_class() for _ in range(len(width_structure) - 2)]
        )

    def reset_parameters(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.orthogonal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            nn.init.zeros_(b)

    def reset_parameters_xavier(self):
        start = 0

        for width in self.parallel_widths:
            nn.init.xavier_normal_(
                self.W_parallel[:, start:start + width]
            )
            start += width

        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, x):
        # Compute all independent weight-input products simultaneously.
        temp_weights = torch.split(
            x @ self.W_parallel,
            self.parallel_widths,
            dim=1
        )

        # Initially the only nonlinear coefficient is 1.
        coefficients = torch.ones(
            x.shape[0],
            1,
            device=x.device,
            dtype=x.dtype
        )

        for i in range(self.layers - 2):
            # S_0 y_i + sum_j S_j b_ij
            z = (
                coefficients[:, :1] * temp_weights[i]
                + coefficients @ self.biases[i]
            )

            s = self.activations[i]._radial_scale(
                z,
                dims=1
            )

            # [S_0, S_1, ..., 1]
            # becomes
            # [S_0 s, S_1 s, ..., s, 1]
            coefficients = torch.cat(
                (
                    coefficients * s,
                    torch.ones_like(s)
                ),
                dim=1
            )

        final = (
            coefficients[:, :1] * temp_weights[-1]
            + coefficients @ self.biases[-1]
        )

        return final