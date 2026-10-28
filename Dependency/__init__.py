from .GPU import try_gpu
from .TrainingModules import training_loop, training_epoch, testing_epoch

from .Models import (
    # Standard MLP
    NormalLinear,
    ElementwiseTanh,

    # Factorised isotropic MLPs
    FactorisedParallelLinear_IndependentChains,
    FactorisedParallelLinear_RecursiveChains,
    FactorisedParallelLinear_GroupedScalars,
    FactorisedParallelLinear_FinalOptimised,

    # Reparameterised isotropic MLPs
    ReparameterisedParallelLinear_VectorBiases,
    ReparameterisedParallelLinear_RecursiveBiases,
    ReparameterisedParallelLinear_MatrixBiases,
    ReparameterisedParallelLinear_GroupedScalars,
    ReparameterisedParallelLinear_CoefficientMatrix,

    # Independently contracted model
    ContractedParallelLinear,
)

from .RewrittenLayers import (
    IsotropicActivation,
    IsotropicTanh,
)