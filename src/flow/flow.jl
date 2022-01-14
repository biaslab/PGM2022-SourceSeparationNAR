export FlowModel
export AdditiveCouplingLayer, AdditiveCouplingLayerSplit, PermutationLayer
export ReluFlow

export forward!, propagate_error!, update!

abstract type AbstractModel end
abstract type AbstractLayer end
abstract type AbstractCouplingLayer <: AbstractLayer end
abstract type AbstractFlow end


include("parameter.jl")


include("models/model.jl")

include("layers/permutation_layer.jl")
include("layers/additivecouplinglayer.jl")
include("layers/additivecouplinglayer_split.jl")

include("flows/neural_network/neural_network.jl")
include("flows/reluflow.jl")