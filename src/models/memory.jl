abstract type AbstractMemory end

mutable struct TrainMemory{T1,T2,T3,T4} <: AbstractMemory
    input           :: T1
    output          :: T2
    gradient_input  :: T3
    gradient_output :: T4
end
function TrainMemory(dim, batch_size)
    return TrainMemory(zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size))
end
function TrainMemory(dim_in, dim_out, batch_size)
    return TrainMemory(zeros(dim_in, batch_size), zeros(dim_out, batch_size), zeros(dim_in, batch_size), zeros(dim_out, batch_size))
end

mutable struct DeployMemory{T1,T2,T3,T4,T5} <: AbstractMemory
    input           :: T1
    output          :: T2
    jacobian        :: T3
    jacobian_input  :: T4
    jacobian_output :: T5
end
function DeployMemory(dim_in, dim_out, start_dim)
    return DeployMemory(zeros(dim_in), zeros(dim_out), zeros(dim_in, dim_out), zeros(start_dim, dim_in), zeros(start_dim, dim_out))
end