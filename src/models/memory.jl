mutable struct Memory{T1,T2,T3,T4}
    input           :: T1
    output          :: T2
    gradient_input  :: T3
    gradient_output :: T4
end
function Memory(dim, batch_size)
    return Memory(zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size))
end
function Memory(dim_in, dim_out, batch_size)
    return Memory(zeros(dim_in, batch_size), zeros(dim_out, batch_size), zeros(dim_in, batch_size), zeros(dim_out, batch_size))
end