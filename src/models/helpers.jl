function setinput!(f, input::AbstractVector; check::Bool=true)
    
    # fetch input
    f_input = f.input

    # assert dimensionality
    len = length(f_input)
    if check
        @assert len == length(input)
    end

    # set input
    @inbounds for k in 1:len
        f_input[k] = input[k]
    end

end

function setinput!(f, input::T) where { T <: Real }
    
    # set input
    f.input = input

end

function setoutput!(f, output::AbstractVector; check::Bool=true)
    
    # fetch output
    f_output = f.output

    # assert dimensionality
    len = length(f_output)
    if check
        @assert len == length(output)
    end

    # set input
    @inbounds for k in 1:len
        f_output[k] = output[k]
    end

end