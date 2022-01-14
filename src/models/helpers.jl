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

    # set output
    @inbounds for k in 1:len
        f_output[k] = output[k]
    end

end

function setoutput!(f, output::T) where { T <: Real }
    
    # set output
    f.output = output

end

function setgradientinput!(f, gradient_input::AbstractVector; check::Bool=true)
    
    # fetch gradient input
    f_gradient_input = f.gradient_input

    # assert dimensionality
    len = length(f_gradient_input)
    if check
        @assert len == length(gradient_input)
    end

    # set gradient input
    @inbounds for k in 1:len
        f_gradient_input[k] = gradient_input[k]
    end

end

function setgradientinput!(f, gradient_input::T) where { T <: Real }
    
    # set gradient input
    f.gradient_input = gradient_input

end

function setgradientoutput!(f, gradient_output::AbstractVector; check::Bool=true)
    
    # fetch gradient output
    f_gradient_output = f.gradient_output

    # assert dimensionality
    len = length(f_gradient_output)
    if check
        @assert len == length(gradient_output)
    end

    # set gradient output
    @inbounds for k in 1:len
        f_gradient_output[k] = gradient_output[k]
    end

end

function setgradientoutput!(f, gradient_output::T) where { T <: Real }
    
    # set gradient output
    f.gradient_output = gradient_output

end