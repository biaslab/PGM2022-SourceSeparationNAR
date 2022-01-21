using LoopVectorization: @turbo

getinput(f)                 = f.input
getoutput(f)                = f.output
getgradientinput(f)         = f.gradient_input
getgradientoutput(f)        = f.gradient_output
getmatinput(f)              = getmat(getinput(f))
getmatoutput(f)             = getmat(getoutput(f))
getmatgradientinput(f)      = getmat(getgradientinput(f))
getmatgradientoutput(f)     = getmat(getgradientoutput(f))
getmat(A::AbstractArray)    = A

setinput!(f, input)                     = f.input = input
setoutput!(f, output)                   = f.output = output
setgradientinput!(f, gradient_input)    = f.gradient_input = gradient_input
setgradientoutput!(f, gradient_output)  = f.gradient_output = gradient_output

function copytoinput!(f, input::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch input
    f_input = getmatinput(f)

    # assert dimensionality
    if check
        @assert axes(f_input) == axes(input)
    end

    # set input
    @turbo f_input .= input

end

function copytoinput!(f, input::T) where { T <: Real }
    
    # set input
    f.input = input

end

function linktoinput!(f, input::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch input
    f_input = getinput(f)

    # assert dimensionality
    if check
        @assert axes(f_input) == axes(input)
    end

    # set input
    setinput!(f, input)

end

function copytooutput!(f, output::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch output
    f_output = getmatoutput(f)

    # assert dimensionality
    if check
        @assert axes(f_output) == axes(output)
    end

    # set output
    @turbo f_output .= output

end

function copytooutput!(f, output::T) where { T <: Real }
    
    # set output
    f.output = output

end

function copytogradientinput!(f, gradient_input::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch gradient input
    f_gradient_input = getmatgradientinput(f)

    # assert dimensionality
    if check
        @assert axes(f_gradient_input) == axes(gradient_input)
    end

    # set gradient input
    @turbo f_gradient_input .= gradient_input

end

function copytogradientinput!(f, gradient_input::T) where { T <: Real }
    
    # set gradient input
    f.gradient_input = gradient_input

end

function copytogradientoutput!(f, gradient_output::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch gradient output
    f_gradient_output = getmatgradientoutput(f)

    # assert dimensionality
    if check
        @assert axes(f_gradient_output) == axes(gradient_output)
    end

    # set gradient output
    @turbo f_gradient_output .= gradient_output

end

function copytogradientoutput!(f, gradient_output::T) where { T <: Real }
    
    # set gradient output
    f.gradient_output = gradient_output

end

function linktogradientoutput!(f, gradient_output::T; check::Bool=true) where { T <: AbstractMatrix }
    
    # fetch gradient output
    f_gradient_output = getmatgradientoutput(f)

    # assert dimensionality
    if check
        @assert axes(f_gradient_output) == axes(gradient_output)
    end

    # set gradient output
    setgradientoutput!(f, gradient_output)

end