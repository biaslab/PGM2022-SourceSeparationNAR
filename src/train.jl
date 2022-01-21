export train!

function train!(model, true_input, true_output, loss_function)

    # run model forward
    predicted_output = forward!(model, true_input)

    # calculate loss
    loss_value = calculate_loss!(loss_function, true_output, predicted_output)

    # calculate gradient of loss
    calculate_dloss!(loss_function, getmatgradientoutput(model), true_output, predicted_output)

    # propagate gradient
    propagate_error!(model)

    # update weights in model
    update!(model)

    # return loss
    return loss_value

end