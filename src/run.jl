export TrainSuite, run!

using Dates
using ProgressMeter: Progress, next!, ijulia_behavior
using JLD2: save

struct TrainSuite{M,L,T1,T2}
    model       :: M
    loss        :: L
    train_data  :: T1
    test_data   :: T2
    epochs      :: Int64
    log_folder  :: String
end

function run!(train_suite::TrainSuite)

    # fetch date and time
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")

    # change settings for ProgressMeter
    ijulia_behavior(:clear)
    
    # create folder for logging and open files
    mkdir(string(train_suite.log_folder, timestamp))
    logfile_train = string(train_suite.log_folder, timestamp, "/train.csv")
    logfile_test  = string(train_suite.log_folder, timestamp, "/test.csv")

    # print model structure
    print_info(train_suite.model, string(train_suite.log_folder, timestamp, "/model.txt"))

    # log headers
    header = string("epoch,", getid(train_suite.loss), "\n")
    open(logfile_train,"w") do f
        write(f, header)
    end
    open(logfile_test,"w") do f
        write(f, header)
    end

    # loop through epochs
    for epoch in 1:train_suite.epochs

        # run for a single epoch
        run_epoch!(train_suite, epoch, logfile_train, logfile_test)

        # save model
        save(string(train_suite.log_folder, timestamp, "/model.jld2"), "model", train_suite.model)

    end

    return

end

function run_epoch!(train_suite::TrainSuite, epoch, logfile_train, logfile_test)

    # fetch variables
    train_data = train_suite.train_data
    test_data = train_suite.test_data
    model = train_suite.model
    loss_function = train_suite.loss

    # shuffle order of train data
    shuffle!(train_data.list)

    # allocate space for input and output
    input, output = similar(model.memory.input), similar(model.memory.output)

    # start progress meter
    p = Progress(length(train_data)+1, 1, string("epoch ", lpad(epoch, 3, "0"),": "))
    
    # training: loop through signals
    loss_value_train = 0.0
    for (index, signal::Vector{Float64}) in enumerate(train_data)

        # train for signal
        loss_value_train_tmp = train_signal!(model, signal, input, output, loss_function)
        loss_value_train += loss_value_train_tmp

        # log train loss
        open(logfile_train,"a") do f
            write(f, string((epoch-1)+index/length(train_data), ",", loss_value_train_tmp, "\n"))
        end

        # update progress meter
        next!(p; showvalues = [
            (Symbol(string(getid(loss_function), "_train")), loss_value_train_tmp),
        ])

    end

    # testing: loop through signals
    loss_value_test = 0.0
    for signal::Vector{Float64} in test_data

        # train for signal
        loss_value_test += test_signal!(model, signal, input, output, loss_function)

    end

    # normalize losses over number of signals
    loss_value_train /= length(train_data)
    loss_value_test /= length(test_data)

    # log test loss
    open(logfile_test,"a") do f
        write(f, string(epoch, ",", loss_value_test, "\n"))
    end

    # log losses to user
    next!(p; showvalues = [
        (Symbol(string(getid(loss_function), "_train")), loss_value_train),
        (Symbol(string(getid(loss_function), "_test")), loss_value_test),             
    ])

end

function train_signal!(model, signal, input, output, loss_function)

    # fetch info
    (dim_in, batch_size) = size(model.memory.input)

    # specify range
    rng = randperm(length(signal)-2*dim_in*batch_size)
    loss_value_train_tmp = 0.0
    for k in rng
        
        # load data
        # output .= reshape(view(signal, k+dim_in*batch_size:k+2*dim_in*batch_size-1), (dim_in, batch_size))    # dim_in step prediction
        input  .= reshape(view(signal, k+dim_in*batch_size-1:-1:k), (dim_in, batch_size))
        output .= reshape(view(signal, k+dim_in*batch_size:-1:k+1), (dim_in, batch_size))
        # tmp = 0.1*randn(dim_in-1, batch_size)
        # input[1:end-1,:] .+= tmp
        # output[2:end,:] .+= tmp

        # train model
        loss_value_train_tmp += mean(train_batch!(model, input, output, loss_function))
        
    end

    # update global train loss
    loss_value_train_tmp /= length(rng)

    # return loss
    return loss_value_train_tmp

end

function test_signal!(model, signal, input, output, loss_function)

    # fetch info
    (dim_in, batch_size) = size(getmatinput(model))

    # specify range
    rng = 1:length(signal)-2*dim_in*batch_size
    loss_value_test_tmp = 0.0
    for k in rng
        
        # load data
        input  .= reshape(view(signal, k+dim_in*batch_size-1:-1:k), (dim_in, batch_size))
        output .= reshape(view(signal, k+dim_in*batch_size:-1:k+1), (dim_in, batch_size))

        # test model
        loss_value_test_tmp += mean(test_batch!(model, input, output, loss_function))
        
    end

    # update global test loss
    loss_value_test_tmp /= length(rng)

    # return loss
    return loss_value_test_tmp

end

function train_batch!(model, true_input, true_output, loss_function)

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

function test_batch!(model, true_input, true_output, loss_function)

    # run model forward
    predicted_output = forward!(model, true_input)

    # calculate loss
    loss_value = calculate_loss!(loss_function, true_output, predicted_output)

    # return loss
    return loss_value

end