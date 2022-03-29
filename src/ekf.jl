function EKF_deployed(signal_mix, model_signal, model_noise, m_x_prior, V_x_prior, R, Q; dim_in=16)

    # allocate output
    m_s_f = zeros(length(signal_mix))
    V_s_f = zeros(length(signal_mix))
    m_n_f = zeros(length(signal_mix))
    V_n_f = zeros(length(signal_mix))

    # initialize intermediate values
    K = zeros(2*dim_in)

    m_x_s = m_x_prior[1:dim_in]
    m_x_n = m_x_prior[1+dim_in:end]

    V_x_ss = V_x_prior[1:dim_in, 1:dim_in]
    V_x_sn = V_x_prior[1:dim_in, 1+dim_in:end]
    V_x_ns = V_x_prior[1+dim_in:end, 1:dim_in]
    V_x_nn = V_x_prior[1+dim_in:end, 1+dim_in:end]
    V_x_new_ss = randn(dim_in, dim_in)
    V_x_new_sn = randn(dim_in, dim_in)
    V_x_new_ns = randn(dim_in, dim_in)
    V_x_new_nn = randn(dim_in, dim_in)

    # kalman filtering
    @inbounds for k in dim_in+1:length(signal_mix)
    
        # run models forward
        m_x_s_new, Jcs = forward_jacobian!(model_signal, m_x_s)
        m_x_n_new, Jcn = forward_jacobian!(model_noise, m_x_n)

        SourceSeparationINN.tri_matmul!(V_x_new_ss, Jcs, V_x_ss, Jcs')
        V_x_new_ss[1,1] += Q[1,1]
        SourceSeparationINN.tri_matmul!(V_x_new_sn, Jcs, V_x_sn, Jcn')
        SourceSeparationINN.tri_matmul!(V_x_new_ns, Jcn, V_x_ns, Jcs')
        SourceSeparationINN.tri_matmul!(V_x_new_nn, Jcn, V_x_nn, Jcn')
        V_x_new_nn[1,1] += Q[1+dim_in, 1+dim_in]

    
        # filtering messages
        y = signal_mix[k] - m_x_s_new[1] - m_x_n_new[1] 
        S = V_x_new_ss[1,1] + V_x_new_sn[1,1] + V_x_new_ns[1,1] + V_x_new_nn[1,1] + R

        # calculate kalman gain
        @turbo for k = 1:dim_in
            K[k] = V_x_new_ss[k,1] + V_x_new_sn[k,1]
            K[k+dim_in] = V_x_new_ns[k,1] + V_x_new_nn[k,1]
        end
        K ./= S

        # update mean
        @turbo for k in 1:dim_in
            m_x_s[k] = m_x_s_new[k] + K[k]*y
            m_x_n[k] = m_x_n_new[k] + K[k+dim_in]*y
        end

        # update variance
        @turbo for m ∈ 1:dim_in, n ∈ 1:dim_in
            V_x_ss[m,n] = V_x_new_ss[m,n] - K[m] * V_x_new_ss[1,n] - K[m] * V_x_new_ns[1,n]
            V_x_sn[m,n] = V_x_new_sn[m,n] - K[m] * V_x_new_sn[1,n] - K[m] * V_x_new_nn[1,n]
            V_x_ns[m,n] = V_x_new_ns[m,n] - K[m+dim_in] * V_x_new_ss[1,n] - K[m+dim_in] * V_x_new_ns[1,n]
            V_x_nn[m,n] = V_x_new_nn[m,n] - K[m+dim_in] * V_x_new_sn[1,n] - K[m+dim_in] * V_x_new_nn[1,n]
        end

        # save values
        m_s_f[k] = m_x_s[1]
        V_s_f[k] = V_x_ss[1,1]
        m_n_f[k] = m_x_n[1]
        V_n_f[k] = V_x_nn[1, 1]

    end


    # return output
    return m_s_f, V_s_f, m_n_f, V_n_f
    
end