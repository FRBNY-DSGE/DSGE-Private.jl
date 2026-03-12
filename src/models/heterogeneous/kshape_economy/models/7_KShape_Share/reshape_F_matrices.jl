

function reshape_F_matrices(F1, F2, F3, F4)
     na, nb, nse = get_idiosyncratic_dims(m)

    ## Populate endo using "next-period" name (i.e., using ′) since
    #  it is easier to remove the ′ than to add it
    n_idio_states            = na + nb + nse - 3 # subtract 3 for dof
    dof_to_remove            = 3
    endo[:marginal_pdf_b′_t] = 1:(nb-1)
    endo[:marginal_pdf_a′_t] = nb:(na + nb -2)
    endo[:marginal_pdf_se′_t] = (nb+na-1):(nb_na_nse-3) 


    endo[:marginal_pdf_a′_t] = 1:(na - 1) 
    endo[:marginal_pdf_b′_t] = (1 + na - 1):(na + nb - 2) 
    endo[:marginal_pdf_s′_t] = (1 + na + nb - 2):n_idio_states
    n_dct_copula             = length(get_setting(m,:dct_compression_indices)[:copula])
    n_distr_states           = n_idio_states + n_dct_copula



end
