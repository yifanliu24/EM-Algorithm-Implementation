import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt
from str2bool import str2bool
num_K = int(sys.argv[1])  # number of gaussian models to find
show_plot = str2bool(sys.argv[2])  # flag to show plot

my_file = open("em_data.txt", "r")  # read the file
content_list = my_file.read()
content_list = content_list.split("\n")  # make the content a list and split on the newlines
content_list = content_list[0:-1]  # cut off the final blank line
float_list = [float(i) for i in content_list]  # convert to a float list
float_np = np.array(float_list)  # convert float list to a numpy array


def main():
    if num_K == 5:  # warning for time
        print("Warning: May take some time to reach local optima")
    if show_plot:  # showing 1d plot of existing data
        plt.plot(float_np, np.zeros_like(float_np), 'x')
        plt.show()
    tolerance = 10 ** -8  # tolerance for convergence
    mu_tol_check = 1  # init tolerances for the parameters
    vars_tol_check = 1
    alphas_tol_check = 1
    mus, vars, alphas = init_vars(num_K)  # init random parameters for step 2
    while mu_tol_check > tolerance or vars_tol_check > tolerance or alphas_tol_check > tolerance:  # check for convergence
        new_mus, new_vars, new_alphas = generate_clusters(mus, vars, alphas, num_K)  # calc new parameters using EM
        mu_tol_check = abs(sum(mus) - sum(new_mus)) / sum(mus)  # find relative change for tolerance check on params
        vars_tol_check = abs(sum(vars) - sum(new_vars)) / sum(vars)
        alphas_tol_check = abs(sum(alphas) - sum(new_alphas)) / sum(alphas)
        mus = new_mus  # set params as new for next loop
        vars = new_vars
        alphas = new_alphas
    print("Means (step 2): ", mus)  # report params
    print("Variances (step 2): ", vars)
    print("Alphas (step 2): ", alphas)
    likelihood = calc_L(mus, vars, alphas)  # calc the log likelihood
    print("Log Likelihood (step 2): ", likelihood)  # report log likelihood
    mus_3, vars_3, alphas_3 = init_vars(num_K)  # init random parameters for step 3
    vars_3 = [1 for i in vars_3]
    mu_tol_check_3 = 1  # init tolerances for the parameters
    alphas_tol_check_3 = 1
    while mu_tol_check_3 > tolerance or alphas_tol_check_3 > tolerance:  # check for convergence
        new_mus_3, new_vars_3, new_alphas_3 = modified_generate_clusters(mus_3, vars_3, alphas_3, num_K)  # calc new parameters using EM
        mu_tol_check_3 = abs(sum(mus_3) - sum(new_mus_3)) / sum(mus_3)  # find relative change for tolerance check on params
        alphas_tol_check_3 = abs(sum(alphas_3) - sum(new_alphas_3)) / sum(alphas_3)
        mus_3 = new_mus_3  # set params as new for next loop
        alphas_3 = new_alphas_3
    print("Means (step 3): ", mus_3)  # report params
    print("Variances (step 3): ", vars_3)
    print("Alphas (step 3): ", alphas_3)
    likelihood_3 = calc_L(mus_3, vars_3, alphas_3)  # calc the log likelihood
    print("Log Likelihood (step 3): ", likelihood_3)  # report log likelihood



# function name: calc_pr
# arguments: numpy array x with input data, scalar mean, scalar standard deviation
# purpose: calculates the probability that each data point is in a given model. Used in E step
# returns: the numpy array of probabilities for each data point
def calc_pr(x, mu, sigma):
    pr = np.exp((((x-mu)/sigma)**2)/-2)/(sigma*math.sqrt(2*math.pi))
    return pr

# function name: init_vars
# arguments: a scalar
# purpose: initializes the parameters with random values in the bounds of the data. makes sure alpha adds to 1
# returns: the random initialized parameters in their own lists
def init_vars(K):
    list_max = max(float_np)  # find the max and min of the data
    list_min = min(float_np)
    mu_list = []  # init empty lists
    var_list = []
    alpha_list = []
    for i in range(K):
        mu_list.append(random.uniform(list_min, list_max))  # choose randomly between bounds
        var_list.append(random.uniform(list_min, list_max))
        alpha_list.append(random.uniform(0, 1-np.sum(alpha_list)))  # choose randomly between 0 and 1, and account for sum
    alpha_list[K-1] = 1-(np.sum(alpha_list)-alpha_list[K-1])  # make sure the list adds up to 1 by changing last ele
    return mu_list, var_list, alpha_list

# function name: calc_wiks
# arguments: the list of means, variances, and alphas and the scalar k indicating which model is being evaluated
# purpose: finds the w_iks per the prompt
# returns: a numpy array of the w_iks for the data array
def calc_wiks(mus, vars, alphas, k):
    pr_k = calc_pr(float_np, mus[k], math.sqrt(vars[k]))  # calc the probabilities that these data are in model k
    numerator = pr_k*alphas[k]  # multiply by alpha of model k
    denominator = 0
    for i in range(len(alphas)):  # find the denominator of the wik equation
        this_pr = calc_pr(float_np, mus[i], math.sqrt(vars[i]))
        denominator += this_pr*alphas[i]
    wik_np = np.divide(numerator, denominator)  # element wise division of each numpy element
    return wik_np

# function name: calc_nk
# arguments: the numpy array of the wi's for model k
# purpose: sums the wi's for model k
# returns: a scalar nk
def calc_nk(wik_list):
    nk = np.sum(wik_list)
    return nk

# function name: calc_alpha_k
# arguments: the scalar nk
# purpose: finds the new proportion alpha for model k
# returns: the new scalar alpha for model k
def calc_alpha_k(nk):
    alpha_k = nk/len(float_list)
    return alpha_k

# function name: calc_mu_k
# arguments: the array of wi's for model k, the scalar nk, the array of data
# purpose: finds the mean for model k
# returns: the new mean as a scalar for model k
def calc_mu_k(wik_list, nk, x_np):
    new_mu_k = np.sum(np.multiply(wik_list, x_np))/nk
    return new_mu_k

# function name: calc_var_k
# arguments: the array of wi's for model k, the scalar nk, the array of data, the new mean for model k
# purpose: finds the variance for model k
# returns: the new variance as a scalar for model k
def calc_var_k(wik_list, nk, x_np, mu_k):
    sq_term = np.square(x_np-mu_k)
    var_k = np.sum(np.multiply(wik_list, sq_term))/nk
    return var_k

# function name: generate_clusters
# arguments: the list of means, variances, and alphas as well as the scalar for the number of clusters
# purpose: performs one iteration of parameter updates via the EM algorithm
# returns: the lists of new parameters
def generate_clusters(mus, vars, alphas, clusters):
    new_mus = []  # init blank lists for the new parameters
    new_vars = []
    new_alphas = []
    for i in range(clusters):  # perform this iteration for each cluster
        # E step
        these_wiks = calc_wiks(mus, vars, alphas, i)  # calc new parameters and add to empty lists
        # M step
        this_nk = calc_nk(these_wiks)
        this_alpha_k = calc_alpha_k(this_nk)
        this_mu_k = calc_mu_k(these_wiks, this_nk, float_np)
        this_var = calc_var_k(these_wiks, this_nk, float_np, this_mu_k)
        new_mus.append(this_mu_k)
        new_alphas.append(this_alpha_k)
        new_vars.append(this_var)
    return new_mus, new_vars, new_alphas

# function name: calc_L
# arguments: the list of means, variances, and alphas
# purpose: calculates the log likelihood of the data given the parameters of the models
# returns: scalar log_likelihood
def calc_L(mus_in, vars_in, alphas_in):
    summation = 0
    for i in range(len(float_np)):
        internal_sum = 0
        data_point = float_np[i]
        for j in range(len(alphas_in)):
            this_PR = math.exp((((data_point - mus_in[j]) / math.sqrt(vars_in[j])) ** 2) / -2) /\
                (math.sqrt(vars_in[j]) * math.sqrt(2 * math.pi))
            internal_sum += alphas_in[j]*this_PR
        summation += math.log(internal_sum)
    log_likelihood = summation
    return log_likelihood


# function name: generate_clusters
# arguments: the list of means, variances, and alphas as well as the scalar for the number of clusters
# purpose: performs one iteration of parameter updates via the EM algorithm
# returns: the lists of new parameters
def modified_generate_clusters(mus, vars, alphas, clusters):
    new_mus = []  # init blank lists for the new parameters
    new_alphas = []
    for i in range(clusters):  # perform this iteration for each cluster
        # E step
        these_wiks = calc_wiks(mus, vars, alphas, i)  # calc new parameters and add to empty lists
        # M step
        this_nk = calc_nk(these_wiks)
        this_alpha_k = calc_alpha_k(this_nk)
        this_mu_k = calc_mu_k(these_wiks, this_nk, float_np)
        new_mus.append(this_mu_k)
        new_alphas.append(this_alpha_k)
    return new_mus, vars, new_alphas



main()
