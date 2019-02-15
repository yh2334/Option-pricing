# FiniteDifferenceMethod.py

import numpy as np

# --------------parameters---------------
# T: expiration time
# t: initial time
# N_size_price
# M_size_time
# interest_rate
# dividend_yiel
# vol : volatility
# strike_price
#-------------------------------------------

# for more information about finite difference method, contact: yh2334@nyu.edu

def EuropeanCallOptionPayoff(stock_price, strike_price):
    # compute the payoff
    european_call_payoff = np.maximum(stock_price-strike_price, 0)
    return european_call_payoff

def EuropeanPutOptionPayoff(stock_price, strike_price):
    # compute the payoff
    european_put_payoff = np.maximum(strike_price-stock_price, 0)
    return european_put_payoff

def Black_Scholes_Explicit_FD(stock_price_max, stock_price_min, N_size_price, M_size_time, t, T, interest_rate, dividend_yield, vol, strike_price, payoff_function):
    # set the parameters
    r = interest_rate  # risk free interest rate
    sigma = dividend_yield  # dividend yield
    delta_s = (stock_price_max - stock_price_min) / N_size_price  # asset value step size
    delta_t = (T - t) / M_size_time  # time step size

    if 0 < delta_t / (delta_s ** 2) < 0.5:
        print('The stability condition holds.')
        print('delta_t = ', delta_t)
        print('delta_s = ', delta_s)

        # Setup Empty numpy Arrays
        value_matrix = np.zeros((N_size_price + 1, M_size_time + 1))
        stock_price = np.linspace(stock_price_min, stock_price_max, N_size_price + 1)
        A = np.zeros((N_size_price - 1, N_size_price - 1))
        b = np.zeros((N_size_price - 1, 1))

        value_matrix[:, 0] = payoff_function(stock_price, strike_price)

        # form the weighting matrix A
        A[0, 0] = 1 - r * delta_t - ((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (delta_s ** 2)
        A[0, 1] = ((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (2 * (delta_s ** 2)) + ((r - sigma) *stock_price[1] * delta_t) / (2 * delta_s)
        for i in range(1, N_size_price - 2):
            A[i, i - 1] = ((vol ** 2) * (stock_price[i + 1] ** 2) * delta_t) / (2 * (delta_s ** 2)) - (( r - sigma) *stock_price[i + 1] * delta_t) / (2 * delta_s)
            A[i, i] = 1 - r * delta_t - ((vol ** 2) * (stock_price[i + 1] ** 2) * delta_t) / (delta_s ** 2)
            A[i, i + 1] = ((vol ** 2) * (stock_price[i + 1] ** 2) * delta_t) / (2 * (delta_s ** 2)) + ((r - sigma) *stock_price[i + 1] * delta_t) / (2 * delta_s)

        A[N_size_price - 2, N_size_price - 3] = ((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (2 * (delta_s ** 2)) - ((r - sigma) * stock_price[-2] * delta_t) / (2 * delta_s)
        A[N_size_price - 2, N_size_price - 2] = 1 - r * delta_t - ((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (delta_s ** 2)

        # interation
        for k in range(1, M_size_time + 1):
            # calculate the matrix b
            b[0, 0] = value_matrix[0, k - 1] * (((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (2 * (delta_s ** 2)) - ((r - sigma) * stock_price[1] * delta_t) / (2 * delta_s))
            b[N_size_price - 2, 0] = value_matrix[N_size_price, k - 1] * (((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (2 * (delta_s ** 2)) + ((r - sigma) * stock_price[-2] * delta_t) / (2 * delta_s))
            value_matrix[1: N_size_price, k] = np.dot(A, value_matrix[1: N_size_price, k - 1]) + b[:, 0]
            # neumann boundary conditions
            value_matrix[N_size_price, k] = 2 * value_matrix[N_size_price - 1, k] - value_matrix[N_size_price - 2, k]
            value_matrix[0, k] = 2 * value_matrix[1, k] - value_matrix[2, k]

    else:
        print('The stability condition does not hold.')
        print('The programme has been terminated')

    return value_matrix[:,-1]


def Black_Scholes_Implicit_FD(stock_price_max, stock_price_min, N_size_price, M_size_time, t, T, interest_rate, dividend_yield, vol, strike_price, payoff_function):
    # set the parameters
    r = interest_rate  # risk free interest rate
    sigma = dividend_yield  # dividend yield
    delta_s = (stock_price_max - stock_price_min) / N_size_price  # asset value step size
    delta_t = (T - t) / M_size_time  # time step size

    # Setup Empty numpy Arrays
    value_matrix = np.zeros((N_size_price + 1, M_size_time + 1))
    stock_price = np.linspace(stock_price_min, stock_price_max, N_size_price + 1)
    A = np.zeros((N_size_price - 1, N_size_price - 1))
    b = np.zeros((N_size_price - 1, 1))

    # Evaluate Terminal Value for Calls or Puts
    value_matrix[:, 0] = payoff_function(stock_price, strike_price)

    # form the weighting matrix
    A[0, 0] = 1 + r * delta_t + ((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (delta_s ** 2)
    A[0, 1] = -((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (2 * (delta_s ** 2)) - ((r - sigma) *stock_price[1] * delta_t) / (2 * delta_s)
    for i in range(1, N_size_price - 2):
        A[i, i - 1] = -((vol ** 2) * (stock_price[i + 1] ** 2) * delta_t) / (2 * (delta_s ** 2)) + (( r - sigma) *stock_price[i + 1] * delta_t) / (2 * delta_s)
        A[i, i] = 1 + r * delta_t + ((vol** 2) * (stock_price[i + 1] ** 2) * delta_t) / (delta_s ** 2)
        A[i, i + 1] = -((vol ** 2) * (stock_price[i + 1] ** 2) * delta_t) / (2 * (delta_s ** 2)) - (( r - sigma) *stock_price[i + 1] * delta_t) / (2 * delta_s)
    A[N_size_price - 2, N_size_price - 3] = -((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (2 * (delta_s ** 2)) + ((r - sigma) * stock_price[-2] * delta_t) / (2 * delta_s)
    A[N_size_price - 2, N_size_price - 2] = 1 + r * delta_t + ((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (delta_s ** 2)

    for k in range(1, M_size_time + 1):
        b[0, 0] = -value_matrix[0, k - 1] * ( -((vol ** 2) * (stock_price[1] ** 2) * delta_t) / (2 * (delta_s ** 2)) + ((r - sigma) * stock_price[1] * delta_t) / (2 * delta_s))
        b[N_size_price - 2, 0] = -value_matrix[N_size_price, k - 1] * (-((vol ** 2) * (stock_price[-2] ** 2) * delta_t) / (2 * (delta_s ** 2)) - ((r - sigma) * stock_price[-2] * delta_t) / (2 * delta_s))
        value_matrix[1: N_size_price, k] = np.linalg.inv(A).dot(value_matrix[1: N_size_price, k - 1] + b[ :,0])
        # neumann boundary conditions
        value_matrix[N_size_price, k] = 2 * value_matrix[N_size_price - 1, k] - value_matrix[N_size_price - 2, k]
        value_matrix[0, k] = 2 * value_matrix[1, k] - value_matrix[2, k]

    
    return value_matrix[:,-1]


