# BlackScholesEuropeanOption.py

import numpy as np
from math import sqrt, exp
from scipy import stats  # import scipy.stats to compute cummulative density of noraml distribution

#--------------------Parameters---------------------------------------
#stock_price : np.array for stock prices
#vol ： volatility
#t: initial time
#T: expiration time
#interest_rate
#dividend_yield
#strike_price
#---------------------------------------------------------------------

# the computation formula for option price and greeks: https://en.wikipedia.org/wiki/Greeks_(finance)


def European_Call_Payoff(stock_price, strike_price): 
    european_call_payoff = np.maximum(stock_price - strike_price, 0) 
    return european_call_payoff 
 
def European_Put_Payoff(stock_price, strike_price): 
    european_put_payoff = np.maximum(strike_price - stock_price, 0) 
    return european_put_payoff 

def BlackScholesEuropeanCall(t, T, stock_price, strike_price, interest_rate, dividend_yield, vol):
    #compute the Black Scholes European Call
    d1 = (np.log(stock_price / strike_price) + (interest_rate - dividend_yield + 0.5 * vol ** 2) * (T-t)) / (vol * sqrt(T-t))
    d2 = d1 - vol * sqrt(T-t)
    BS_european_call_price = stock_price * exp(-dividend_yield * (T-t)) * stats.norm.cdf(d1, 0.0, 1.0) - strike_price * exp(-interest_rate * (T-t)) * stats.norm.cdf(d2, 0.0, 1.0)
    BS_european_call_delta = exp(-dividend_yield * (T-t)) * stats.norm.cdf(d1,0.0,1.0)
    BS_european_call_gamma = (exp(-dividend_yield * (T-t)) * stats.norm.pdf(d1,0.0,1.0))/(stock_price * vol * sqrt(T-t))
    BS_european_call_theta = (-exp(-dividend_yield * (T-t)) * stock_price * stats.norm.pdf(d1,0.0,1.0) * vol)/(2 * sqrt(T-t))\
                            - interest_rate * strike_price * exp(-interest_rate * (T-t)) * stats.norm.cdf(d2, 0.0, 1.0) + dividend_yield * exp(-dividend_yield * (T-t)) * stock_price * stats.norm.cdf(d1,0.0,1.0)
    BS_european_call_vega = stock_price * exp(-dividend_yield * (T-t))* stats.norm.pdf(d1,0.0,1.0) * sqrt(T-t)
    BS_european_call_rho = strike_price * (T - t) * exp(interest_rate * (T - t)) * stats.norm.cdf(d2, 0.0, 1.0) 
    return BS_european_call_price, BS_european_call_delta, BS_european_call_gamma,\
            BS_european_call_theta, BS_european_call_vega, BS_european_call_rho

def BlackScholesEuropeanPut(t, T, stock_price, strike_price, interest_rate, dividend_yield, vol):
    #compute the Black Scholes European Put
    d1 = (np.log(stock_price / strike_price) + (interest_rate - dividend_yield + 0.5 * vol ** 2) * (T - t)) / (vol * sqrt(T - t))
    d2 = d1 - vol * sqrt(T - t)
    BS_european_put_price = -stock_price * exp(-dividend_yield * (T - t)) * stats.norm.cdf(-d1, 0.0, 1.0) + strike_price * exp(-interest_rate * (T - t)) * stats.norm.cdf(-d2, 0.0, 1.0)
    BS_european_put_delta = -exp(-dividend_yield * (T - t)) * stats.norm.cdf(-d1, 0.0, 1.0)
    BS_european_put_gamma = (exp(-dividend_yield * (T - t)) * stats.norm.pdf(d1, 0.0, 1.0)) / (stock_price * vol * sqrt(T - t))
    BS_european_put_theta = (-exp(-dividend_yield * (T - t)) * stock_price * stats.norm.pdf(d1, 0.0, 1.0) * vol) / ( 2 * sqrt(T - t)) \
                             + interest_rate * strike_price * exp(-interest_rate * (T - t)) * stats.norm.cdf(-d2, 0.0,1.0) - dividend_yield * exp( -dividend_yield * (T - t)) * stock_price * stats.norm.cdf(-d1, 0.0, 1.0)
    BS_european_put_vega = stock_price * exp(-dividend_yield * (T - t)) * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T - t)
    BS_european_put_rho = - strike_price * (T - t) * exp( -interest_rate * (T - t)) * stats.norm.cdf( -d2, 0.0, 1.0) 
    return BS_european_put_price, BS_european_put_delta, BS_european_put_gamma,\
            BS_european_put_theta, BS_european_put_vega, BS_european_put_rho


