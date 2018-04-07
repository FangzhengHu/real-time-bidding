import matplotlib.pyplot as plt
import numpy as np

## ------------------------- Bidding Strategies ------------------------- ##
def linear_bid(base_bid, p_ctr, avg_ctr):
    """
    bid = base_bid * p_ctr / avg_ctr
    """
    
    return base_bid * p_ctr / avg_ctr



def round_bid(base_bid, p_ctr, avg_ctr):

    return np.round_(base_bid * p_ctr / avg_ctr)



def square_root_bid(base_bid, p_ctr, avg_ctr):
    """
    bid = base_bid * (p_ctr / avg_ctr)^(1/2)
    """
    return base_bid * np.sqrt(p_ctr / avg_ctr)



## ------------------- Bidding Strategies Evaluation -------------------- ##
def calculate_click_numb_slow(bids, slots):
    """
    # calculate number of clicks a bidding portfolio generates
    # Input:
    # @bids[array]: bidding price for each ad-slot
    # @slots[df]: payprice and click for each ad-slot
    """
    
    budget = 6250 * 1000
    accum_click = 0
    
    for i, row in slots.iterrows():
        bid = bids[i]
        # win if payprice no larger than the bidding
        if row.payprice <= bid:
            budget -= row.payprice
            accum_click += row.click
        if budget <= 0:
            break
    
    return accum_click, budget<=0


def count_click_numb(bidprice, payprice, click, budget=6250 * 1000):
    
    accum_click = 0
    accum_imp = 0
    
    for i in range(len(bidprice)):
        # win if payprice no larger than the bidding
        if payprice[i] <= bidprice[i]:
            budget -= payprice[i]
            accum_imp += 1
            accum_click += click[i]
        if budget <= 0:
            break
            
    return accum_click, accum_imp, budget, i



def stats(ls):
    
    print('mean: {}, min: {}, max: {}'.format(
    np.mean(ls), np.min(ls), np.max(ls)))
    
    return None


def plot_interval_search(xs, ys):
    
    plt.plot(xs, ys)
    plt.xlabel('search value')
    plt.ylabel('number of clicks gained')
    plt.title('visualization of interval search')
    
    return None
    





