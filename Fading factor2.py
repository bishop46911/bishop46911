
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',18)
pd.set_option('display.width', 500)

path = "C:/Users/he_xi/Documents/GitHub/IMC_Prosperity2/data/round-1-island-data-bottle"

round = 1

prices_day_neg2_df = pd.read_csv(filepath_or_buffer = f'{path}/prices_round_{round}_day_-2.csv', index_col='timestamp', sep=';')
prices_day_neg1_df = pd.read_csv(filepath_or_buffer = f'{path}/prices_round_{round}_day_-1.csv', index_col='timestamp', sep=';')
prices_day_0_df = pd.read_csv(filepath_or_buffer = f'{path}/prices_round_{round}_day_0.csv', index_col='timestamp', sep=';')

prices_day_0_nn = pd.read_csv(filepath_or_buffer = f'{path}/trades_round_{round}_day_0_nn.csv', index_col='timestamp', sep=';')
prices_day_1_nn = pd.read_csv(filepath_or_buffer = f'{path}/trades_round_{round}_day_-1_nn.csv', index_col='timestamp', sep=';')
prices_day_2_nn = pd.read_csv(filepath_or_buffer = f'{path}/trades_round_{round}_day_-2_nn.csv', index_col='timestamp', sep=';')

prices_day_0_nnSTAR = prices_day_0_nn[prices_day_0_nn.symbol == 'STARFRUIT']
prices_day_0_nnAMET = prices_day_0_nn[prices_day_0_nn.symbol == 'AMETHYSTS']
prices_day_0_dfSTAR = prices_day_0_df[prices_day_0_df['product'] == 'STARFRUIT']
prices_day_0_dfAMET = prices_day_0_df[prices_day_0_df['product'] == 'AMETHYSTS']

prices_day_0_comb = prices_day_0_nnSTAR.merge(prices_day_0_dfSTAR, left_on='timestamp', right_on='timestamp')
#prices_day_0_comb.columns


'''Estimate fair price using volume weighted price'''
Weight_bid_STAR_col = prices_day_0_dfSTAR[['bid_price_1','bid_price_2','bid_price_3']].to_numpy() * prices_day_0_dfSTAR[['bid_volume_1','bid_volume_2','bid_volume_3']].to_numpy()
Pb = pd.DataFrame(Weight_bid_STAR_col, columns=['Bid1V', 'Bid2V','Bid3V']).sum(axis = 1)
Vb = prices_day_0_dfSTAR[['bid_volume_1','bid_volume_2','bid_volume_3']].sum(axis = 1)
Pb.index = Vb.index
Weight_bid_STAR_df = Pb/Vb
Weight_ask_STAR_col = prices_day_0_dfSTAR[['ask_price_1','ask_price_2','ask_price_3']].to_numpy() * prices_day_0_dfSTAR[['ask_volume_1','ask_volume_2','ask_volume_3']].to_numpy()
Pa = pd.DataFrame(Weight_ask_STAR_col, columns=['Ask1V', 'Ask2V','Ask3V']).sum(axis = 1)
Va = prices_day_0_dfSTAR[['ask_volume_1','ask_volume_2','ask_volume_3']].sum(axis = 1)
Pa.index = Va.index
Weight_ask_STAR_df = Pa/Va
df_STAR_fairP = (Weight_bid_STAR_df+Weight_ask_STAR_df)/2

P_delta = df_STAR_fairP - prices_day_0_nnSTAR.price
P_delta.name = 'P_delta'
prices_day_0_nnSTAR_p = prices_day_0_nnSTAR.merge(P_delta,left_index=True,right_index=True)
minimal_delta_threshold = abs(prices_day_0_nnSTAR_p.P_delta).quantile(0.05)
Fill_Threshold = prices_day_0_nnSTAR_p[abs(prices_day_0_nnSTAR_p.P_delta) < minimal_delta_threshold].quantity.sum()
Fill_total = prices_day_0_nnSTAR_p.quantity.sum()


Lambda0 = Fill_Threshold/Fill_total
Alpha = 1.5
Beta = 0.6

def calculate_lambda(lambda_0,delta,t_elapse,alpha,beta):
    """Calculate time-dependent Poisson rate."""
    return lambda_0 * np.exp(-alpha/beta * delta)*np.exp(-t_elapse)


def logistic_growth(L, k, delta):
    """General logistic growth function for both time and distance adjustments."""
    return L / (1 + np.exp(-k * delta))


def calculate_lambda_adjusted(elapsed_time, distance_to_fair_price, L_t=5, L_d=5, k_t=0.2,  k_d=0.1,):
    """Calculate adjusted lambda based on elapsed time and distance to fair price."""
    f_t = logistic_growth( L_t, k_t, elapsed_time)  # Time adjustment factor
    g_d = logistic_growth( L_d, k_d, distance_to_fair_price,)  # Distance adjustment factor
    return  np.round(f_t + g_d)


def adjust_order_price(order_p, fair_p,t_elapse,lambda_0,alpha,beta, order, adj = 1, Pos_adj = 0.2):
    """Adjust the price of the order based on elapsed time, λ(t), and position size."""
    delta = (fair_p - order_p)
    #Lambda = calculate_lambda(lambda_0,delta,t_elapse,alpha,beta)

    position_adjustment_factor = Pos_adj * order["position_size"]  #1% more aggressive per unit of position size
    #adjustment strategy
    # price_adjustment = adj * Lambda
    price_adjustment = calculate_lambda_adjusted(t_elapse,delta)

    if order["order_type"] == "sell": #position > 0
        fair_new_price = fair_p - price_adjustment - np.floor(position_adjustment_factor)  # Decrease price more for larger positions
    else:  # "buy"  Position < 0
        fair_new_price = fair_p + price_adjustment + np.floor(position_adjustment_factor)# Increase price more for larger positions to fill orders faster

    print(f"Adjusting price of order {order['order_id']} to {fair_new_price}")
    # Example: your_order_system_function_to_adjust_price(order['order_id'], new_price)
    order["limit_price"] = fair_new_price  # Update the order's price

    return order




