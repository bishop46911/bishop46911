from typing import Dict, List, Any
import math
from math import erf
import string
import json
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
# storing string as const to avoid typos
SUBMISSION = "SUBMISSION"

AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"

CHOCOLATE = "CHOCOLATE"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
GIFT_BASKET = "GIFT_BASKET"

COCONUT = "COCONUT"
COCONUT_COUPON = "COCONUT_COUPON"

PRODUCTS_GIFT = [
    GIFT_BASKET,
    CHOCOLATE,
    STRAWBERRIES,
    ROSES
]

PRODUCTS = [
    AMETHYSTS,
    STARFRUIT,
    ORCHIDS,
    CHOCOLATE,
    STRAWBERRIES,
    ROSES,
    GIFT_BASKET,
    COCONUT,
    COCONUT_COUPON
]

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
    #ORCHIDS seems to be very volatile
    ORCHIDS: 1_100,
    COCONUT_COUPON: 637.63
}

STORAGE_FEE = {
    ORCHIDS: 0.1 #per timestamp
}

GIFT_BASKET_WEIGHTS = {
    CHOCOLATE: 4,
    STRAWBERRIES: 6,
    ROSES: 1
}

HISTORICAL_ROUNDS = 3 * (100_000 / 100)

#BS model
#initial vol 0.16 is solved using: fit_vol(0.1,10000,637.63), assuming at the money strike at time 0, with the given option premium at time 0
def fun_BS_quick(S = 10000, K = 10000, vol = 0.16, T = 1, r = 0, q = 0, ReturnDelta = False):

    d1 = (np.log(S/K)+ (r+vol**2/2)*T)/vol/np.sqrt(T)
    d2 = d1 - vol*np.sqrt(T)

    normcdf = lambda x: (1 + erf(x/np.sqrt(2)))/2
    N1 = normcdf(d1)
    N2 = normcdf(d2)

    px = S*N1 - K*np.exp((q-r)*T)*N2

    if ReturnDelta:
        return N1
    else:
        return px

def fit_vol(vol_fit = 0.10, S = 9990, px = 620.5, T = 1, step = 0.0001):
    for i in range(30):
        px_new = fun_BS_quick(S=S,vol = vol_fit, T=T)
        #print('px_new',px_new)
        #print('px',px)
        if abs(px_new-px)<0.01:
            #print(px,px_new)

            break
        vol_fit = vol_fit + (px - px_new)*step
    return vol_fit

def fit_implied_volatility(S, K, T, r, market_price, sigma_init=0.1, learning_rate=0.01, tolerance=1e-6,
                           max_iterations=10000):
    """
    Fit the implied volatility for a European call option using gradient descent.

    Parameters:
    - S (float): Stock price.
    - K (float): Strike price.
    - T (float): Time to maturity (in years).
    - r (float): Risk-free interest rate.
    - market_price (float): Observed market price of the option.
    - sigma_init (float): Initial guess for the volatility.
    - learning_rate (float): Step size for gradient descent.
    - tolerance (float): Convergence tolerance.
    - max_iterations (int): Maximum number of iterations for the gradient descent.

    Returns:
    - float: Fitted implied volatility.
    """
    normcdf = lambda x: (1 + erf(x/np.sqrt(2)))/2
    # Black-Scholes price calculation function for a call option
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * normcdf(d1) - K * np.exp(-r * T) * normcdf(d2)

    # Loss function: squared difference between market and model prices
    def loss_function(sigma):
        model_price = black_scholes_call(S, K, T, r, sigma)
        return (model_price - market_price) ** 2

    # Initialize sigma
    sigma = sigma_init

    # Gradient descent loop
    for _ in range(max_iterations):
        # Calculate the current model price and loss
        current_loss = loss_function(sigma)

        # Compute gradient: numerical derivative of the loss function
        sigma_increment = 0.0001
        gradient = (loss_function(sigma + sigma_increment) - current_loss) / sigma_increment

        # Update sigma based on the gradient
        sigma -= learning_rate * gradient

        # Convergence check
        if np.abs(gradient) < tolerance:
            break

    return sigma


# # Example usage
# S = 100  # Stock price
# K = 100  # Strike price
# T = 1  # Time to maturity in years
# r = 0.05  # Risk-free rate
# market_price = 10  




class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:

    def __init__(self) -> None:
        
        #print("Initializing Trader...")

        self.position_limit = {
            AMETHYSTS : 20,
            STARFRUIT : 20,
            ORCHIDS: 100,
            CHOCOLATE : 250,
            STRAWBERRIES : 350,
            ROSES: 60,
            GIFT_BASKET: 60,
            COCONUT: 300,
            COCONUT_COUPON: 600
        }

        #keep track of each round
        self.round = 0
        # positions can be obtained from state.position
        
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # optional for EMA based srtategy - self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = 0

        # typically it equals 2/(N+1) where N is the period of EMA
        self.ema_param = 0.1

        #based on training set
        self.edge_max = 252.5

        self.price_chocolate_mean = 7915.34725
        self.price_strawberries_mean = 4026.83735
        self.price_roses_mean = 14506.89705
        self.price_gift_basket_mean = 70708.80063333333

        self.mean_spread_starfruit = None
        self.mean_spread_orchids = None

        #may update with historical prices later
        self.mean_spread_gb = abs(-10.864)
        self.mean_spread_chocolate = abs(-1.3888333333333334)
        self.mean_spread_strawberries = abs(-1.3869)
        self.mean_spread_roses = abs(-1.2893)

        self.mean_edge_hist = 380
        self.mean_edge = 380
        self.std_edge = 76.42
        #self.std_edge = 390.93
        #min threshold to trigger trades
        self.std_edge_coeff = 0.8

        # self.ema_periods = 200
        # self.coco_alpha = 2/(self.ema_periods+1)
        # self.price_coconut_mean = 9999.9009
        # self.price_coupon_mean = 635.0464

        self.annual_trading_days = 252

        self.last_coconut_mid_price, self.last_coupon_mid_price = 9882.5, 575.5
        self.last_coconut_return = -0.000152
        
        # TODO - may change it to the 3-day avg IV which 0.1591(no big diff)
        self.iv_initial = 0.16
        # based on historical returns in 3-day
        self.iv_ema = 0.160209
        # typically it equals 2/(N+1) where N is the period of EMA
        self.iv_ema_period = 20
        self.iv_ema_param = 2 / (self.iv_ema_period + 1)
        self.iv_ema_weight = 0.4
        self.iv_min = 0
        self.iv_max = 0

        #annualized already (df_all_mid_prices_day_123['Returns'].ewm(span=N, adjust=False).mean()*252*100)[-1]
        self.his_ret_ema_period1 = -0.499042
        self.his_ret_ema_period2 = -0.167494
        self.his_ret_ema_period3 = -0.063089

        #annualized already (df_all_mid_prices_day_123['Returns'].ewm(span=N, adjust=False).std()*np.sqrt(252)*100)[-1]
        self.hv_ema_period1 = 0.169063
        self.hv_ema_period2 = 0.159894
        self.hv_ema_period3 = 0.155075
        self.hv_period1 = 21
        self.hv_period2 = 63
        self.hv_period3 = 126
        self.hv_ema_period1_param = 2 / (self.hv_period1 + 1)
        self.hv_ema_period2_param = 2 / (self.hv_period2 + 1)
        self.hv_ema_period3_param = 2 / (self.hv_period3 + 1)

        self.hv_period1_weight = 0.2
        self.hv_period2_weight = 0.2
        self.hv_period3_weight = 0.2

        self.iv_quantile_threshold = 0.5

        self.coupon_default_positon = 550

        #from df_all_mid_prices_day_123['Implied_Volatility'].quantile(0.1) to 0.4, increment by 0.025
        self.iv_10_40_quantiles = [0.1548126483102763, 0.15524564478925557, 0.15559389236534613, 0.15593730205783163,
            0.15627209500721972, 0.15654949361399448, 0.15680083271943834, 0.157015403816229,
            0.157224422027324, 0.15743515099803368, 0.15766677190957712, 0.15787996694243617,
            0.15811622787186605]
        
        #from df_all_mid_prices_day_123['Implied_Volatility'].quantile(0.6) to 0.9, increment by 0.025
        self.iv_60_90_quantiles = [0.15992161636675647, 0.16015348901861012, 0.16038658352235557, 0.16064647332822493,
            0.16092613458422378, 0.1612014438414916, 0.16148954535368149, 0.1618132410062684,
            0.16213397937019522, 0.16248968170571057, 0.1628304893697475, 0.16321238977320887,
            0.16371509536720638]
        
        #self.iv_10_40_quantiles_length, self.iv_60_90_quantiles_length = 13, 13




    # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0) 
    
    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]


        # if default_price is None:
        #     default_price = DEFAULT_PRICES[product]

        # if product not in state.order_depths:
        #     return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = list(dict(sorted(market_bids.items(), reverse=True)).items())[0][0]
        best_ask = list(dict(sorted(market_asks.items())).items())[0][0]
        return (best_bid + best_ask)/2
    
    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of dollar value currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)
    
    def logistic_growth(self, L, k, diff):
        """General logistic growth function for both time and distance adjustments."""
        return L / (1 + np.exp(-k * diff))


    def calculate_lambda_adjusted(self, distance_to_fair_price, position_max_limit, position_cur, volume_change
                                        , L_d=0.5, P_d=1, k_d=0.5, k_p=1, L_v=0.1, k_v=0.5):
        """Calculate adjusted lambda based on elapsed time and distance to fair price, position."""
        #It’s very hard to implement elapsed_time as the Order object we sent has no timestamp
        if position_cur >= 0:
            position_mid = np.round(position_max_limit/2)
        else:
            position_mid = -1 * np.round(position_max_limit/2)

        #f_t = self.logistic_growth(L_t, k_t, elapsed_time)  # Time adjustment factor
        g_d = self.logistic_growth(L_d, k_d, distance_to_fair_price)  # Distance adjustment factor
        h_p = self.logistic_growth(P_d, k_p, position_cur - position_mid)   #adjustment based on position, capped at 20, half way adjustmnt when position at 10
        v_p = self.logistic_growth(L_v, k_v, volume_change) # Volume difference factor(abs diff on best_bid_vol and best_ask_vol)
        

        return  np.round(g_d + h_p + v_p)

    def calculate_lambda_adjusted_r3(self, price_distance, required_pos
                                        , L_d=2, k_d=0.05
                                        , L_p=2, k_p=0.05):
        """Calculate adjusted lambda based on elapsed time and distance to fair price, position."""
        #It’s very hard to implement elapsed_time as the Order object we sent has no timestamp
        #2 / (1 + exp(-0.05 * 50)) = 1.8482
        g_d = self.logistic_growth(L_d, k_d, abs(price_distance))   #adjustment based on position
        h_p = self.logistic_growth(L_p, k_p, abs(required_pos))   #adjustment based on position

        return  np.round(g_d + h_p)
    
    def update_ema_vol(self, vol_type, cur_vol):
        """
        Update the exponential moving average of the volatility(implied vol or realized vol).
        alpha: set a defined parameter other than the class variable self.ema_param
        """
        if vol_type=='IV':
        # do ema update if mid_price is not none
            if cur_vol is not None:
                # ema update - base case
                if self.iv_ema is None:
                    self.iv_ema = cur_vol
                # ema update - recursive case
                else:
                    self.iv_ema = self.iv_ema_param * cur_vol + (1-self.iv_ema_param) * self.iv_ema

        elif vol_type=='HV_Period1':
            if cur_vol is not None:
                if self.hv_ema_period1 is None:
                    self.hv_ema_period1 = cur_vol
                else:
                    self.hv_ema_period1 = self.hv_ema_period1_param * cur_vol + (1-self.hv_ema_period1_param) * self.hv_ema_period1

        elif vol_type=='HV_Period2':
            if cur_vol is not None:
                if self.hv_ema_period2 is None:
                    self.hv_ema_period2 = cur_vol
                else:
                    self.hv_ema_period2 = self.hv_ema_period2_param * cur_vol + (1-self.hv_ema_period2_param) * self.hv_ema_period2

        elif vol_type=='HV_Period3':
            if cur_vol is not None:
                if self.hv_ema_period3 is None:
                    self.hv_ema_period3 = cur_vol
                else:
                    self.hv_ema_period3 = self.hv_ema_period3_param * cur_vol + (1-self.hv_ema_period3_param) * self.hv_ema_period3

        else:
            print('support vol_type IV or HV_Period1, HV_Period2, HV_Period3 at the moment')
            return None

    def update_ema_prices(self, product, state : TradingState, alpha=None):
        """
        Update the exponential moving average of the prices of each product.
        alpha: set a defined parameter other than the class variable self.ema_param
        """
        mid_price = self.get_mid_price(product, state)
        # do ema update if mid_price is not none
        if mid_price is not None:
            # ema update - base case
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            # ema update - recursive case
            else:
                if alpha is None:
                    self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]
                else:
                    self.ema_prices[product] = alpha * mid_price + (1-alpha) * self.ema_prices[product]

    def update_mean_value(self, mean_val, val, prev_n):
        """
        keep a rolling mean_val for each incoming val
        """
        if prev_n<0:
            return None
        if mean_val is None:
            mean_val = val
        new_mean_val = (mean_val * prev_n + val) / (prev_n + 1)

        return new_mean_val
    
    #maintain two list to keep 40 and 60 quantiles of iv values
    def percentile_two_queues(self, new_val):
        if new_val<max(self.iv_10_40_quantiles):
                _ = self.iv_10_40_quantiles.pop(0)
                self.iv_10_40_quantiles.append(new_val)
                self.iv_10_40_quantiles = sorted(self.iv_10_40_quantiles)

        elif new_val>min(self.iv_60_90_quantiles):
                _ = self.iv_60_90_quantiles.pop(-1)
                self.iv_60_90_quantiles.append(new_val)
                self.iv_60_90_quantiles = sorted(self.iv_60_90_quantiles)

        else:
            pass



    
    # TODO - think of a more dynamic way to update percentile
    # the current method will flunctuate too much if total min max range changes very rapidly
    def percentile_within_minmax(self, cur_val, min_val, max_val):
        if min_val==max_val:
            return np.nan  # Not enough data to compute a range and percentile
        min_val = min(min_val, cur_val)
        max_val = max(max_val, cur_val)

        # Normalize last value within the min-max range
        # Avoid division by zero if min and max are the same
        if max_val != min_val:
            percentile = (cur_val - min_val) / (max_val - min_val) * 100
        else:
            percentile = 0  # All values are the same in the window

        #update min_val and max_val based on cur_val if cur_val is min or max
        if min_val==cur_val:
            self.iv_min = cur_val
        if max_val==cur_val:
            self.iv_max = cur_val

        return percentile

    def compute_vwap(self, orders_dict: dict, top_n=3):
        topn_orders_lst = list(orders_dict.items())[:top_n]
        weighted_sum = sum([int(price) * int(vol) for price, vol in topn_orders_lst])
        total_vol = sum([int(vol) for _, vol in topn_orders_lst])
        vwap = weighted_sum / total_vol

        return vwap
    
    def compute_vwamp_bidask(self, bid_orders_dict: dict, ask_orders_dict: dict):
        weighted_bid = self.compute_vwap(bid_orders_dict)
        weighted_ask = self.compute_vwap(ask_orders_dict)

        return (weighted_bid + weighted_ask)/2
    
    def amethysts_strategy(self, state : TradingState):
        '''
        Returns a list of orders with trades of amethysts.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(AMETHYSTS, state)

        order_depth: OrderDepth = state.order_depths[AMETHYSTS]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        # no fading of fair_price based on cur_position, self.position_limit[AMETHYSTS]
        fair_price = DEFAULT_PRICES[AMETHYSTS]
        #fair_price = self.ema_prices[AMETHYSTS]

        #for sell order, their vol are in negative
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < fair_price) or ((cur_position<0) and (ask == fair_price))) \
                                                                and cur_position < self.position_limit[AMETHYSTS]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[AMETHYSTS] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(AMETHYSTS, ask, order_for))


        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, fair_price + 1))

        
        if cur_position < self.position_limit[AMETHYSTS]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[AMETHYSTS], self.position_limit[AMETHYSTS] - cur_position)
            orders.append(Order(AMETHYSTS, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(AMETHYSTS, state)
    
        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > fair_price) or ((cur_position>0) and (bid == fair_price))) \
                                                        and cur_position > -self.position_limit[AMETHYSTS]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[AMETHYSTS]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(AMETHYSTS, bid, order_for))


        if cur_position > -self.position_limit[AMETHYSTS]:
            num = max(-2*self.position_limit[AMETHYSTS], -self.position_limit[AMETHYSTS]-cur_position)
            orders.append(Order(AMETHYSTS, sell_pr, num))
            cur_position += num

        return orders
    
    def starfruit_strategy(self, state : TradingState, n_round : int):
        '''
        Returns a list of orders with trades of starfruit.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(STARFRUIT, state)

        order_depth: OrderDepth = state.order_depths[STARFRUIT]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[STARFRUIT]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        

        max_position_limit = self.position_limit[STARFRUIT]

        price_adjustment = self.calculate_lambda_adjusted(distance_to_fair_price=fair_price_distance
                                                          , position_max_limit=max_position_limit
                                                          , position_cur=cur_position
                                                          , volume_change=diff_best_bid_ask
                                                          , L_d=0.5, P_d=1
                                                          , k_d=0.5, k_p=1
                                                          , L_v=0.1, k_v=0.5)

        #we add price adjustment at most 3, otherwise the loss would explode
        #3 is mid point of the avg best_bid, best_ask which is -6 based on data exploration
        #half_mean_spread = 3

        #have a dynamic half_mean_spread based on real data
        best_bid_ask_spread = abs(best_ask - best_bid)
        half_mean_spread = np.round(0.5 * self.update_mean_value(mean_val=self.mean_spread_starfruit 
                                                        ,val=best_bid_ask_spread, prev_n=HISTORICAL_ROUNDS+n_round))

        price_adjustment = min(half_mean_spread, price_adjustment)
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        #logger.print(f'orig_fair_price: {orig_fair_price}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[STARFRUIT]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[STARFRUIT] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(STARFRUIT, ask, order_for))
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        
        if cur_position < self.position_limit[STARFRUIT]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[STARFRUIT], self.position_limit[STARFRUIT] - cur_position)
            orders.append(Order(STARFRUIT, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(STARFRUIT, state)

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[STARFRUIT]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[STARFRUIT]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(STARFRUIT, bid, order_for))


        if cur_position > -self.position_limit[STARFRUIT]:
            num = max(-2*self.position_limit[STARFRUIT], -self.position_limit[STARFRUIT]-cur_position)
            orders.append(Order(STARFRUIT, sell_pr, num))
            cur_position += num

        return orders
    
    def orchids_strategy(self, state : TradingState, n_round : int, observations : Observation):
        '''
        Returns a list of orders with trades of orchids.
        '''
        observ = observations.conversionObservations[ORCHIDS]

        orders: list[Order] = []
        cur_position = self.get_position(ORCHIDS, state)

        order_depth: OrderDepth = state.order_depths[ORCHIDS]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[ORCHIDS]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        
        max_position_limit = self.position_limit[ORCHIDS]

        #no price_adjustment for now
        price_adjustment = self.calculate_lambda_adjusted(distance_to_fair_price=fair_price_distance
                                                          , position_max_limit=max_position_limit
                                                          , position_cur=cur_position
                                                          , volume_change=diff_best_bid_ask
                                                          , L_d=0.5, P_d=1
                                                          , k_d=0.5, k_p=1
                                                          , L_v=0.1, k_v=0.5)

        #we add price adjustment at most 3, otherwise the loss would explode
        #3 is mid point of the avg best_bid, best_ask which is -6 based on data exploration
        #half_mean_spread = 3

        #have a dynamic half_mean_spread based on real data
        best_bid_ask_spread = abs(best_ask - best_bid)
        half_mean_spread = np.round(0.5 * self.update_mean_value(mean_val=self.mean_spread_orchids
                                                ,val=best_bid_ask_spread, prev_n=HISTORICAL_ROUNDS+n_round))

        price_adjustment = min(half_mean_spread, price_adjustment)
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        # buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # if we can buy from the local market with the ask price and sell to the south for a higher price, it's a good deal
        # we need to store for one day and convert to south the next day, so we pay one day of storage fee
        buy_orders_fair_price = observ.bidPrice - observ.transportFees - observ.exportTariff - STORAGE_FEE[ORCHIDS]

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        # sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)


        # if we can buy from the south and sell to the bid price at local market for a lower price, it's a good deal
        # no storage fee as we can trade at local market right away
        sell_orders_fair_price = observ.askPrice + observ.transportFees + observ.importTariff

        #logger.print(f'orig_fair_price: {orig_fair_price}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')


        #the below two types of trade will at most happen once, as we assume local_best_bid <= local_best_ask and observ.bidPrice <= observ.askPrice
        #thus, either local_best_ask < observ.bidPrice OR  observ.askPrice < local_best_bid
        order_for = 0

        #! cur_order_price <= ask, thus will do market order if ask < buy_orders_fair_price otherwise try limit order if cur_order_price <= buy_orders_fair_price
        market_buy_order_succeed = 0

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[ORCHIDS]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[ORCHIDS] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(ORCHIDS, ask, order_for))
                market_buy_order_succeed = 1
        
        if not market_buy_order_succeed:
            # place limit order to long at the cur_order_price at local when real_bid_price at south >= cur_order_price
            if buy_orders_fair_price >= cur_order_price and cur_position < self.position_limit[ORCHIDS]:
                num = min(2*self.position_limit[ORCHIDS], self.position_limit[ORCHIDS] - cur_position)
                orders.append(Order(ORCHIDS, math.ceil(cur_order_price), num))
                #if sell limit orders are filled
                after_limit_order_position = self.get_position(ORCHIDS, state)
                #set the conversions if limit orders are filled
                filled_quantity = after_limit_order_position - cur_position
                if filled_quantity > 0:
                    #sell those filled_quantity via conversions to south
                    order_for = filled_quantity
                    #cancel the sell limit order by sending an opposite order(limit buy this case) with the remaining quantity
                    #because south may have a different price at a new timestamp
                    #Note that after cancellation of the algorithm’s orders but before the next Tradingstate comes in, bots might also trade with each other.

                    #say num=10, cur_position=5, after_limit_order_position=7, filled_quantity=2, remaining_quantity=8
                    remaining_quantity = num - filled_quantity
                    if remaining_quantity > 0:
                        orders.append(Order(ORCHIDS, -1 * cur_order_price, -1 * remaining_quantity))

        # don't use for orchids we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        # bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        

        #cur_position = self.get_position(ORCHIDS, state)

        #! cur_order_price >= bid, thus will do market order if bid > sell_orders_fair_price otherwise try limit order if cur_order_price >= sell_orders_fair_price

        # don't use for orchids we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        # sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        market_sell_order_succeed = 0

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are longing we short to cover the long
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[ORCHIDS]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[ORCHIDS]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(ORCHIDS, bid, order_for))
                market_sell_order_succeed = 1

        cur_position = self.get_position(ORCHIDS, state)

        if not market_sell_order_succeed:
            # place limit order to short at the cur_order_price at local when real_ask_price at south <= cur_order_price
            if sell_orders_fair_price <= cur_order_price and cur_position > -self.position_limit[ORCHIDS]:
                #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
                num = max(-2*self.position_limit[ORCHIDS], -self.position_limit[ORCHIDS]-cur_position)
                orders.append(Order(ORCHIDS, math.floor(cur_order_price), num))

                #if buy limit orders are filled
                after_limit_order_position = self.get_position(ORCHIDS, state)
                #set the conversions if limit orders are filled
                filled_quantity = after_limit_order_position - cur_position
                if filled_quantity < 0:
                    order_for = filled_quantity
                    #cancel the buy limit order by sending an opposite order(limit sell this case) with the remaining quantity
                    #because south may have a different price at a new timestamp
                    #Note that after cancellation of the algorithm’s orders but before the next Tradingstate comes in, bots might also trade with each other.
                    
                    #say num=-10, cur_position=-5, after_limit_order_position=-7, filled_quantity=-2, remaining_quantity=-8
                    remaining_quantity = num - filled_quantity
                    if remaining_quantity < 0:
                        orders.append(Order(ORCHIDS, -1 * cur_order_price, -1 * remaining_quantity))

        
        
        return orders

    def gift_basket_edge_strategy(self, state : TradingState, n_round : int):
        """
        a pair trading strategy with gift basket and its components
        get an edge which is the spread between the basket and its components weighted sum

        position for basket - target position = edge/edge_max * pos_limit
        position for any component = edge/edge_max * pos_limit * weights_in_basket(4/11 for chocolate
                                                                    , 6/11 for strawberries, 1/11 for roses)

        prices_gift_basket = 4*price_chocolate+6*price_strawberries+1*price_roses

        no entry/exit but only rebalancing based on speard and its std
        when diff(target_pos, cur_pos) is small(based on lambda function), do limit trade
        when diff(target_pos, cur_pos) is large(based on lambda function), do market trade
        """
        #gb_orders, c_orders, s_orders, r_orders = [], [], [], []
        orders_map_dict = {
            GIFT_BASKET: [],
            CHOCOLATE: [],
            STRAWBERRIES: [],
            ROSES: []
        }

        target_pos_dict = {
            GIFT_BASKET: 0,
            CHOCOLATE: 0,
            STRAWBERRIES: 0,
            ROSES: 0
        }

        cur_pos_dict = {
            GIFT_BASKET: 0,
            CHOCOLATE: 0,
            STRAWBERRIES: 0,
            ROSES: 0
        }

        required_pos_dict = {
            GIFT_BASKET: 0,
            CHOCOLATE: 0,
            STRAWBERRIES: 0,
            ROSES: 0
        }

        mid_price_gift_basket = self.get_mid_price(GIFT_BASKET, state)
        mid_price_chocolate = self.get_mid_price(CHOCOLATE, state)
        mid_price_strawberries = self.get_mid_price(STRAWBERRIES, state)
        mid_price_roses = self.get_mid_price(ROSES, state)

        fair_price_basket = GIFT_BASKET_WEIGHTS[CHOCOLATE] * mid_price_chocolate + GIFT_BASKET_WEIGHTS[STRAWBERRIES] * mid_price_strawberries + GIFT_BASKET_WEIGHTS[ROSES] * mid_price_roses + self.mean_edge

        edge = mid_price_gift_basket - fair_price_basket
        #edge_demean = edge - self.mean_edge

        # target_pos_gb will be capped at self.position_limit[GIFT_BASKET]
        self.edge_max = max(abs(self.edge_max), abs(edge))

        #if we have a positive edge, it means basket is more expensive so we short it and long components
        edge_ratio = -1 * edge / self.edge_max
        target_pos_dict[GIFT_BASKET] = np.round(edge_ratio * self.position_limit[GIFT_BASKET])

        #logger.print(f'mean_edge old: {self.mean_edge}')
        #self.mean_edge = self.update_mean_value(mean_val=self.mean_edge, val=edge, prev_n=HISTORICAL_ROUNDS+n_round)

        self.mean_edge = (self.mean_edge * (HISTORICAL_ROUNDS + n_round - 1) + edge - self.mean_edge_hist) / (HISTORICAL_ROUNDS + n_round)
        #assume const std_edge
        self.std_edge = np.sqrt((self.std_edge**2 * (HISTORICAL_ROUNDS + n_round - 2) + (edge-self.mean_edge)**2) / (HISTORICAL_ROUNDS + n_round - 1))
        logger.print(f'std_edge: {self.std_edge}')

        #self.std_edge = self.update_mean_value(mean_val=self.std_edge**2, val=(edge-self.mean_edge)**2, prev_n=HISTORICAL_ROUNDS+n_round-1)


        logger.print(f'edge: {edge}')
        #logger.print(f'edge_ratio: {edge_ratio}')
        logger.print(f'mean_edge new: {self.mean_edge}')
        
        #opposite position of components vs basket
        
        target_pos_dict[CHOCOLATE] = -1 * np.round(edge_ratio * (self.price_chocolate_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[CHOCOLATE])
        
        target_pos_dict[STRAWBERRIES] = -1 * np.round(edge_ratio * (self.price_strawberries_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[STRAWBERRIES])
        
        target_pos_dict[ROSES] = -1 * np.round(edge_ratio * (self.price_roses_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[ROSES])

        #update mean prices for each product, same as using self.update_mean_value
        self.price_chocolate_mean = (self.price_chocolate_mean * (HISTORICAL_ROUNDS + n_round - 1) + mid_price_chocolate) / (HISTORICAL_ROUNDS  + n_round)
        self.price_strawberries_mean = (self.price_strawberries_mean * (HISTORICAL_ROUNDS + n_round - 1) + mid_price_strawberries) / (HISTORICAL_ROUNDS + n_round)
        self.price_roses_mean = (self.price_roses_mean * (HISTORICAL_ROUNDS + n_round - 1) + mid_price_roses) / (HISTORICAL_ROUNDS + n_round)
        self.price_gift_basket_mean = (self.price_gift_basket_mean * (HISTORICAL_ROUNDS + n_round - 1) + mid_price_gift_basket) / (HISTORICAL_ROUNDS + n_round)

        # cur_positions = [self.get_position(pos, state) for pos in PRODUCTS_GIFT]
        # target_positions = [target_pos - cur_pos for target_pos, cur_pos in zip([target_pos_gb, target_pos_chocolate, target_pos_strawberries, target_pos_roses], cur_positions)]

        cur_pos_dict[GIFT_BASKET], cur_pos_dict[CHOCOLATE], cur_pos_dict[STRAWBERRIES], cur_pos_dict[ROSES] = self.get_position(GIFT_BASKET, state), self.get_position(CHOCOLATE, state), \
                                                self.get_position(STRAWBERRIES, state), self.get_position(ROSES, state)
        
        if ((cur_pos_dict[GIFT_BASKET]==self.position_limit[GIFT_BASKET] and target_pos_dict[GIFT_BASKET]>0) \
                    or (cur_pos_dict[GIFT_BASKET]==-1*self.position_limit[GIFT_BASKET] and target_pos_dict[GIFT_BASKET]<0)):
            return [], [], [], []

        # short basket and long components, market order only for now
        if edge >= self.std_edge_coeff * self.std_edge:
            logger.print(f'target_pos_dict: {target_pos_dict}')
            logger.print(f'cur_pos_dict: {cur_pos_dict}')

            gb_order_depth: OrderDepth = state.order_depths[GIFT_BASKET]
            gb_obuy = dict(sorted(gb_order_depth.buy_orders.items()))
            gb_best_bid = list(gb_obuy.items())[0][0]

            gb_price_adjustment=self.calculate_lambda_adjusted_r3(
                                    price_distance=edge
                                    , required_pos=target_pos_dict[GIFT_BASKET]
                                    , L_p=2, k_p=0.05
                                    , L_d=1, k_d=0.05)
            gb_sell_orders_fair_price = np.round(gb_best_bid - gb_price_adjustment)
            #cap sell price at the mid price
            gb_sell_orders_fair_price = min(gb_sell_orders_fair_price, mid_price_gift_basket)
            #logger.print(f'gb_sell_orders_fair_price:{gb_sell_orders_fair_price}')
            
            #do partial fill if we will reach the position limit, only happen if cur_pos_dict[GIFT_BASKET] < 0
            # if -50+-20<-60, target_pos = -60--50 = -10
            if cur_pos_dict[GIFT_BASKET] + target_pos_dict[GIFT_BASKET] < -1 * self.position_limit[GIFT_BASKET]:
                new_target_gb = -1 * self.position_limit[GIFT_BASKET] - cur_pos_dict[GIFT_BASKET]
                #target_pos_chocolate/target_pos_gb = -1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                
                new_target_cho = new_target_gb * (-1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                
                new_target_str = new_target_gb * (-1 * (self.price_strawberries_mean/self.price_gift_basket_mean) * self.position_limit[STRAWBERRIES]/self.position_limit[GIFT_BASKET])
                new_target_r = new_target_gb * (-1 * (self.price_roses_mean/self.price_gift_basket_mean) * self.position_limit[ROSES]/self.position_limit[GIFT_BASKET])

                #if any of components has reached the limit after this trade, don't trade at this round
                if (cur_pos_dict[CHOCOLATE]+new_target_cho > self.position_limit[CHOCOLATE]) or \
                        (cur_pos_dict[STRAWBERRIES]+new_target_str > self.position_limit[STRAWBERRIES]) or \
                             (cur_pos_dict[ROSES]+new_target_r > self.position_limit[ROSES]):
                                    return [], [], [], []

                target_pos_dict[GIFT_BASKET], target_pos_dict[CHOCOLATE], target_pos_dict[STRAWBERRIES], target_pos_dict[ROSES] = new_target_gb, new_target_cho, new_target_str, new_target_r
            #orders_map_dict[GIFT_BASKET].append(Order(GIFT_BASKET, math.floor(gb_sell_orders_fair_price), int(target_pos_dict[GIFT_BASKET])))
            orders_map_dict[GIFT_BASKET].append(Order(GIFT_BASKET, gb_best_bid, int(target_pos_dict[GIFT_BASKET])))


            for product in [CHOCOLATE, STRAWBERRIES, ROSES]:
                order_depth: OrderDepth = state.order_depths[product]
                osell = dict(sorted(order_depth.sell_orders.items()))
                best_ask = list(osell.items())[0][0]
                price_adjustment=self.calculate_lambda_adjusted_r3(
                                    price_distance=edge
                                    , required_pos=target_pos_dict[product]
                                    , L_p=2, k_p=0.05
                                    , L_d=1, k_d=0.05)
                buy_orders_fair_price = np.round(best_ask + price_adjustment)
                #cap buy price at the mid price
                buy_orders_fair_price = max(buy_orders_fair_price, self.get_mid_price(product, state))
                #logger.print(f'product:{product}')
                #logger.print(f'buy_orders_fair_price:{buy_orders_fair_price}')
                
                #orders_map_dict[product].append(Order(product, math.ceil(buy_orders_fair_price), int(target_pos_dict[product])))
                orders_map_dict[product].append(Order(product, best_ask, int(target_pos_dict[product])))


            return orders_map_dict[GIFT_BASKET], orders_map_dict[CHOCOLATE], \
                                 orders_map_dict[STRAWBERRIES], orders_map_dict[ROSES]

        # long basket and short components
        elif edge <= -1 * self.std_edge_coeff * self.std_edge:
            logger.print(f'target_pos_dict: {target_pos_dict}')
            logger.print(f'cur_pos_dict: {cur_pos_dict}')

            gb_order_depth: OrderDepth = state.order_depths[GIFT_BASKET]
            gb_osell = dict(sorted(gb_order_depth.sell_orders.items()))
            gb_best_ask = list(gb_osell.items())[0][0]

            gb_price_adjustment=self.calculate_lambda_adjusted_r3(
                                    price_distance=edge
                                    , required_pos=target_pos_dict[GIFT_BASKET]
                                    , L_p=2, k_p=0.05
                                    , L_d=1, k_d=0.05)
            gb_buy_orders_fair_price = np.round(gb_best_ask + gb_price_adjustment)
            #cap buy price at the mid price
            gb_buy_orders_fair_price = max(gb_buy_orders_fair_price, mid_price_gift_basket)
            #logger.print(f'gb_buy_orders_fair_price:{gb_buy_orders_fair_price}')

            # if 50+20>60, target_pos = 60-50 = 10
            if cur_pos_dict[GIFT_BASKET] + target_pos_dict[GIFT_BASKET] > self.position_limit[GIFT_BASKET]:
                new_target_gb = self.position_limit[GIFT_BASKET] - cur_pos_dict[GIFT_BASKET]
                #target_pos_chocolate/target_pos_gb = -1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                new_target_cho = new_target_gb * (-1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                new_target_str = new_target_gb * (-1 * (self.price_strawberries_mean/self.price_gift_basket_mean) * self.position_limit[STRAWBERRIES]/self.position_limit[GIFT_BASKET])
                new_target_r = new_target_gb * (-1 * (self.price_roses_mean/self.price_gift_basket_mean) * self.position_limit[ROSES]/self.position_limit[GIFT_BASKET])
            
                #if any of components has reached the limit after this trade, don't trade at this round
                if (cur_pos_dict[CHOCOLATE]+new_target_cho < -1 * self.position_limit[CHOCOLATE]) or \
                   (cur_pos_dict[STRAWBERRIES]+new_target_str < -1 * self.position_limit[STRAWBERRIES]) or \
                   (cur_pos_dict[ROSES]+new_target_r < -1 * self.position_limit[ROSES]):
                        return [], [], [], []

                target_pos_dict[GIFT_BASKET], target_pos_dict[CHOCOLATE], target_pos_dict[STRAWBERRIES], target_pos_dict[ROSES] = new_target_gb, new_target_cho, new_target_str, new_target_r
            #orders_map_dict[GIFT_BASKET].append(Order(GIFT_BASKET, math.ceil(gb_buy_orders_fair_price), int(target_pos_dict[GIFT_BASKET])))
            orders_map_dict[GIFT_BASKET].append(Order(GIFT_BASKET, gb_best_ask, int(target_pos_dict[GIFT_BASKET])))


            for product in [CHOCOLATE, STRAWBERRIES, ROSES]:
                order_depth: OrderDepth = state.order_depths[product]
                obuy = dict(sorted(order_depth.sell_orders.items()))
                best_bid = list(obuy.items())[0][0]

                price_adjustment=self.calculate_lambda_adjusted_r3(
                                        price_distance=edge
                                        , required_pos=target_pos_dict[product]
                                        , L_p=2, k_p=0.05
                                        , L_d=1, k_d=0.05)
                sell_orders_fair_price = np.round(best_bid - price_adjustment)
                #cap sell price at the mid price
                sell_orders_fair_price = min(sell_orders_fair_price, self.get_mid_price(product, state))
                
                #orders_map_dict[product].append(Order(product, math.floor(sell_orders_fair_price), int(target_pos_dict[product])))
                orders_map_dict[product].append(Order(product, best_bid, int(target_pos_dict[product])))
                
                #logger.print(f'product:{product}')
                #logger.print(f'sell_orders_fair_price:{sell_orders_fair_price}')


            return orders_map_dict[GIFT_BASKET], orders_map_dict[CHOCOLATE], \
                                orders_map_dict[STRAWBERRIES], orders_map_dict[ROSES]


        #if edge minus mean is within -+1 * self.std_edge_coeff * self.std_edge, don't trade
        else:
            return [], [], [], []

    def coco_iv_strategy(self, state : TradingState, n_round : int):

        coco_orders_map_dict = {
            COCONUT_COUPON: [],
            COCONUT: []
        }
        
        iv_cur, iv_percentile = None, None
        mid_price_coconut = self.get_mid_price(COCONUT, state)
        mid_price_coupon = self.get_mid_price(COCONUT_COUPON, state)

        iv_cur = fit_vol(0.1, mid_price_coconut, mid_price_coupon, 1)
        
        #iv_cur = fit_implied_volatility(S=mid_price_coconut, K=10000, T=1, r=0, market_price=mid_price_coupon)
        #iv_percentile = self.percentile_within_minmax(iv_cur, min_val=self.iv_min, max_val=self.iv_max)

        self.update_ema_vol(vol_type='IV', cur_vol=iv_cur)

        cur_coconut_ret = np.log(mid_price_coconut/self.last_coconut_mid_price)
        self.last_coconut_mid_price = mid_price_coconut
        #same ema param to update historical return and historical vol
        self.his_ret_ema_period1 = self.hv_ema_period1_param * cur_coconut_ret + (1-self.hv_ema_period1_param) * self.his_ret_ema_period1 \
                                                                                                # * self.annual_trading_days * 100
        
        self.hv_ema_period1 = np.sqrt((self.hv_ema_period1_param * (cur_coconut_ret-self.his_ret_ema_period1)**2 +  \
                                            (1-self.hv_ema_period1_param) * self.hv_ema_period1**2)) * np.sqrt(self.annual_trading_days) * 100

        self.his_ret_ema_period2 = self.hv_ema_period2_param * cur_coconut_ret + (1-self.hv_ema_period2_param) * self.his_ret_ema_period2 \
                                                                                               # * self.annual_trading_days * 100
        
        self.hv_ema_period2 = np.sqrt((self.hv_ema_period2_param * (cur_coconut_ret-self.his_ret_ema_period2)**2 +  \
                                            (1-self.hv_ema_period2_param) * self.hv_ema_period2**2)) * np.sqrt(self.annual_trading_days) * 100

        self.his_ret_ema_period3 = self.hv_ema_period3_param * cur_coconut_ret + (1-self.hv_ema_period3_param) * self.his_ret_ema_period3 \
                                                                                                # * self.annual_trading_days * 100
        
        self.hv_ema_period3 = np.sqrt((self.hv_ema_period3_param * (cur_coconut_ret-self.his_ret_ema_period3)**2 +  \
                                            (1-self.hv_ema_period3_param) * self.hv_ema_period3**2)) * np.sqrt(self.annual_trading_days) * 100

        #use hv as a leading indicator of iv; here the signal is if hv has a momentum of increasing
        hv_cross_signal = (self.hv_ema_period1 > self.hv_ema_period2) & (self.hv_ema_period2 > self.hv_ema_period3)

        # fair_vol = self.hv_period1_weight * self.hv_ema_period1 + self.hv_period2_weight * self.hv_ema_period2 + \
        #                     self.hv_period3_weight * self.hv_ema_period3 + self.iv_ema_weight * self.iv_ema

        fair_vol = self.iv_initial-.001
        
        if iv_cur < fair_vol:
            fair_vol_signal = True
        if iv_cur > fair_vol:
            fair_vol_signal = False

        self.percentile_two_queues(new_val=iv_cur)
        iv_quantile_40 = self.iv_10_40_quantiles[-1]
        iv_quantile_60 = self.iv_60_90_quantiles[0]

        if iv_cur < iv_quantile_40:
            iv_quantile_signal = True
        if iv_cur > iv_quantile_60:
            iv_quantile_signal = False

        delta = fun_BS_quick(mid_price_coconut, 10000, iv_cur, T = 1, r = 0, q = 0, ReturnDelta = True)

        logger.print(f'iv_cur: {iv_cur}')
        logger.print(f'self.iv_ema: {self.iv_ema}')
        logger.print(f'fair_vol: {fair_vol}')
        logger.print(f'self.his_ret_ema_period1: {self.his_ret_ema_period1}')
        logger.print(f'self.hv_ema_period1: {self.hv_ema_period1}')

        logger.print(f'self.his_ret_ema_period2: {self.his_ret_ema_period2}')
        logger.print(f'self.hv_ema_period2: {self.hv_ema_period2}')

        logger.print(f'self.his_ret_ema_period3: {self.his_ret_ema_period3}')
        logger.print(f'self.hv_ema_period3: {self.hv_ema_period3}')

        logger.print(f'iv_quantile_40: {iv_quantile_40}')
        logger.print(f'iv_quantile_60: {iv_quantile_60}')

        logger.print(f'delta: {delta}')


        cur_pos_coupon = self.get_position(COCONUT_COUPON, state)
        cur_pos_coconut = self.get_position(COCONUT, state)

        #long option, short equity
        if fair_vol_signal:
            if cur_pos_coupon <= 0:
                logger.print('a long option signal was found')
                # 550 - -550 = 1100
                required_pos_coupon = self.coupon_default_positon - cur_pos_coupon
                required_pos_coconut = -1 * required_pos_coupon * delta

                coupon_order_depth: OrderDepth = state.order_depths[COCONUT_COUPON]
                coupon_osell = dict(sorted(coupon_order_depth.sell_orders.items()))
                coupon_best_ask = list(coupon_osell.items())[0][0]
                
                coco_orders_map_dict[COCONUT_COUPON].append(Order(COCONUT_COUPON, coupon_best_ask, int(required_pos_coupon)))

                coconut_order_depth: OrderDepth = state.order_depths[COCONUT]
                coconut_obuy = dict(sorted(coconut_order_depth.buy_orders.items()))
                coconut_best_bid = list(coconut_obuy.items())[0][0]
                
                coco_orders_map_dict[COCONUT].append(Order(COCONUT, coconut_best_bid, int(required_pos_coconut)))

                return coco_orders_map_dict[COCONUT], coco_orders_map_dict[COCONUT_COUPON]
            else:
                return [], []

        #short option, long equity
        else:
            if cur_pos_coupon >= 0:
                logger.print('a short option signal was found')
                # -550 - 550 = -1100
                required_pos_coupon =  -1 * self.coupon_default_positon - cur_pos_coupon
                required_pos_coconut = -1 * required_pos_coupon * delta

                coupon_order_depth: OrderDepth = state.order_depths[COCONUT_COUPON]
                coupon_obuy = dict(sorted(coupon_order_depth.buy_orders.items()))
                coupon_best_bid = list(coupon_obuy.items())[0][0]
                
                coco_orders_map_dict[COCONUT_COUPON].append(Order(COCONUT_COUPON, coupon_best_bid, int(required_pos_coupon)))

                coconut_order_depth: OrderDepth = state.order_depths[COCONUT]
                coconut_osell = dict(sorted(coconut_order_depth.sell_orders.items()))
                coconut_best_ask = list(coconut_osell.items())[0][0]
                
                coco_orders_map_dict[COCONUT].append(Order(COCONUT, coconut_best_ask, int(required_pos_coconut)))

                return coco_orders_map_dict[COCONUT], coco_orders_map_dict[COCONUT_COUPON]
            else:
                return [], []

        
        # #buy option signal if all 3 signals are True
        # if hv_cross_signal & fair_vol_signal & iv_quantile_signal:
        #     logger.print('a long option signal was found')
        #     pos_coupon = self.coupon_default_positon
        #     pos_coconut = -1 * self.coupon_default_positon * delta

        #     coupon_order_depth: OrderDepth = state.order_depths[COCONUT_COUPON]
        #     coupon_osell = dict(sorted(coupon_order_depth.sell_orders.items()))
        #     coupon_best_ask = list(coupon_osell.items())[0][0]
            
        #     coco_orders_map_dict[COCONUT_COUPON].append(Order(COCONUT_COUPON, coupon_best_ask, int(pos_coupon)))

        #     coconut_order_depth: OrderDepth = state.order_depths[COCONUT]
        #     coconut_obuy = dict(sorted(coconut_order_depth.buy_orders.items()))
        #     coconut_best_bid = list(coconut_obuy.items())[0][0]
            
        #     coco_orders_map_dict[COCONUT].append(Order(COCONUT, coconut_best_bid, int(pos_coconut)))

        #     return coco_orders_map_dict[COCONUT], coco_orders_map_dict[COCONUT_COUPON]




        # #sell option signal if all 3 signals are False
        # elif not (hv_cross_signal | fair_vol_signal | iv_quantile_signal):
        #     logger.print('a short option signal was found')
        #     pos_coupon = -1 * self.coupon_default_positon
        #     pos_coconut = -1 * self.coupon_default_positon * delta

        #     coupon_order_depth: OrderDepth = state.order_depths[COCONUT_COUPON]
        #     coupon_obuy = dict(sorted(coupon_order_depth.buy_orders.items()))
        #     coupon_best_bid = list(coupon_obuy.items())[0][0]
            
        #     coco_orders_map_dict[COCONUT_COUPON].append(Order(COCONUT_COUPON, coupon_best_bid, int(pos_coupon)))

        #     coconut_order_depth: OrderDepth = state.order_depths[COCONUT]
        #     coconut_osell = dict(sorted(coconut_order_depth.sell_orders.items()))
        #     coconut_best_ask = list(coconut_osell.items())[0][0]
            
        #     coco_orders_map_dict[COCONUT].append(Order(COCONUT, coconut_best_ask, int(pos_coconut)))

        #     return coco_orders_map_dict[COCONUT], coco_orders_map_dict[COCONUT_COUPON]

        # else:
        #     return [], []
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.round += 1
        # Initialize the method output dict as an empty dict
        result = {}
        self.update_ema_prices(product=AMETHYSTS, state=state)
        self.update_ema_prices(product=STARFRUIT, state=state)


        self.update_ema_prices(product=CHOCOLATE, state=state)
        self.update_ema_prices(product=STRAWBERRIES, state=state)
        self.update_ema_prices(product=ROSES, state=state)
        self.update_ema_prices(product=GIFT_BASKET, state=state)

        # may not need to do try catch exceptions, exceptions will shown in the log 
        result[AMETHYSTS] = self.amethysts_strategy(state)

        result[STARFRUIT] = self.starfruit_strategy(state, n_round=self.round)

        gb_orders, c_orders, s_orders, r_orders = self.gift_basket_edge_strategy(state, n_round=self.round)

        result[GIFT_BASKET] = gb_orders
        result[CHOCOLATE] = c_orders
        result[STRAWBERRIES] = s_orders
        result[ROSES] = r_orders

        coconut_orders, coupon_orders = self.coco_iv_strategy(state, n_round=self.round)
        
        result[COCONUT] = coconut_orders
        result[COCONUT_COUPON] = coupon_orders
        
        



        result[ORCHIDS] = self.orchids_strategy(state=state, n_round=self.round
                                                                    , observations=state.observations)
        
        # cancel out the current position of ORCHIDS based on conversions
        cur_position = self.get_position(ORCHIDS, state)
        conversions = -1 * cur_position
        # state.position.update({ORCHIDS: cur_position+conversions})

        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "TEST" 


        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
