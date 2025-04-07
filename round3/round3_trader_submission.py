from typing import Dict, List, Any
import math
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
    GIFT_BASKET
]

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
    #ORCHIDS seems to be very volatile
    ORCHIDS: 1_100
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
            GIFT_BASKET: 60
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
            self.ema_prices[product] = None

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

        self.mean_edge = 380
        self.std_edge = 76.42
        #min threshold to trigger trades
        self.std_edge_coeff = 0.05

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
    

    def update_ema_prices(self, product, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        mid_price = self.get_mid_price(product, state)
        # do ema update if mid_price is not none
        if mid_price is not None:
            # ema update - base case
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            # ema update - recursive case
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

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

    def chocolate_strategy(self, state : TradingState, n_round : int):
        '''
        Returns a list of orders with trades of chocolate.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(CHOCOLATE, state)

        order_depth: OrderDepth = state.order_depths[CHOCOLATE]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[CHOCOLATE]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        

        max_position_limit = self.position_limit[CHOCOLATE]

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
        half_mean_spread = np.round(0.5 * self.update_mean_value(val=best_bid_ask_spread, prev_n=HISTORICAL_ROUNDS+n_round))

        
        # TODO - tune hyperparams of lambda func
        # price_adjustment = 0
        price_adjustment = min(half_mean_spread, price_adjustment)
        
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        logger.print(f'gift_basket:')

        logger.print(f'orig_fair_price: {orig_fair_price}')
        logger.print(f'price_adjustment: {price_adjustment}')

        logger.print(f'fair_price_distance: {fair_price_distance}')
        logger.print(f'cur_position: {cur_position}')
        logger.print(f'diff_best_bid_ask: {diff_best_bid_ask}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[CHOCOLATE]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[CHOCOLATE] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(CHOCOLATE, ask, order_for))
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        
        if cur_position < self.position_limit[CHOCOLATE]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[CHOCOLATE], self.position_limit[CHOCOLATE] - cur_position)
            orders.append(Order(CHOCOLATE, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(CHOCOLATE, state)

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[CHOCOLATE]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[CHOCOLATE]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(CHOCOLATE, bid, order_for))


        if cur_position > -self.position_limit[CHOCOLATE]:
            num = max(-2*self.position_limit[CHOCOLATE], -self.position_limit[CHOCOLATE]-cur_position)
            orders.append(Order(CHOCOLATE, sell_pr, num))
            cur_position += num

        return orders
    

    def strawberries_strategy(self, state : TradingState, n_round : int):
        '''
        Returns a list of orders with trades of strawberries.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(STRAWBERRIES, state)

        order_depth: OrderDepth = state.order_depths[STRAWBERRIES]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[STRAWBERRIES]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        

        max_position_limit = self.position_limit[STRAWBERRIES]

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
        half_mean_spread = np.round(0.5 * self.update_mean_value(val=best_bid_ask_spread, prev_n=HISTORICAL_ROUNDS+n_round))

        
        # TODO - tune hyperparams of lambda func
        # price_adjustment = 0
        price_adjustment = min(half_mean_spread, price_adjustment)
        
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        logger.print(f'strawberries:')

        logger.print(f'orig_fair_price: {orig_fair_price}')
        logger.print(f'price_adjustment: {price_adjustment}')

        logger.print(f'fair_price_distance: {fair_price_distance}')
        logger.print(f'cur_position: {cur_position}')
        logger.print(f'diff_best_bid_ask: {diff_best_bid_ask}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[STRAWBERRIES]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[STRAWBERRIES] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(STRAWBERRIES, ask, order_for))
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        
        if cur_position < self.position_limit[STRAWBERRIES]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[STRAWBERRIES], self.position_limit[STRAWBERRIES] - cur_position)
            orders.append(Order(STRAWBERRIES, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(STRAWBERRIES, state)

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[STRAWBERRIES]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[STRAWBERRIES]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(STRAWBERRIES, bid, order_for))


        if cur_position > -self.position_limit[STRAWBERRIES]:
            num = max(-2*self.position_limit[STRAWBERRIES], -self.position_limit[STRAWBERRIES]-cur_position)
            orders.append(Order(STRAWBERRIES, sell_pr, num))
            cur_position += num

        return orders
    
    def roses_strategy(self, state : TradingState, n_round : int):
        '''
        Returns a list of orders with trades of roses.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(ROSES, state)

        order_depth: OrderDepth = state.order_depths[ROSES]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[ROSES]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        

        max_position_limit = self.position_limit[ROSES]

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
        half_mean_spread = np.round(0.5 * self.update_mean_value(val=best_bid_ask_spread, prev_n=HISTORICAL_ROUNDS+n_round))

        
        # TODO - tune hyperparams of lambda func
        # price_adjustment = 0
        price_adjustment = min(half_mean_spread, price_adjustment)
        
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        logger.print(f'roses:')

        logger.print(f'orig_fair_price: {orig_fair_price}')
        logger.print(f'price_adjustment: {price_adjustment}')

        logger.print(f'fair_price_distance: {fair_price_distance}')
        logger.print(f'cur_position: {cur_position}')
        logger.print(f'diff_best_bid_ask: {diff_best_bid_ask}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[ROSES]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[ROSES] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(ROSES, ask, order_for))
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        
        if cur_position < self.position_limit[ROSES]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[ROSES], self.position_limit[ROSES] - cur_position)
            orders.append(Order(ROSES, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(ROSES, state)

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[ROSES]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[ROSES]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(ROSES, bid, order_for))


        if cur_position > -self.position_limit[ROSES]:
            num = max(-2*self.position_limit[ROSES], -self.position_limit[ROSES]-cur_position)
            orders.append(Order(ROSES, sell_pr, num))
            cur_position += num

        return orders
    
    # def gift_basket_strategy(self, state : TradingState, n_round : int):
        '''
        Returns a list of orders with trades of gift_basket.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(GIFT_BASKET, state)

        order_depth: OrderDepth = state.order_depths[GIFT_BASKET]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_bid, best_ask = list(obuy.items())[0][0], list(osell.items())[0][0]
        best_bid_vol, best_ask_vol = abs(list(obuy.items())[0][1]), abs(list(osell.items())[0][1])
        diff_best_bid_ask = abs(best_bid_vol - best_ask_vol)
        #logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

        #orig_fair_price could be ema or vwap of (all the top 3 ask or bid orders)
        #orig_fair_price = self.ema_prices[GIFT_BASKET]

        orig_fair_price = self.compute_vwamp_bidask(bid_orders_dict=obuy, ask_orders_dict=osell)
        #cur_order_price - current mid_price
        cur_order_price = np.round((best_bid+best_ask)/2)

        fair_price_distance = abs(orig_fair_price - cur_order_price)
        

        max_position_limit = self.position_limit[GIFT_BASKET]

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
        half_mean_spread = np.round(0.5 * self.update_mean_value(val=best_bid_ask_spread, prev_n=n_round))

        
        # TODO - tune hyperparams of lambda func
        # price_adjustment = 0
        price_adjustment = min(half_mean_spread, price_adjustment)
        
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        logger.print(f'gift_basket:')

        logger.print(f'orig_fair_price: {orig_fair_price}')
        logger.print(f'price_adjustment: {price_adjustment}')

        logger.print(f'fair_price_distance: {fair_price_distance}')
        logger.print(f'cur_position: {cur_position}')
        logger.print(f'diff_best_bid_ask: {diff_best_bid_ask}')
        #logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        #logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

        #for sell order, their vol are in negative
        #iterate curent ask orders and send our bid order for trade signals
        for ask, vol in osell.items():
            # if best ask is less than fair_price we long
            # or if it is <= fair_price and we are shorting we long to cover the short
            if ((ask < buy_orders_fair_price) or ((cur_position<0) and (ask == buy_orders_fair_price))) \
                                                                            and cur_position < self.position_limit[GIFT_BASKET]:
                #max volume we can buy
                order_for = min(-vol, self.position_limit[GIFT_BASKET] - cur_position)
                assert(order_for >= 0)
                cur_position += order_for
                # buy the max limit amount with the appropriate ask
                orders.append(Order(GIFT_BASKET, ask, order_for))
        

        # we offer the bid to be +1 on the current best bid, but it should be at most the fair price minus one(to make the trade profitable)
        bid_pr = math.ceil(min(best_bid + 1, buy_orders_fair_price - 1))

        # we offer the ask to be -1 on the current best ask, but it should be at least the fair price plus one(to make the trade profitable)
        sell_pr = math.floor(max(best_ask - 1, sell_orders_fair_price + 1))

        
        if cur_position < self.position_limit[GIFT_BASKET]:
            #if we short the max amount, we can at most cover them all then buy max amount(essentially buy the double of limit)
            num = min(2*self.position_limit[GIFT_BASKET], self.position_limit[GIFT_BASKET] - cur_position)
            orders.append(Order(GIFT_BASKET, bid_pr, num))
            cur_position += num

        cur_position = self.get_position(GIFT_BASKET, state)

        for bid, vol in obuy.items():
            # if best bid is greater than fair_price we long
            # or if it is >= fair_price and we are shorting we long to cover the short
            if ((bid > sell_orders_fair_price) or ((cur_position>0) and (bid == sell_orders_fair_price))) \
                                                                                    and cur_position > -self.position_limit[GIFT_BASKET]:
                # order_for is a negative number denoting how much we can sell
                order_for = max(-vol, -self.position_limit[GIFT_BASKET]-cur_position)
                assert(order_for <= 0)
                cur_position += order_for
                # sell the max limit amount with the appropriate bid
                orders.append(Order(GIFT_BASKET, bid, order_for))


        if cur_position > -self.position_limit[GIFT_BASKET]:
            num = max(-2*self.position_limit[GIFT_BASKET], -self.position_limit[GIFT_BASKET]-cur_position)
            orders.append(Order(GIFT_BASKET, sell_pr, num))
            cur_position += num

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

        #self.mean_edge = (self.mean_edge * (HISTORICAL_ROUNDS + n_round - 1) + edge) / (HISTORICAL_ROUNDS + n_round)
        #assume const std_edge
        self.std_edge = np.sqrt((self.std_edge**2 * (HISTORICAL_ROUNDS + n_round - 2) + (edge-self.mean_edge)**2) / (HISTORICAL_ROUNDS + n_round - 1))
        logger.print(f'std_edge: {self.std_edge}')

        #self.std_edge = self.update_mean_value(mean_val=self.std_edge**2, val=(edge-self.mean_edge)**2, prev_n=HISTORICAL_ROUNDS+n_round-1)


        logger.print(f'edge: {edge}')
        #logger.print(f'edge_ratio: {edge_ratio}')
        #logger.print(f'mean_edge new: {self.mean_edge}')
        
        #opposite position of components vs basket
        
        target_pos_dict[CHOCOLATE] = -1 * np.round(edge_ratio * (self.price_chocolate_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[CHOCOLATE])
        
        target_pos_dict[STRAWBERRIES] = -1 * np.round(edge_ratio * (self.price_strawberries_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[STRAWBERRIES])
        
        target_pos_dict[ROSES] = -1 * np.round(edge_ratio * (self.price_roses_mean/self.price_gift_basket_mean) * \
                                        self.position_limit[ROSES])

        #update mean prices for each product, same as using self.update_mean_value
        # self.price_chocolate_mean = (self.price_chocolate_mean * HISTORICAL_ROUNDS + mid_price_chocolate) / (HISTORICAL_ROUNDS + 1)
        # self.price_strawberries_mean = (self.price_strawberries_mean * HISTORICAL_ROUNDS + mid_price_strawberries) / (HISTORICAL_ROUNDS + 1)
        # self.price_roses_mean = (self.price_roses_mean * HISTORICAL_ROUNDS + mid_price_roses) / (HISTORICAL_ROUNDS + 1)
        # self.price_gift_basket_mean = (self.price_gift_basket_mean * HISTORICAL_ROUNDS + mid_price_gift_basket) / (HISTORICAL_ROUNDS + 1)

        self.price_chocolate_mean = (self.price_chocolate_mean * (HISTORICAL_ROUNDS + n_round) + mid_price_chocolate) / (HISTORICAL_ROUNDS  + n_round + 1)
        self.price_strawberries_mean = (self.price_strawberries_mean * (HISTORICAL_ROUNDS + n_round) + mid_price_strawberries) / (HISTORICAL_ROUNDS + n_round + 1)
        self.price_roses_mean = (self.price_roses_mean * (HISTORICAL_ROUNDS + n_round) + mid_price_roses) / (HISTORICAL_ROUNDS + n_round + 1)
        self.price_gift_basket_mean = (self.price_gift_basket_mean * (HISTORICAL_ROUNDS + n_round) + mid_price_gift_basket) / (HISTORICAL_ROUNDS + n_round + 1)

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
                
                new_target_cho = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                
                new_target_str = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_strawberries_mean/self.price_gift_basket_mean) * self.position_limit[STRAWBERRIES]/self.position_limit[GIFT_BASKET])
                new_target_r = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_roses_mean/self.price_gift_basket_mean) * self.position_limit[ROSES]/self.position_limit[GIFT_BASKET])

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
                new_target_cho = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_chocolate_mean/self.price_gift_basket_mean) * self.position_limit[CHOCOLATE]/self.position_limit[GIFT_BASKET])
                new_target_str = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_strawberries_mean/self.price_gift_basket_mean) * self.position_limit[STRAWBERRIES]/self.position_limit[GIFT_BASKET])
                new_target_r = target_pos_dict[GIFT_BASKET] * (-1 * (self.price_roses_mean/self.price_gift_basket_mean) * self.position_limit[ROSES]/self.position_limit[GIFT_BASKET])
            
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

        self.execute_gift_basket()



        #return gb_orders, c_orders, s_orders, r_orders 

    def execute_gift_basket(self, state: TradingState):


        return None
    
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

        # result[CHOCOLATE] = self.chocolate_strategy(state, n_round=self.round)

        # result[STRAWBERRIES] = self.strawberries_strategy(state, n_round=self.round)

        # result[ROSES] = self.roses_strategy(state, n_round=self.round)

        # result[GIFT_BASKET] = self.gift_basket_strategy(state, n_round=self.round)

        gb_orders, c_orders, s_orders, r_orders = self.gift_basket_edge_strategy(state, n_round=self.round)

        result[GIFT_BASKET] = gb_orders
        result[CHOCOLATE] = c_orders
        result[STRAWBERRIES] = s_orders
        result[ROSES] = r_orders



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
    
