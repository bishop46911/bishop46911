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

PRODUCTS = [
    AMETHYSTS,
    STARFRUIT,
]

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
}

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
        }

        # Values to compute pnl
        #self.cash = 0

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

        # to be adjusted, typically it equals 2/(N+1) where N is the period of EMA
        self.ema_param = 0.1

        # optional for z-score based srtategy -
        self.zscore_scale = 19.89

            # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0) 
    
    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of dollar value currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)
    
    def update_pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price
        
        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value
        
        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()
    
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

    def update_mean_value(self, val, prev_n):
        """
        keep a rolling mean_val for each incoming val
        """
        if prev_n<0:
            return None
        mean_val = None
        if mean_val is None:
            mean_val = val
        mean_val = (mean_val * prev_n + val) / (prev_n + 1)

        return np.round(mean_val)

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

    # Algorithm logic
    def amethysts_strategy(self, state : TradingState):
        '''
        Returns a list of orders with trades of amethysts.
        '''
        orders: list[Order] = []
        cur_position = self.get_position(AMETHYSTS, state)

        order_depth: OrderDepth = state.order_depths[AMETHYSTS]

        osell = dict(sorted(order_depth.sell_orders.items()))
        obuy = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        # TODO optional - fading of fair_price based on cur_position, self.position_limit[AMETHYSTS]
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
        logger.print(f'best_bid_vol - best_ask_vol: {best_bid_vol - best_ask_vol}')

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
                                                          , volume_change=diff_best_bid_ask)

        #we add price adjustment at most 3, otherwise the loss would explode
        #3 is mid point of the avg best_bid, best_ask which is -6 based on data exploration
        #half_mean_spread = 3

        #have a dynamic half_mean_spread based on real data
        best_bid_ask_spread = abs(best_ask - best_bid)
        half_mean_spread = np.round(0.5 * self.update_mean_value(val=best_bid_ask_spread, prev_n=n_round))

        price_adjustment = min(half_mean_spread, price_adjustment)
        # For buy orders, increase fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        buy_orders_fair_price = np.round(orig_fair_price + price_adjustment)

        # For sell orders, decrease fair price more for larger positions, longer time, bigger price difference so that the orders will be filled more likely
        sell_orders_fair_price = np.round(orig_fair_price - price_adjustment)

        logger.print(f'orig_fair_price: {orig_fair_price}')
        logger.print(f'buy_orders_fair_price: {buy_orders_fair_price}')
        logger.print(f'sell_orders_fair_price: {sell_orders_fair_price}')

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

        # AMETHYSTS STRATEGY
        # may not need to do try catch exceptions, exceptions will shown in the log 
        result[AMETHYSTS] = self.amethysts_strategy(state)

        # STARFRUIT STRATEGY
        result[STARFRUIT] = self.starfruit_strategy(state, n_round=self.round)

        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "TEST" 

        #later will figure out how to use conversions 
        conversions = 0

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData