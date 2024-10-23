#### Tradovate-Tradingview V8 ####

1. OSO Order signal:
    
    {
        "AlertType":"TRADE",
        "TradeType":"long", 
        "Market":"ESZ2",
        "OrderType": "OSO",
        "Side":"buy",
        "Size": 1,
        "LimitPrice": 4025,
        "TPLimitPrice": 4030,
        "SLStopPrice": 4020
    }

    (i)     AlertType should be equal to TRADE to begin processing of signal
    (ii)    TradeType can be long or short
    (iii)   OrderType should be equal to OSO to place an OSO order
    (iv)    When OrderType is OSO, Only following singals would be valid signals:
            (As there won't be any need of separate exit signal)
                A. TradeType is long and Side is buy
                B. TradeType is short and Side is sell
    (v)     LimitPrice sets limit price of entry order
    (vi)    SLStopPrice sets stop price of Stop-Loss exit stop order (bracket1)
    (vii)   TPLimitPrice sets limit price of Take-Profit exit limit order (bracket2)

2. Stop Order signal:
    
    { 
        "AlertType":"TRADE",
        "TradeType":"long",
        "Market":"MESZ2",
        "OrderType": "stop",
        "StopPrice": 3900,
        "Side":"buy",
        "Size": 5
    }

    (i)     AlertType should be equal to TRADE to begin processing of signal
    (ii)    TradeType can be long or short
    (iii)   OrderType should be equal to stop for stop orders
    (iv)    When OrderType is stop, StopPrice parameter should contain stop price for the order
    (v)     Following case are valid treadted as valid signals:
                A. TradeType is long and Side is buy # Long Entry
                B. TradeType is long and Side is sell # Long Exit
                C. TradeType is short and Side is sell # Short Entry
                D. TradeType is short and Side is buy # Long Exit

3. Limit Order signal:
    
    { 
        "AlertType":"TRADE",
        "TradeType":"long",
        "Market":"MESZ2",
        "OrderType": "limit",
        "LimitPrice": 3900,
        "Side":"buy",
        "Size": 5
    }

    (i)     AlertType should be equal to TRADE to begin processing of signal
    (ii)    TradeType can be long or short
    (iii)   OrderType should be equal to limit for limit orders
    (iv)    When OrderType is limit, LimitPrice parameter should contain limit price for the order
    (v)     Following cases are treated as valid signals:
                A. TradeType is long and Side is buy # Long Entry
                B. TradeType is long and Side is sell # Long Exit
                C. TradeType is short and Side is sell # Short Entry
                D. TradeType is short and Side is buy # Long Exit

4. Market Order signal:
    
    { 
        "AlertType":"TRADE",
        "TradeType":"long",
        "Market":"MESZ2",
        "OrderType": "market",
        "Side":"buy",
        "Size": 5
    }

    (i)     AlertType should be equal to TRADE to begin processing of signal
    (ii)    TradeType can be long or short
    (iii)   OrderType should be equal to market for market orders
    (iv)     Following cases are treated as valid signals:
                A. TradeType is long and Side is buy # Long Entry
                B. TradeType is long and Side is sell # Long Exit
                C. TradeType is short and Side is sell # Short Entry
                D. TradeType is short and Side is buy # Long Exit

5. cancel all signal:
    
    { 
       "AlertType":"TRADE",
      "TradeType":"cancel_all", 
      "Market":"ESH3"
    }

    (i)     AlertType should be equal to TRADE to begin processing of signal
    (ii)    TradeType can be cancel_all
    (iii)   This will cancel all non-filled orders for the provided market
    (iv)    To test this in demo account, user can use post_app.py script. While main.py is running,
            run post_app.py to place OSO type order. Then run post_app.py to send cancel_all signal






