import requests
import json
#_webhook_url = "https://662d-44-197-41-219.ngrok.io"
_webhook_url = "http://127.0.0.1:6000"
_payload_url = f"{_webhook_url}/tradesignal"

# # Trade signal
# msg = {"AlertType":"TRADE",
#       "TradeType":"long", # short long
#       "Market":"MESZ2",
#       "OrderType": "stop", #limit market stop
#       "StopPrice": 3900,
#       "Side":"sell",
#       "Size": 5
#       }

#OSO signal long
msg = {"AlertType":"TRADE",
      "TradeType":"long", # short long
      "Market":"ESH3",#"MESZ2",
      "OrderType": "OSO",
      "Side":"buy",
      "Size": 1,
      "LimitPrice": 3857,
      "TPLimitPrice": 3900,
      "SLStopPrice": 3800
      }
#cancel all signal
msg2 = {
    "AlertType":"TRADE",
      "TradeType":"cancel_all", 
      "Market":"ESH3"
}

# #OSO signal short
# msg = {"AlertType":"TRADE",
#       "TradeType":"short", # short long
#       "Market":"ESZ2",#"MESZ2",
#       "OrderType": "OSO",
#       "Side":"sell",
#       "Size": 1,
#       "LimitPrice": 4025,
#       "TPLimitPrice": 4021,
#       "SLStopPrice": 4029
#       }

resp = requests.post(_payload_url, data=json.dumps(
            msg2, sort_keys=True, default=str), headers={'Content-Type': 'application/json'}, timeout=10.0)
print(resp)