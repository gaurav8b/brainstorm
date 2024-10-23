from requests import Request, Session, Response
from typing import Optional, Dict, Any, List
from datetime import datetime,timezone,timedelta
import pandas as pd
import numpy as np
import os
import uuid

from dotenv import load_dotenv
from os.path import join, dirname

dotenv_path = join(dirname(__file__), 'config.env') #py
#dotenv_path = join(dirname("__file__"), 'config.env') #Notebook
load_dotenv(dotenv_path)
tz_utc = timezone.utc

class TVClient:
    
    def __init__(self, accName, logger = None) -> None:
        self._logger =  logger 
        self._session = Session()
        self._env = os.environ.get('TV_API_ENV')
        self._api_secret = os.environ.get('TV_API_SECRET')
        self._api_acc_name = accName
        self._api_name = os.environ.get('name')
        self._api_pass = os.environ.get('password')
        self._api_appId = os.environ.get('appId')
        self._api_appver = os.environ.get('appVersion')
        self._api_cid = os.environ.get('cid')
        
        self._base_url = "https://demo.tradovateapi.com/v1"
        if self._env == 'LIVE':
            self._base_url = "https://live.tradovateapi.com/v1"
        self._base_url_md = "https://md.tradovateapi.com"
        self._accessToken = ''
        self.init_access_token()
        self.init_account_id()

    def print_and_log(self,msg):
        log_msg = f'{msg}'
        print(log_msg)
        if self._logger != None:
            self._logger.info(log_msg)
    
    def init_access_token(self):
        url = '/auth/accessTokenRequest'
        params = {  "name":       self._api_name,        #"Your credentials here",
                    "password":   self._api_pass,        #"Your credentials here",
                    "appId":      self._api_appId,       #"Sample App",
                    "appVersion": self._api_appver,      #"1.0",
                    "cid":        self._api_cid,         #0,
                    "sec":        self._api_secret       #"Your API secret here"
             }
        try:
            response =  self._post(url,params)
            self._accessToken = response['accessToken']
            self._mdAccessToken = response['mdAccessToken']
            self._expirationTime = response['expirationTime']
            self._userStatus = response['userStatus']
            self._userId = response['userId']
            self._name = response['name']
            self._hasLive = response['hasLive']
            self._outdatedTaC = response['outdatedTaC']
            self._hasFunded = response['hasFunded']
            self._hasMarketData = response['hasMarketData']
            self._outdatedLiquidationPolicy = response['outdatedLiquidationPolicy']
        except Exception as e:
            self.print_and_log(response)
            self.print_and_log(f'Error: {e}')
            
    def init_account_id(self):
        path ='/account/list'
        res = self._get(path)
        df = pd.DataFrame(res)
        acc_id = df[df['name']==self._api_acc_name]['id']
        acc_id = int(acc_id.iloc[0])
        self.print_and_log(f'Account=> Name: {self._api_acc_name} Id: {acc_id}')
        self._accountId = acc_id
        
    def check_access_token_expiry(self):
        ext = self._expirationTime
        dt_ex = pd.to_datetime(ext)
        dt_now = datetime.now(tz_utc)
        if dt_now >= dt_ex or  (dt_ex-dt_now).seconds<10:
            self.print_and_log('Access token expired, renewing ..')
            self.init_access_token()
            
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, json=params)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('DELETE', path, json=params)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        request = Request(method, self._base_url + path, **kwargs)
        self._sign_request(request,path)
        response = self._session.send(request.prepare())
        return self._process_response(response)

    def _sign_request(self, request: Request,path) -> None:
        request.headers['Content-Type'] = 'application/json'
        if len(self._accessToken)>0 or path.count('accessTokenRequest')==0:
            request.headers['Authorization'] = f'Bearer {self._accessToken}'

    def _process_response(self, response: Response) -> Any:
        #self.print_and_log(f'Response : {response.text}')
        if response.status_code == 200:
            response = response.json()
        else:
            self.print_and_log(f'Error: {response.text}')
        return response
    
    def get_contract_info_by_id(self,contractId):    
        path ='/contract/item'
        params = {'id':int(contractId)}
        res = self._get(path,params)
        return res

    def get_contract_name_by_id(self,contractId):
        res = self.get_contract_info_by_id(contractId)
        name = res['name']
        return name
        
    def get_account_balance(self):
        path = '/cashBalance/getcashbalancesnapshot'
        params = {'accountId':self._accountId}
        res = self._post(path,params)
        return res

    def get_cash_balance(self):
        res = self.get_account_balance()
        cash_bal = float(res['totalCashValue'])
        return cash_bal

    def get_account_bal_df(self):
        res = self.get_account_balance()
        df = pd.DataFrame()
        res = [res]
        if len(res)>0:
            df = pd.DataFrame(res)
        return df
    
    def filter_df_for_subaccount(self,df):
        if len(df)>0:
            accId = self._accountId
            df = df[df['accountId']==accId].copy()
            df = df.reset_index(drop=True)
        return df

    def get_exe_report_list(self):
        path = f'/executionReport/list'
        res = self._get(path)
        return res
    
    def get_exe_report_df(self):
        res = self.get_exe_report_list()
        df = pd.DataFrame()
        if len(res)>0:
            df = pd.DataFrame(res)
            df['contract'] = ''
            for i,r in df.iterrows():
                cid = r['contractId']
                name = self.get_contract_name_by_id(cid)
                df.at[i,'contract'] = name
            df = self.filter_df_for_subaccount(df)
        return df 
    
    def get_fills_list(self):
        path = f'/fill/list'
        res = self._get(path)
        return res
    
    def get_fills_df(self):
        res = self.get_fills_list()
        df = pd.DataFrame()
        if len(res)>0:
            df = pd.DataFrame(res)
            df['contract'] = ''
            for i,r in df.iterrows():
                cid = r['contractId']
                name = self.get_contract_name_by_id(cid)
                df.at[i,'contract'] = name
        return df 
        
    def get_orders_list(self):
        path = '/order/list'
        res =  self._get(path)
        return res
    
    def get_orders_df(self):
        res = self.get_orders_list()
        df = pd.DataFrame()
        if len(res)>0:
            df = pd.DataFrame(res)
            df['contract'] = ''
            for i,r in df.iterrows():
                cid = r['contractId']
                name = self.get_contract_name_by_id(cid)
                df.at[i,'contract'] = name
            df = self.filter_df_for_subaccount(df)
        return df

    
    def get_curr_pos(self):
        path = f'/position/list'
        res = self._get(path)
        return res
    
    def get_curr_pos_df(self):
        res = self.get_curr_pos()
        df = pd.DataFrame(res)
        df['name'] = ''
        for i,r in df.iterrows():
            cid = r['contractId']
            name = self.get_contract_name_by_id(cid)
            df.at[i,'name'] = name
        df = self.filter_df_for_subaccount(df)
        return df
    
    def get_curr_pos_by_symbol(self,symbol):
        df =  self.get_curr_pos_df()
        row =  df[df['name']==symbol]
        net_pos = 0
        if len(row)>0:
            net_pos = row['netPos'].values[0]
        return net_pos
    
    def cancel_all(self,symbol):
        all_orders = self.get_orders_df()
        if len(all_orders)>0:
            for i,r in all_orders.iterrows():
                if r['ordStatus'] != "Filled" and r['contract'] == symbol and r['ordStatus'] != 'Canceled':
                    ordId = r['id']
                    self.cancel_order(ordId)

    def cancel_order(self,id):
        id = int(id)
        path = f'/order/cancelorder'
        params = {
            'orderId': id,
             }
        response =  self._post(path,params)
        self.print_and_log(f'Order Cancel: {id}\n{response}')
        return response


    def place_market_order(self,symbol,side,size,clientOrderId=None):
        path = f'/order/placeorder'
        if clientOrderId == None:
            clientOrderId =  f"{symbol}_{ uuid.uuid4().hex[:7]}"
        params = {
            'clOrdId': clientOrderId,
            'accountSpec': self._api_name,
            'accountId': self._accountId,
            'action': side,
            'symbol': symbol,
            'orderQty': size,
            'orderType': "Market",
            'isAutomated': True
             }
        self.print_and_log(f'OrderSumbitted: \n{params}')
        response =  self._post(path,params)
        return response

    def place_limit_order(self,symbol,side,size,limit_price,clientOrderId=None):
        path = f'/order/placeorder'
        if clientOrderId == None:
            clientOrderId =  f"{symbol}_{ uuid.uuid4().hex[:7]}"
        params = {
            'clOrdId': clientOrderId,
            'accountSpec': self._api_name,
            'accountId': self._accountId,
            'action': side,
            'symbol': symbol,
            'orderQty': size,
            'price': limit_price, #use for single value like limit or stop
            'orderType': "Limit",
            'isAutomated': True
             }
        self.print_and_log(f'OrderSumbitted: \n{params}')
        response =  self._post(path,params)
        return response

    def place_stop_order(self,symbol,side,size,stop_price,clientOrderId=None):
        path = f'/order/placeorder'
        if clientOrderId == None:
            clientOrderId =  f"{symbol}_{ uuid.uuid4().hex[:7]}"
        params = {
            'clOrdId': clientOrderId,
            'accountSpec': self._api_name,
            'accountId': self._accountId,
            'action': side,
            'symbol': symbol,
            'orderQty': size,
            'stopPrice': stop_price,
            'orderType': "Stop",
            'isAutomated': True
             }
        self.print_and_log(f'OrderSumbitted: \n{params}')
        response =  self._post(path,params)
        return response

    def place_OSO(self,symbol,side,size,limit_price,tp_price,sl_price,clientOrderId=None):
        path = f'/order/placeOSO'
        if clientOrderId == None:
            clientOrderId =  f"{symbol}_{ uuid.uuid4().hex[:7]}"
        
        exit_action = 'Sell' if side=='Buy' else 'Buy'

        TP = {
                "action": exit_action,
                "orderType": 'Limit',
                "price": tp_price
            }
        SL = {
                "action": exit_action,
                "orderType": 'Stop',
                "stopPrice": sl_price
            }
        params = {
            'clOrdId': clientOrderId,
            'accountSpec': self._api_name,
            'accountId': self._accountId,
            'action': side,
            'symbol': symbol,
            'orderQty': size,
            'price': limit_price, #use for single value like limit or stop
            #'stopPrice': stop_price, # for stop limit order
            'orderType': "Limit",
            'isAutomated': True,
            'bracket1': SL,
            'bracket2': TP
             }
            
        
        self.print_and_log(f'OrderSumbitted: \n{params}')
        response =  self._post(path,params)
        return response

if __name__ == '__main__':
    pass