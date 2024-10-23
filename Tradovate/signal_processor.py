import os
import time
import numpy as np 
import pandas as pd
import uuid
from dotenv import load_dotenv
from os.path import join, dirname

dotenv_path = join(dirname(__file__), 'config.env') #py
#dotenv_path = join(dirname("__file__"), 'config.env') #Notebook
load_dotenv(dotenv_path)

from utils import *
from Tradovate import TVClient
tz_utc = timezone.utc

class SignalProcessor:
    
    def __init__(self,base_dir,logger=None):
        self._base_dir = base_dir
        self._reg_bal = f'balance/'
        self._reg_fills = f'fills/'
        self._reg_ords = f'orders/'
        self._reg_pos = f'positions/'
        self._reg_exes = f'executions/'
        self._subaccounts = self.prepare_subaccount_list()
        self.create_folders()
        self._logger = logger
    
    def prepare_subaccount_list(self):
        fname = os.environ.get('SUB_ACCOUNTS_FILE')
        df = pd.read_csv(fname,index_col=None)
        df['Name'] = df['Name'].astype(str)
        sub_accounts = df['Name'].to_list()
        return sub_accounts

    def get_dir(self,sub_acc,region):
        dir = f'{self._base_dir}{sub_acc}/{region}'
        return dir

    def create_folders(self):
        for subacc in self._subaccounts:
            dir_bal = self.get_dir(subacc,self._reg_bal)
            create_dir(dir_bal)
            dir_fills = self.get_dir(subacc,self._reg_fills)
            create_dir(dir_fills)
            dir_ords = self.get_dir(subacc,self._reg_ords) 
            create_dir(dir_ords)
            dir_pos = self.get_dir(subacc,self._reg_pos) 
            create_dir(dir_pos)
            dir_exes = self.get_dir(subacc,self._reg_exes) 
            create_dir(dir_exes)

    def print_and_log(self,msg):
        log_msg = f'{msg}'
        print(log_msg)
        if self._logger != None:
            self._logger.info(log_msg)

    def save_account_state(self,subacc,tvClient,suffix='bpa'):
        dt_now = datetime.now(tz_utc)
        dt_str = dt_now.strftime('%Y_%m_%d_%H_%M')

        ### Before processing alert #####------------
         # get balance 
        bal_bpa = tvClient.get_account_bal_df()
        if len(bal_bpa)>0:
            dir = self.get_dir(subacc,self._reg_bal)
            fname = f'{dir}Bal_{dt_str}_{suffix}.csv'
            bal_bpa.to_csv(fname,index=None)
        #print(f'Balance:\n {bal}')

        # get fills
        fills_bpa = tvClient.get_fills_df()
        if len(fills_bpa)>0:
            dir = self.get_dir(subacc,self._reg_fills)
            fname = f'{dir}Fills_{dt_str}_{suffix}.csv'
            fills_bpa.to_csv(fname,index=None)
        #print(f'Fills:\n {fills}')

        # get orders
        orders_bpa = tvClient.get_orders_df()
        if len(orders_bpa)>0:
            dir = self.get_dir(subacc,self._reg_ords)
            fname = f'{dir}Orders_{dt_str}_{suffix}.csv'
            orders_bpa.to_csv(fname,index=None)
        #print(f'Orders:\n {orders}')

        # get positions
        pos_bpa = tvClient.get_curr_pos_df()
        if len(pos_bpa)>0:
            dir = self.get_dir(subacc,self._reg_pos)
            fname = f'{dir}Positions_{dt_str}_{suffix}.csv'
            pos_bpa.to_csv(fname,index=None)

        # get executions
        exe_bpa = tvClient.get_exe_report_df()
        if len(exe_bpa)>0:
            dir = self.get_dir(subacc,self._reg_exes)
            fname = f'{dir}Executions_{dt_str}_{suffix}.csv'
            exe_bpa.to_csv(fname,index=None)

    def process_alert_for_sub_acc(self,tvClient,subacc,alert_json):
        ##### Processing alert #### ------------------
        market = alert_json['Market']
        tradeType = alert_json['TradeType']
        if tradeType == 'cancel_all':
            try:
                tvClient.cancel_all(market)
            except Exception as e:
                #print(f'Tradovate-api_error: {e}')
                self.print_and_log(f'Tradovate-api_error: {e}')
            finally:
                return 
        orderType = alert_json['OrderType']
        side = alert_json['Side']
        size = int(alert_json['Size'])
        clientId = f"botv8_{uuid.uuid4().hex[:7]}"
        limitPrice = 0
        stopPrice = 0
        tpPrice = 0
        slPrice = 0
        if orderType == 'OSO' and 'LimitPrice' in alert_json.keys():
            limitPrice = float(alert_json['LimitPrice'])
            tpPrice = float(alert_json['TPLimitPrice'])
            slPrice = float(alert_json['SLStopPrice'])
        elif orderType == 'limit' and 'LimitPrice' in alert_json.keys():
            limitPrice = float(alert_json['LimitPrice'])
        elif orderType == 'stop' and 'StopPrice' in alert_json.keys():
            stopPrice = float(alert_json['StopPrice'])
        place_order = True

        self.print_and_log(f'-----Processing alert for sub-account: {subacc}----')
        ########################
        if orderType == 'limit' and limitPrice == 0:
            self.print_and_log('Invalid limit price')
            return False
        if orderType == 'stop' and stopPrice == 0:
            self.print_and_log('Invalid stop price')
            return False
        if orderType == 'OSO':
            if limitPrice == 0:
                self.print_and_log('Invalid entry order limit price')
                return False
            if tpPrice == 0:
                self.print_and_log('Invalid TP limit price')
                return False
            if slPrice == 0:
                self.print_and_log('Invalid SL stop price')
                return False 
        
        #####------------------------------###########################
        if tradeType == 'long':
            ## buy size check
            if side=='buy':
                curr_bal = tvClient.get_cash_balance()
                #price =  get_market_price(market)
                #min_size = get_market_min_size(market)
                #max_size_ability = round_to_min_size(usd_bal/price,min_size)
                msg = f"alert_size: {size}, total_cash: {curr_bal}"
                self.print_and_log(msg)
                side = 'Buy'
                # if size>max_size_ability:
                #     size = max_size_ability
                # if min_size> size:
                #     place_order = False
                #     print_and_log('Can not place buy order, Not enough funds to buy min quantity')
            if side == 'sell':
                current_pos = tvClient.get_curr_pos_by_symbol(market)
                msg = f"alert_size: {size}, current_pos: {current_pos}"
                self.print_and_log(msg)
                side = 'Sell'
                if size>current_pos:
                    size = int(current_pos)
                if size==0:
                    place_order = False
                    self.print_and_log('Can not place sell order, size is zero')
        
        if tradeType == 'short':
            ## sell size check
            if side=='sell':
                curr_bal = tvClient.get_cash_balance()
                #price =  get_market_price(market)
                #min_size = get_market_min_size(market)
                #max_size_ability = round_to_min_size(usd_bal/price,min_size)
                msg = f"alert_size: {size}, total_cash: {curr_bal}"
                self.print_and_log(msg)
                side = 'Sell'
                # if size>max_size_ability:
                #     size = max_size_ability
                # if min_size> size:
                #     place_order = False
                #     print_and_log('Can not place buy order, Not enough funds to buy min quantity')
            if side == 'buy':
                current_pos = tvClient.get_curr_pos_by_symbol(market)
                msg = f"alert_size: {size}, current_pos: {current_pos}"
                self.print_and_log(msg)
                side = 'Buy'
                current_pos = abs(current_pos)
                if size>current_pos:
                    size = int(current_pos)
                if size==0:
                    place_order = False
                    self.print_and_log('Can not place sell order, size is zero')


        if place_order:
            self.print_and_log('Placing order...')
            #print('Placing order...') ###
            res = None
            try:
                if orderType == 'OSO':
                    res = tvClient.place_OSO(symbol=market,side=side,size=size,
                            limit_price=limitPrice,tp_price=tpPrice,sl_price=slPrice,clientOrderId=clientId)
                elif orderType == 'stop':
                    res = tvClient.place_stop_order(symbol=market,side=side,size=size,stop_price=stopPrice,clientOrderId=clientId)   
                elif orderType == 'limit':
                    res = tvClient.place_limit_order(symbol=market,side=side,size=size,limit_price=limitPrice,clientOrderId=clientId)
                elif orderType == 'market':
                    res = tvClient.place_market_order(symbol=market,side=side,size=size,clientOrderId=clientId)
                    
                #print(f'Response: {res}')
                time.sleep(1)
                self.print_and_log(f'Response: {res}')
            except Exception as e:
                #print(f'Tradovate-api_error: {e}')
                self.print_and_log(f'Tradovate-api_error: {e}')

        ### After processing alert #####------------
        # time.sleep(1)


    def process_signal(self,alert_json):
        sub_accounts =  self._subaccounts

        # Phase-1
        for subacc in sub_accounts:
            try: 
                tvClient = TVClient(accName=subacc,logger = self._logger)
                self.save_account_state(subacc,tvClient,'bpa')
                self.process_alert_for_sub_acc(tvClient,subacc,alert_json)  
            except Exception as e:
                self.print_and_log(f'{subacc} process_signal phase-1 error: {e}')

        # Phase-2
        self.print_and_log('Saving accounts status...')
        time.sleep(3)
        for subacc in sub_accounts:
            try:
                tvClient = TVClient(accName=subacc,logger = self._logger)
                self.save_account_state(subacc,tvClient,'apa')
            except Exception as e:
                self.print_and_log(f'{subacc} process_signal phase-2 error: {e}')


if __name__ == '__main__':
    sp = SignalProcessor('logs/')
    print(sp._subaccounts)
    



    



    

        
