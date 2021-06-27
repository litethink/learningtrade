
class TradeOperator(object):
    """docstring for Trade_Operator"""
    def __init__(self,capital):
        self.balance        = capital
        self.fewest_volume  = 5
        self.charge         = 0.002
        #accounts:current price, position, cost 
        self.accounts    = dict()
        self.price_history   = dict()
        self.profits         = dict()

    #
    def init_symbol(self, symbol,current_price):
        self.price_history[symbol] = [current_price,]
        self.accounts[symbol] = dict(
                    current_price    = current_price ,
                    total_position = 0
        )
        self.profits[symbol] = list()
    def update_price(self,symbol,price):
        self.price_history[symbol].append(price)
        self.accounts[symbol]["current_price"] = price

    def buy(self,symbol,volume):
        _price = self.accounts[symbol]["current_price"]
        assert self.balance >= volume
        assert volume > self.fewest_volume
        assert _price > 0
        _cost = (_price + _price * self.charge)
        _new_postion = volume/_cost
        _last_position = self.accounts[symbol]["total_position"]
        self.balance -= volume
        _last_cost = self.accounts[symbol].get("cost")
        if _last_cost:
            _total = self.accounts[symbol]["total_position"]
            _last_position = self.accounts[symbol]["total_position"]
            self.accounts[symbol]["cost"] = (_last_cost * _last_position + _cost * _new_postion)/ \
            (_last_position + _new_postion)
        else:
            self.accounts[symbol]["cost"] = _cost
        self.accounts[symbol]["total_position"] += _new_postion



    def sell(self,symbol,amount):
        price = self.accounts[symbol]["current_price"]
        _cost = (price - price * self.charge)
        assert amount * _cost > self.fewest_volume
        assert amount <= self.accounts[symbol]["total_position"]
        self.accounts[symbol]["total_position"] -= amount
        _volume = amount * _cost
        self.balance += _volume
        self.profits[symbol].append(_volume - amount * self.accounts[symbol]["cost"])

  
to = TradeOperator(capital=10000)
to.init_symbol("btc",1)
to.price_history
to.accounts
to.buy("btc",10)
to.buy("btc",10)
to.update_price("btc",2)
to.buy("btc",30)
to.buy("btc",30)
to.balance
to.accounts
to.sell("btc",to.accounts["btc"]["total_position"])

    








