"""Script to gather market data from OKCoin Spot Price API."""
import requests
from pytz import utc
from datetime import datetime
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler

client = MongoClient()
database = client['okcoindb']
collection = database['historical_data']
tf = '%Y-%m-%dT%H:%M:%S'


def tick():
    """Gather market data from OKCoin Spot Price API and insert them into a
       MongoDB collection."""
    ticker = requests.get('https://www.okcoin.com/api/spot/v3/instruments/ticker').json()
    depth = requests.get('https://www.okcoin.com/api/spot/v3/instruments/BTC-USDT/book?size=60&depth=0.2').json()
    ts = ticker[1]['timestamp'][:-5]
    date = datetime.strptime(ts,tf)
    price = float(ticker[0]['best_ask'])
    v_bid = sum([float(bid[1]) for bid in depth['bids']])
    v_ask = sum([float(ask[1]) for ask in depth['asks']])
    collection.insert_one({'date': date, 'price': price, 'v_bid': v_bid, 'v_ask': v_ask})
    print(date, price, v_bid, v_ask)


def main():
    """Run tick() at the interval of every ten seconds."""
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=10)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == '__main__':
    main()
