"""
CS620
HW3
@author: Joseph Cannella
"""
import pandas as pd
import xml.etree.ElementTree as et

# Rad in the CSV Data to a Data Frame
csv_data = pd.read_csv('SP500_ind.csv')

# Read in the XML Data to a Dictionary
xml_tree = et.parse('SP500_symbols.xml')
xml_dict = xml_tree.getroot()


# Generate list of unique symbol values
ticker = csv_data['Symbol'].unique()

print(xml_dict[0].attrib)

def ticker_find(xml_dict, ticker):
    """This function takes in the xml_dict and a
    Symbol (ticker). Return the name of the ticker
    Ex: for ticker “A”, the function returns Agilent Technologies Inc
    """
    for child in xml_dict:
        if child.attrib['ticker'] == ticker:
            return child.attrib['name']
    return "No data in SP500"


def calc_avg_open(csv_data, ticker):
    """This function takes in the csv_data and a ticker.
    Return the average opening price for the stock as a float.
    """
    return csv_data[csv_data['Symbol'] == ticker]['Open'].mean()

def vwap(csv_data, ticker):
    """This function takes in the csv_data and a ticker. Return the volume weighted average
    price (VWAP) of the stock. In order to do this, first find the average price of the stock on
    each day. Then, multiply that price with the volume on that day. Take the sum of these
    values. Finally, divide that value by the sum of all the volumes.
    (hint: average price for each day = (high + low + close)/3)
    """
    context = csv_data[csv_data['Symbol'] == ticker]
    return (context['Volume']*(context['High'] + context['Low'] + context['Close'])/3.0).sum()/context['Volume'].sum()


for t in ticker:
    name = ticker_find(xml_dict, t)
    avg_open = calc_avg_open(csv_data, t)
    vwap_val = vwap(csv_data, t)
    print(f'{name} {avg_open} {vwap_val}')