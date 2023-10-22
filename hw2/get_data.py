import pandas as pd
import requests
from bs4 import BeautifulSoup


# Create a URL object
def get_dataframe(url, dataset, table_id, header_tag='th', row_tag='tr', data_tag='td'):
    """
    url: website where the data is stored
    example: https://climatology.nelson.wisc.edu/first-order-station-climate-data/madison-climate/lake-ice/history-of-ice-freezing-and-thawing-on-lake-mendota/
    return: a pandas dataframe of the data
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'lxml')
    table1 = soup.find('table', id=table_id)
    headers = []
    for i in table1.find_all(header_tag):
        header = i.text
        headers.append(header)
    data = pd.DataFrame(columns=headers)
    for j in table1.find_all(row_tag)[1:]:
        row_data = j.find_all(data_tag)
        row = [i.text for i in row_data]
        length = len(data)
        data.loc[length] = row
    ###################################
    # only to format these specific table
    data['Winter'] = data['Winter'].apply(lambda x: x.split('-')[0])
    data.drop(labels=['Freeze-Over Date', 'Thaw Date'], axis=1, inplace=True)
    data.set_index('Winter', inplace=True)
    colname = "Days_" + dataset
    data.rename(columns={"Days of Ice Cover": colname}, inplace=True)
    data.loc[:, colname].replace(to_replace='–', value=None, inplace=True)
    data.dropna(axis=0, inplace=True)
    return data
