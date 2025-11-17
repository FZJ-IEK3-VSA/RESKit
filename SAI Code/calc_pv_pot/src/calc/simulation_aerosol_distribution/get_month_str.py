MONTH_DICT = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'  }

def get_month_str(month):
    '''
    Args: month as number
    Return: 
    - month_str = month as str
    - month_calc = number of month in range 1-12 for calculation (if > 12)
    '''

    # for month > 12 
    for i in range(1,13):
        if month % i == 0:
            month_str = MONTH_DICT[i]

    # if month > 12:
    #     for key, val in MONTH_DICT.items():
    #         if val == month_str:
    #             month_calc = key
    # else:
    #     month_calc = month

    return  month_str  #, month_calc




