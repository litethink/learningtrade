def reckon_position(pw,trade=1000,b=1,account=100,positive=2/100,nagative=2/100):
    market  = np.random.binomial(1,pw,trade)
    kelly_value  = (pw * b - (1 - pw)) / b
    for i in market:
    if i == 1:
        account +=  account * kelly_value * positive
    else:
        account -= account * kelly_value * nagative
    return account 
