while b[-i] == 122:
    b[-i] = 97
    if b[-i-1] != 122:
        b[-i-1] += 1
    else:
        i += 1
    
