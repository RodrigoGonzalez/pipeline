# Answer
s0 = 'azzf'
def palindrome_string(s):
    mid,left = divmod(len(s0), 2) # (3,0)
    s1 = bytearray(s0)

    while s1[:mid] != s1[-mid:][::-1]:
        for i in range(mid):
            a,b = s1[i], s1[-i-1]
            if a > b:
                s1[-i-1] = a
            elif a < b:
                s1[-i-1] = a
                s1[-i-2] += 1
            else:
                pass

        if 123 in s1:
            loc_inc = s1.index(chr(123))
            while 123 in s1:
                chr_loc = s1.index(chr(123))
                s1[chr_loc] = 97
            s1[chr_loc-1] += 1

    return str(s1.decode())


# Old that takes forever
s1 = s0[:mid + left] + s0[:mid][::-1]
s1, s0 = bytearray(s1), bytearray(s0)

while s1[:mid] != s1[-mid:][::-1]:
    i = 1
    while s1[-i] == 122:
        s1[-i] = 97
        if s1[-i-1] != 122:
            s1[-i-1] += 1
        else:
            i += 1
    if
    s1[-1] += 1
    print s1

print s1
