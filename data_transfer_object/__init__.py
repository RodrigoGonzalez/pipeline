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

# Nested Loops: Comfortable Numbers
seg,pairs = xrange(L,R+1),[]
for x in seg:
    bound = sum(map(int, str(x)))
    pairs.append([(x,y) for y in range(x-bound, x+bound+1) if (y!=x and y in seg)])

P = [(x,y) for x,y in sum(pairs,[]) if x < y]

# Nested Loops: Weak Numbers
divs = lambda x: [y for y in range(1,x) if not x%y]
D = [len(divs(x)) for x in xrange(1,n+1)]
Weak = [len(filter(lambda x: x > D[i], D[:i])) for i in xrange(1,n+1)]

# Rectangle Rotation
a,b = 6,4
P = [(x,y) for x in range(-b/2,b/2+1) for y in range(-a/2,a/2+1)]


def rectangleRotation(a, b):
    x_new = lambda x,y: (x-y)*math.sqrt(2)/2
    y_new = lambda x,y: (x+y)*math.sqrt(2)/2
    slope = lambda x1,y1,x2,y2: (y2-y1)/float((x2-x1))
    intercept = lambda slope,x,y: y - x*slope
    y_calc = lambda x,slope,intercept: slope*x + intercept
    sign_pos = lambda y,y_calc: y_calc - y >= 0
    sign_neg = lambda y,y_calc: y_calc - y <= 0
    P = [(x,y) for x in range(-b/2,b/2+1) for y in range(-a/2,a/2+1)]
    x1,y1,x2,y2,x3,y3,x4,y4 = -b/2., a/2., b/2., a/2., -b/2., -a/2.,b/2., -a/2.,
    slope1_4 = slope(x_new(x1,y1),y_new(x1,y1),x_new(x4,y4),y_new(x4,y4))
    slope1_2 = slope(x_new(x1,y1),y_new(x1,y1),x_new(x2,y2),y_new(x2,y2))
    slope2_3 = slope(x_new(x2,y2),y_new(x2,y2),x_new(x3,y3),y_new(x3,y3))
    slope3_4 = slope(x_new(x3,y3),y_new(x3,y3),x_new(x4,y4),y_new(x4,y4))
    int1_4 = intercept(slope1_4,x1,y1)
    int1_2 = intercept(slope1_2,x2,y2)
    int2_3 = intercept(slope2_3,x3,y3)
    int3_4 = intercept(slope3_4,x4,y4)
    pts_inside = []
    for x,y in P:
        y_calc14 = y_calc(x,slope1_4,int1_4)
        y_calc12 = y_calc(x,slope1_2,int1_2)
        y_calc23 = y_calc(x,slope2_3,int2_3)
        y_calc34 = y_calc(x,slope3_4,int3_4)
        if all(sign_pos(sign_pos(y,y_calc12)), sign_pos(y,y_calc14), sign_neg(y,y_calc23), sign_neg(y,y_calc34)):
            pts_inside.append((x,y))
    return len(pts_inside)
