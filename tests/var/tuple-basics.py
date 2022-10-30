t1 = (1,)
print(t1[0])

t2 = (True, 1, False)
print(1 if t2[2] else 0)

t3 = (1, t2, t1)
print(t3[1][1])
