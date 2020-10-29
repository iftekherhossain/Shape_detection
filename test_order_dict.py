from collections import OrderedDict
from pytictoc import TicToc

t = TicToc()

a = OrderedDict()
b = OrderedDict()
for i in range(10):
    a[i] = list(str(i))
for i in range(10):
    b[i] = list(str(i+1))
# print(a)
t.tic()
t.toc()
a_inv = {v: k for k, v in a.items()}
#a_inv = dict(zip(a.values(), a.keys()))
# a_inv = dict(map(reversed, a.items()))
# print(a_inv)
t.toc()

p = [a, b]

for e in p:
    a_inv = {v: k for k, v in e.items()}
    print(a_inv)
