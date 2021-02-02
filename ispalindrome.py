import sys
print("He","llo", sys.argv[1])

import pylab as pl



pl.plot(list(range(10)))
pl.figure()
pl.plot(list(range(20)))
pl.savefig("test.png")
pl.show()

