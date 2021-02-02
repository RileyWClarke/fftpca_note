import sys

def reverse(s):
    rs = ""
    for i in range(1,len(s)+1):
        rs += s[-i]
    return rs

for arg in sys.argv[1:]:

    if len(arg) == 1:
        print("{} is only one letter long!".format(arg))

    elif arg == reverse(arg):
        print("{} is a palindrome!".format(arg))

    else:
        print("{} is not a palindrome!".format(arg))

