# 15-9
def pwD(pwd):
    pwdlen = len(pwd)
    if pwdlen < 5:
        raise Exception("the passwrods length isn't enough.")
    elif pwdlen > 8:
        raise Exception("the passwords length is too long.")
    print('passwords length is correct.')

for pwd in ('aaabbbccc', 'aaa', 'aaabbb'):
    try:
        pwD(pwd)
    except Exception as err:
        print("passwords length check has error: ", str(err))