# ch15-4
def wordsNum(fn):
    # adapt for the English file, input the article file name, can calculate the article of number
    try:
        with open(fn) as file_Obj:
            data = file_Obj.read()  # use default mode=r to open the file
    except FileNotFoundError:
        print(f"Can't find the {fn} file.")
    else:
        wordList = data.split()
        print(f"{fn} article of words number is {len(wordList)}")   # words number

file = 'data15_4.txt'
wordsNum(file)