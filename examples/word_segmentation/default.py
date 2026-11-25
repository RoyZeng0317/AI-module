
article = [
    'this is the first file',
    'this is the second file'
    'and the third one one',
    'is this the first second file'
]
dic = corpora.Dictionary(wordList)
newArticle = [dict.doc2bow(i) for i in wordList]

tfidf = models.TfidfModel(newArticle)
tfidf.save("TFIDFmodel.tfidf")

tfidf.models.TfidfModel.load("TFIDFmodel.tfidf")
