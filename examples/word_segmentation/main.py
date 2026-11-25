# pip install jieba scikit-learn
import os
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix

corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]

keywords = jieba.analyse.extract_tags(str(corpus), topK=3, withWeight=True)
print(keywords)

title_files = []

def read_news():
    path = "C:\\news"
    files = os.listdir(path)
    corpus = []
    for file in files:
        with open(path + "\\" + file, "r", encoding='utf-8-sig') as fp:
            corpus.append(fp.read())
        title_files.append(file)
    return corpus

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def jieba_TFIDF(corpus):
    stopwords_path = os.path.join(os.path.dirname(__file__), "stopwords.txt")
    stopwords = stopwordslist(stopwords_path)
    words = []
    text = []
    textlist = []
    for item in corpus:
        seg_list = jieba.lcut(item, cut_all=False)
        seg_list = [w for w in seg_list if w not in stopwords]
        words.append(seg_list)
        text.append(" ".join(seg_list))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    keyword = vectorizer.get_feature_names_out()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    assert isinstance(tfidf, csr_matrix)
    weight = tfidf.toarray()
    for i in range(len(weight)):
        textlist.append(list(zip(keyword, weight[i])))
    for i in range(len(title_files)):
        print(title_files[i])
        textlist[i] = sorted(textlist[i], key=lambda x: x[1], reverse=True)
        for j in range(5):
            print(textlist[i][j])
    return textlist
