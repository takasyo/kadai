import numpy as np
import csv
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def make_word2id(): #make_word2id関数の定義
    word2id = {} #ディクショナリ word2id = {'単語':単語id}

    with open('train.txt', 'r', encoding='utf_8') as f:
        morphemes = [s.strip()[1:] for s in f.readlines()]

    for line in morphemes:
        for word in line.split():
            if word not in word2id: #word2idにwordがなかったら
                word2id[word] = len(word2id)

    with open('dic.txt', 'w', encoding='utf_8') as f2:
        for word in word2id:
            f2.write('{},{}\n'.format(word, word2id[word]))
    return word2id

word2id = make_word2id() #make_word2id関数の実行

def make_feature(word2id): #make_feature関数の定義
    for text_name in ['train', 'test']:

        with open(text_name + '.txt', 'r', encoding='utf_8') as f:
            morphemes = [s.strip()[1:] for s in f.readlines()]

        bow_set = []
        for line in morphemes:
        #  bow = [0 for i in range(len(word2id))]
            bow = [0] * len(word2id)
            for word in line.split():
                if word in word2id:
                    if bow[word2id[word]] == 0:
                        bow[word2id[word]] = 1
            bow_set.append(bow)

        with open(text_name + '_feature.txt', 'w') as f:
            for i in range(len(bow_set)):
                for j in range(len(bow_set[i])):
                    f.write(str(bow_set[i][j]))
                    if j == len(bow_set[i])-1:
                        f.write('\n')
                    else:
                        f.write(',')

make_feature(word2id) #make_feature関数の実行

def load_data(X_text,y_text): #load_data関数の定義
    # 特徴量の読み込み
    with open(X_text,encoding="utf_8") as f:
        reader = csv.reader(f, delimiter=',')
        X_data = [row for row in reader]

    # カテゴリーIDの読み込み
    with open(y_text,encoding="utf_8") as f:
        reader = csv.reader(f, delimiter='\t')
        y_data = [row[0] for row in reader]

    # str型をfloat型に変換
    for i in range(len(X_data)):
        X_data[i] = [float(n) for n in X_data[i]]

    y_data = [float(n) for n in y_data]

    return X_data,y_data


# 学習データの読み込み
X_train,y_train = load_data('train_feature.txt','train.txt')

# 評価データの読み込み
X_test,y_test = load_data('test_feature.txt','test.txt')

# ロジスティック回帰モデルを学習
lr = LogisticRegression()
lr.fit(X_train,y_train)
joblib.dump(lr, 'model.joblib')

# 学習モデルの読み込み
lf = joblib.load('model.joblib')

# 正解率表示
print(f'Accuracy: {accuracy_score(y_test, lf.predict(X_test))}')

#特徴量の重みの確認
category = ['Business', 'Entertainment', 'Science and Technology', 'Health']
id2word = {v: k for k, v in word2id.items()}
for list in range(lf.coef_.shape[0]):
    largest_index = np.argsort(lf.coef_[list][:])[-11:-1]
    smallest_index = np.argsort(lf.coef_[list][:])[0:9]
    print(category[list] + " largest:")
    print([id2word[x] for x in largest_index])
    print([lf.coef_[list][x] for x in largest_index])
    print(category[list] + " smallest:")
    print([id2word[x] for x in smallest_index])
    print([lf.coef_[list][x] for x in smallest_index])

