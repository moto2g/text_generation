"""LSTMで文章自動生成モデルを作成する。
input:
    学習用テキスト in_text.txt (UTF-8)
output:
    モデル trained_model_～.h5
    ログ log_～.txt

usage: train.py maxlength step times skip_generate [path_in_model]

options:
    maxlength 切り出し文字数
    step 切り出し時の移動文字数
    times 学習回数
    skip_generate 学習の都度、自動生成するのをスキップするかどうか(True/False)
    path_in_model 学習済みモデルを読み込む場合のパス

ex: train.py 10 3 100 False trained
"""
import datetime
import random
import sys

import numpy as np

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop

from predict import generate_text


def write_log(f, text):
    f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\t" + text + "\n")


def main(maxlength=10, step=3, times=100, skip_generate=False, path_in_model=None):
    if path_in_model is None:
        path_in_model = ''

    with open('in_text.txt', mode='r', encoding='UTF-8') as f:
        text = f.read()
    print('Size of text: ', len(text))

    # textに含まれる文字のリスト
    chars = sorted(list(set(text)))
    print('Total chars:', len(chars))

    # 文字　から　index　を調べるための辞書
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # index　から　文字　を調べるための辞書
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # textをmaxlength文字ずつ切り出した文
    sentences = []
    # 切り出した文の次の1文字
    next_chars = []

    for i in range(0, len(text) - maxlength, step):
        sentences.append(text[i: i + maxlength])
        next_chars.append(text[i + maxlength])

    print('Number of sentences: ', len(sentences))

    # テキストをベクトル化する
    X = np.zeros((len(sentences), maxlength, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # モデルを定義する
    if path_in_model:
        # パス指定ありならモデルを読み込んで利用
        print("学習済みモデルを利用:" + path_in_model)
        model = load_model(path_in_model)
    else:
        # パス指定なしならモデルを最初から作成
        print('モデルを最初から作成')
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlength, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print("maxlength:{0} step={1} times={2}".format(maxlength, step, times))

    # ログファイル、学習済みモデルのファイル名の末尾につけるsuffixを決める(パラメータと日時)
    suffix = 'maxlength{0}_step{1}_times{2}_{3}'.\
        format(maxlength, step, times, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    path_log = "log_" + suffix + ".txt"
    with open(path_log, mode='w', encoding='UTF-8') as f:
        write_log(f, "path_in_model:{0} / maxlength:{1} / step:{2}".format(path_in_model, maxlength, step))

        for iteration in range(1, times + 1):
            write_log(f, '-' * 50)

            write_log(f, "学習 " + str(iteration) + "回目 開始")
            print("学習 " + str(iteration) + "回目")
            dt1 = datetime.datetime.now()
            model.fit(X, y, batch_size=128, epochs=1)
            dt2 = datetime.datetime.now()
            write_log(f, "学習時間\t" + str(dt2-dt1))

            if skip_generate:
                continue

            # 現在の学習状態で文章生成
            start_index = random.randint(0, len(text) - maxlength - 1)
            seed_sentence = text[start_index: start_index + maxlength]
            write_log(f, "----- Seed : [" + seed_sentence + "]")

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                write_log(f, "----- diveristy : " + str(diversity))
                print("----- diveristy : " + str(diversity))

                generated = generate_text(model, seed_sentence, diversity, char_indices, indices_char)

                write_log(f, "\n" + generated)
                print(generated)

    path_out_model = "trained_model_" + suffix + ".h5"
    model.save(path_out_model)


if __name__ == '__main__':
    if len(sys.argv) == 6:
        main(maxlength=int(sys.argv[1]),
             step=int(sys.argv[2]),
             times=int(sys.argv[3]),
             skip_generate=True if sys.argv[4] == 'True' else False,
             path_in_model=sys.argv[5])
    elif len(sys.argv) == 5:
        main(maxlength=int(sys.argv[1]),
             step=int(sys.argv[2]),
             times=int(sys.argv[3]),
             skip_generate=True if sys.argv[4] == 'True' else False)
    else:
        print(__doc__)
