"""学習済みモデルを使って文章を自動生成する。
input:
    学習で使ったテキスト in_text.txt (UTF-8)
output:
    生成した文章 標準出力に表示

usage: predict.py path_in_model seed_sentence diversity

options:
    path_in_model 学習済みモデルのパス
    seed_sentence シード文字列
    diversity 係数

ex: predict.py usingmodel.h5 hogehoge 1.0
"""
import sys

import numpy as np

from keras.models import load_model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed_sentence, diversity, char_indices, indices_char):
    generated = seed_sentence
    w_seed_sentence = seed_sentence
    max_length = int(model.layers[0].get_input_at(0).get_shape()[1])

    for _ in range(800):
        x = np.zeros((1,
                      max_length,
                      int(model.layers[0].get_input_at(0).get_shape()[2])))
        t = 0
        for i, char in enumerate(w_seed_sentence):
            if char not in char_indices.keys():
                pass
            else:
                x[0, t, char_indices[char]] = 1.

            t += 1
            if t == max_length:
                break

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        w_seed_sentence = w_seed_sentence[1:] + next_char

    return generated


def main(path_in_model, seed_sentence, diversity):

    with open('in_text.txt', mode='r', encoding='UTF-8') as f:
        text = f.read()

    # textに含まれる文字のリスト
    chars = sorted(list(set(text)))
    # 文字　から　index　を調べるための辞書
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # index　から　文字　を調べるための辞書
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # モデルを読み込む
    model = load_model(path_in_model)

    # 文書生成
    generated_sentences = generate_text(model, seed_sentence, diversity, char_indices, indices_char)
    print(generated_sentences)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(path_in_model=sys.argv[1], seed_sentence=sys.argv[2], diversity=float(sys.argv[3]))
    else:
        print(__doc__)



