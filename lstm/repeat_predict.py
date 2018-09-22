"""学習済みモデルを使って文章を自動生成する。（ランダムシード＆リピート）
10個のシードで、diversityを変えながら生成する。

input:
    学習で使ったテキスト in_text.txt (UTF-8)
output:
    生成ログ generated_～.txt (UTF-8)

usage: repeat_predict.py path_in_model

options:
    path_in_model 学習済みモデルのパス
"""
import datetime
import random
import sys

from keras.models import load_model

from predict import generate_text


def main(path_in_model):

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

    max_length = int(model.layers[0].get_input_at(0).get_shape()[1])

    path_log = "generated_{0}_{1}.txt".format(max_length, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    with open(path_log, mode='w', encoding='UTF-8') as f:
        for _ in range(10):
            start_index = random.randint(0, len(text) - max_length - 1)
            seed_sentence = text[start_index: start_index + max_length]

            f.write("-" * 20)
            f.write('\n')
            f.write("seed=" + seed_sentence)
            f.write('\n')
            for diversity in [0.2, 0.5, 0.7, 1.0, 1.2]:
                f.write("-" * 20)
                f.write('\n')
                f.write("diversity=" + str(diversity))
                f.write('\n')
                f.write("-" * 20)
                f.write('\n')

                generated_sentences = generate_text(model, seed_sentence, diversity, char_indices, indices_char)

                f.write(generated_sentences + "\n")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(path_in_model=sys.argv[1])
    else:
        print(__doc__)
