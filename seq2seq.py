"""

Sequence 2 Sequence

https://github.com/YoheiFukuhara/keras-for-beginner/blob/master/Keras11_seq2seq.ipynb

モデルの構造はこう
※APIの引数情報は一旦無視

・終端記号の追加
・逆順学習


■ 学習時

    ◆ encoder側

enc_input = Input(shape=(1学習サンプル長, 次元数)) <-EmbeddingもOK？
# enc_input.shape: (None, 1学習サンプル長, 次元数) <-例のトリプル(データ層数, 1学習サンプル長, 次元数

enc_lstm = LSTM(中間層数)
enc_output, st_h, st_c = enc_lstm(enc_input)
# enc_LSTM.shape: (None, 中間層数) <- これは使わない
# st_h.shape: (None, 中間層数) <- こいつらがdecoderのLSTM層とつながり、
# st_c.shape: (None, 中間層数) <- encoder側のシーケンス情報を伝播させる
# ここのLSTMは上記3つのトリプルを持っているが後ろ二つのみdecoderで使用

    ◆ decoder側

# decoderの入出力層:
# 入力層：教師サンプル（y) <- yなのに入力側にいる妙 先頭は開始記号
# 出力層：教師サンプル（y)を1時刻うしろにずらしたシーケンス


dec_input = Input(例のトリプル(データ層数, 1教師サンプル長, 次元数))
# dec_input.shape: (None, 1教師サンプル長, 次元数)

dec_lstm = LSTM(中間層数)
dec_output, _, _ = dec_lstm(dec_input, initial_state=[st_h, st_c])
# dec_output
# 入力側.shape: (None, 1教師サンプル長, 次元数), (None, 中間層数), (None, 中間層数)
# 出力側.shape: (None, 1教師サンプル長, 中間層数), _(None, 中間層数), _(None, 中間層数)
# 出力側の後ろ二つのトリプル（状態情報）は使わない

dec_dense = Dense(次元数)
dec_output = dec_dense(dec_output)
# dec_dens.shape: (None, 1教師サンプル長, 次元数)

model = Model([enc_input, dec_input], dec_outpu)
# encoder側入力層、decoder側入力層、デコーダ側出力層
# でモデル構築


■ 予測時

    ◆ encoder側

pred_enc_model = Model(enc_input, [st_h, st_c])
# Modelの入力はいずれも学習時のモデルを流用

    ◆ decoder側

pred_dec_input = Input(1, 次元数)
# pred_dec_input.shape: (None, 1予測サンプルの1要素, 次元数)

pred_dec_st_in = [Input(中間層数), Input(中間層数)]
# 伝搬する状態情報　それぞれ　[隠れ層, メモリセル]

pred_dec_output, pred_dec_st_h, pred_dec_st_c =
    dec_lstm(pred_dec_input, initial_state=pred_dec_st_in)
pred_dec_st = [pred_dec_st_h, pred_dec_st_c]
# dec_lstmは学習済みモデル

pred_dec_output = dec_dense(pred_dec_output)
# dec_denseは学習済みモデル

pred_dec_model = Mode(
    [pred_dec_input] + pred_dec_st_in, 
    [pred_dec_output] + pred_dec_st
)
# + はリストの結合
# 1サンプルが出力される

"""

import os
import numpy as np
import pandas as pd
import pickle
import sudachipy
from sudachipy import dictionary
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 作業ディレクトリ
work_dir = 'D:/DataSet/deep-learning-from-scratch-2-master/dataset/'
# データセット
dataset_file = 'addition.txt'
tokenizer_file = 'tokenizer.pickle'
train_file = 'train.txt'
test_file = 'test.txt'
# Seq2Seqモデル
encoder_file = 'encoder.h5' # HDF5ファイルに保存すればモデル構造のみならず学習結果も保存可能
decoder_file = 'decoder.h5'

# モデルパラメータ
N_RNN_enc = 7
N_RNN_dec = 5 # 
N_MID = 16 # 中間層層数


def make_dataset(max_sample_num):
    """
    データセット作成
    Seq2Seqモデル構築とは別途やった方がいい
    
    """
    
    df_all = pd.read_csv(work_dir+dataset_file, sep='_', header=None)
    #print(df_all.head(10))

    # 元データをシーケンスにする
    # Seq2Seqはencoder側の入力X_enc, decoder側の入力X_dec, 出力Y_decの3データがある
    X_enc_seq = []
    X_dec_seq = [] # これの存在がややこしい　本来教師データYであるはずのものがdecoderでは入力にも使われる
    Y_dec_seq = []
    
    sample_num = min(max_sample_num, df_all.shape[0])
    for i in range(sample_num):
        # いずれも空白区切りに変換
        
        # encoder側の入力層X
        x_seq = df_all[0].iloc[i].replace(' ','')
        X_enc_seq.append(' '.join([x for x in x_seq]))
        
        y_seq = '_' + str(df_all[1].iloc[i]) + '$' # 先頭にBOS記号(_)、末尾にEOS記号($)を付与
        # decoder側の入力層X yなのにX（入力側）になる妙
        X_dec_seq.append(' '.join([y for y in y_seq[:-1]]))
        # decoder側の出力層Y
        Y_dec_seq.append(' '.join([y for y in y_seq[1:]]))
    
    #print(X_enc_seq) # ['1 6 + 7 5', '5 2 + 6 0 7', ... ]
    #print(X_dec_seq) # ['_ 9 1'    , '_ 6 5 9'    , ... ]
    #print(Y_dec_seq) # ['9 1 $'    , '6 5 9 $'    , ... ]
    
    # x,yをトークナイズ
    # デフォルトだと演算子+と_と$をフィルタするので引数filtersで再定義してこれらを除外
    tokenizer = Tokenizer(filters='!"#%&()*,./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(X_enc_seq + X_dec_seq + Y_dec_seq)
    # うしろにpaddingするには引数padding='post'とする（デフォルトはpadding='pre'で前方パディング）
    X_enc = pad_sequences(tokenizer.texts_to_sequences(X_enc_seq), maxlen=N_RNN_enc, padding='post')
    X_dec = pad_sequences(tokenizer.texts_to_sequences(X_dec_seq), maxlen=N_RNN_dec, padding='post')
    Y_dec = pad_sequences(tokenizer.texts_to_sequences(Y_dec_seq), maxlen=N_RNN_dec, padding='post')
    #print(X_enc) # [[ 1  8  2  5  6  0  0], [ 6  4  2  8 11  5  0 ], ... ]
    #print(X_dec) # [[ 7  4  1  0  0]      , [ 7  8  6  4  0]       , ... ]
    #print(Y_dec) # [[ 4  1  0  0  0]      , [ 8  6  4  0  0]       , ... ]
    
    # train test 分割
    X_enc_train, X_enc_test, X_dec_train, X_dec_test, Y_dec_train, Y_dec_test = \
        train_test_split(X_enc, X_dec, Y_dec, test_size=0.1, random_state=0)
    
    # ファイル保存
    with open(work_dir + tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_dataset(filename, X_enc, X_dec, Y_dec):
        with open(work_dir + filename, 'w') as f:
            for i in range(len(X_enc)):
                x_enc = ' '.join([str(x) for x in X_enc[i]])
                x_dec = ' '.join([str(x) for x in X_dec[i]])
                y_dec = ' '.join([str(y) for y in Y_dec[i]])
                f.write(f'{x_enc},{x_dec},{y_dec}\n')
                
    save_dataset(train_file, X_enc_train, X_dec_train, Y_dec_train)
    save_dataset(test_file, X_enc_test, X_dec_test, Y_dec_test)
    
    return


def load_dataset():
    """
    保存したデータセットをロードする
    こうなってくるとデータセット一式まるめたクラスみたいのにした方がいいような気がするなー
    データローダーみたいな？
    
    """
    def load_train_test(filename):
        df_all = pd.read_csv(filename, sep=',', header=None)
        
        sample = [x for x in df_all[0].iloc[0].split(' ')]
        X_enc = np.zeros(shape=(0, len(sample)))

        sample = [x for x in df_all[1].iloc[0].split(' ')]
        X_dec = np.zeros(shape=(0, len(sample)))

        sample = [x for x in df_all[2].iloc[0].split(' ')]
        Y_dec = np.zeros(shape=(0, len(sample)))
        
        for i in range(df_all.shape[0]):
            # X_enc
            seq = np.array([np.int32(x) for x in df_all[0].iloc[i].split(' ')])
            seq = np.reshape(seq, (1, len(seq)))
            X_enc = np.insert(X_enc, X_enc.shape[0], seq, axis=0)
            # X_dec
            seq = np.array([np.int32(x) for x in df_all[1].iloc[i].split(' ')])
            seq = np.reshape(seq, (1, len(seq)))
            X_dec = np.insert(X_dec, X_dec.shape[0], seq, axis=0)
            # Y_dec
            seq = np.array([np.int32(x) for x in df_all[2].iloc[i].split(' ')])
            seq = np.reshape(seq, (1, len(seq)))
            Y_dec =  np.insert(Y_dec, Y_dec.shape[0], seq, axis=0)
            
        return X_enc, X_dec, Y_dec
    
    X_enc_train, X_dec_train, Y_dec_train = load_train_test(work_dir + train_file)
    X_enc_test, X_dec_test, Y_dec_test = load_train_test(work_dir + test_file)
    
    train = {'X_enc': X_enc_train, 'X_dec': X_dec_train, 'Y_dec': Y_dec_train}
    test = {'X_enc': X_enc_test, 'X_dec': X_dec_test, 'Y_dec': Y_dec_test}
    
    # tokenizer
    with open(work_dir + tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return  train, test, tokenizer


def train_Seq2Seq(X_enc, X_dec, Y_dec, tokenizer, X_tec, X_ted, Y_ted):
    """
    Seq2Seqの学習  
    学習と予測で異なるモデルを用いる。こちらは学習だけ実施
    
    学習終了時に予測用のencoder, decoderを生成し、
    ファイルとして保存しておくと毎回再学習する必要がないので便利だよ
    その時はテストデータ側を保存しておく
    
    X_enc [[ 1  8  2  5  6  0  0], [ 6  4  2  8 11  5  0 ], ... ]
    X_dec [[ 3 10  0  0]         , [ 3  8  6  0]          , ... ]
    Y_dec [[10  1  0  0]         , [ 8  6 10  0]          , ... ]

    # 時系列予測みたいにYの出力そのものを使うタスクならいいが
    # 文書生成や分類問題の場合はonehot化する
    x_enc(トークン) 1次元 -> one-hot 単語数次元 -> LSTM
    x_dec(トークン) 1次元 -> one-hot 単語数次元 -> LSTM -> Dense(Softmax) 単語数次元
    y_dec(トークン) 1次元 -> one-hot 単語数次元 <-Dense(Softmax) 単語数次元 <- LSTM
    y_decからx_decに再帰入力するときにpredict_classesでトークンに戻す
    """
    
    N_IN = len(tokenizer.word_index) + 1  # 入力次元数　
    N_OUT = len(tokenizer.word_index) + 1 # 出力次元数
    
    # Seq2Seqモデルの構築

    """
    学習時のモデルを構築
    """
    
    # まずはEncoder側
    enc_input = keras.layers.Input(shape=(X_enc.shape[1], N_IN), name='enc_input') # enc入力層
    enc_lstm = LSTM(N_MID, return_state=True, name='enc_lstm') # 中間層 return_stateはTrue
    enc_output, st_h, st_c = enc_lstm(enc_input)
    # enc_outputはdiscard

    # つぎにDecoder側
    dec_input = keras.layers.Input(shape=(X_dec.shape[1], N_IN), name='dec_input') # dec入力層
    dec_lstm = LSTM(N_MID, return_sequences=True, return_state=True, name='dec_lstm') # 中間層 returnなんちゃらはTrue
    dec_output, _, _ = dec_lstm(dec_input, initial_state=[st_h, st_c])
    
    dec_dense = Dense(N_OUT, activation='softmax', name='dec_dense') # 全結合層 予測時にも流用
    dec_output = dec_dense(dec_output) # dec出力層
    
    #　入出力層３つを統合してひとつのモデルにする
    model = Model([enc_input, dec_input], dec_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    
    # 学習
    # X_encを逆順にする
    for i, x in enumerate(X_enc):
        X_enc[i] = x[::-1]
    
    # one-hot化する
    X_enc_cat = keras.utils.to_categorical(X_enc, num_classes=N_IN)
    X_dec_cat = keras.utils.to_categorical(X_dec, num_classes=N_IN)
    Y_dec_cat = keras.utils.to_categorical(Y_dec, num_classes=N_OUT)
    
    history = model.fit([X_enc_cat, X_dec_cat], Y_dec_cat, batch_size=8, epochs=100)
    loss = history.history['loss']
    plt.plot(np.arange(len(loss)), loss)
    plt.show()
    
    """
    予測時のモデル構築までやってしまう
    
    encoder側：全部流用
    decoder側：LSTM層とDense層を流用
    
    モデルは分離。encoderで得た中間層状態はdecoderへオフラインで渡され予測に使われる
    """
    
    # encoder
    encoder = Model(enc_input, [st_h, st_c])

    # decoder
    # 使いまわし以外の先頭にpred_を付加している
    pred_dec_input = keras.layers.Input(shape=(1, N_IN))
    # 伝搬する状態情報　それぞれ　[隠れ層, メモリセル]   
    pred_dec_st_in = [keras.layers.Input(shape=(N_MID,)), keras.layers.Input(shape=(N_MID,))]
    # LSTM層 dec_lstmは使いまわし
    pred_dec_output, pred_dec_st_h, pred_dec_st_c = dec_lstm(pred_dec_input, initial_state=pred_dec_st_in)
    pred_dec_states = [pred_dec_st_h, pred_dec_st_c]
    # 全結合層 dec_denseは使いまわし
    pred_dec_output = dec_dense(pred_dec_output)
    # モデル
    decoder = Model([pred_dec_input] + pred_dec_st_in, [pred_dec_output] + pred_dec_states)
    
    # 使いまわせるよう保存する
    encoder.save(work_dir+encoder_file)
    decoder.save(work_dir+decoder_file)
    
    return



def test_Seq2Seq(X_enc, X_dec, Y_dec, tokenizer:Tokenizer):
    """
    学習したSeq2Seqモデルで予測テスト
    
    入力を入力らしく扱うのはX_encのみ
    """
    
    # encoder
    encoder = load_model(work_dir + encoder_file)
    encoder.summary()
    # decoder
    decoder = load_model(work_dir + decoder_file)
    decoder.summary()
    

    def predict_sequence(x_enc_sequence, encoder, decoder):
        """
        sequenceからsequenceをpredictする

        入力シーケンスをいじる（one-hot化する、逆順にする等）処理は
        この中で行うと切り分けとして綺麗

        """
        # 素性の次元数
        word_vec_dim = len(tokenizer.word_index)+1
        # 入力のシーケンスを逆順にする
        x_enc_sequence = x_enc_sequence[::-1]
        # 入力に併せてone-hot化＆リシェイプ
        x_enc_sequence = keras.utils.to_categorical(x_enc_sequence, num_classes=word_vec_dim)
        x_enc_sequence = np.reshape(x_enc_sequence, (1, x_enc_sequence.shape[0], x_enc_sequence.shape[1]))

        # decoder側入力（予測値から再帰的に構築）
        x_dec_pred = np.zeros(shape=(1,1,1))
        x_dec_pred[0][0][0] = tokenizer.word_index['_'] # 開始文字

        # 予測値
        predicted = []
        
        # 入力シーケンスx_enc_sequenceをまず初期状態として入れる以降は再帰処理
        state_value = encoder.predict(x_enc_sequence)
        
        # 時系列的な再帰処理の部分
        for i in range(N_RNN_dec):
            # decoder入力の構造にリシェイプ
            x_dec_pred = keras.utils.to_categorical(x_dec_pred, num_classes=word_vec_dim)
            x_dec_pred = np.reshape(x_dec_pred, (1, 1, word_vec_dim))

            # 予測!!
            y, h, c = decoder.predict([x_dec_pred] + state_value)

            y_seq_pred = y[0][0] # 例の3次元テンソル（サンプル数, 系列長, 素性）の素性を抽出
            y_pred = np.argmax(y_seq_pred) # 素性からトークンを予測

            # 終端記号なら終了
            if y_pred == tokenizer.word_index['$']:
                break

            predicted.append(y_pred)
            
            x_dec_pred = np.zeros(shape=(1,1,1)) # 毎ターンリセットする
            x_dec_pred[0][0][0] = y_pred # 今のセルの出力を次のセルの入力に回す
            state_value = [h, c] # 状態も伝搬
        
        return predicted
    
    def token2sentence(tokenizer, sequence):
        """
        トークン列（sequence）を文字列に戻す処理
        """
        sentence = ""
        for s in sequence:
            word = [k for k, v in tokenizer.word_index.items() if v == s]
            if len(word) > 0:
                sentence = sentence + word[0]
        return sentence
    
    
    count = 0
    for i, x_seq in enumerate(X_enc):
        # 入力の式
        x_sentence = token2sentence(tokenizer, x_seq)
        print(f'Input:{x_sentence}')
        
        # トークン列からトークン列をpredictする
        y_seq = predict_sequence(x_seq, encoder, decoder)
        
        # トークン列を単語に戻す
        y_sentence = token2sentence(tokenizer, y_seq)
        
        print(f'Output: {y_sentence}')
        
        # 正解
        y_dec = Y_dec[i]
        answer = token2sentence(tokenizer, y_dec)
        print(f'Answer: {answer[:-1]}\n')        
        
        if count > 30:
            break
        else:
            count += 1

        
    
    

if __name__ == '__main__':
    
    # データセット、トークナイザは別途作成しておく
    
    # 最大サンプル数
    max_sample_num = 100000
    
    #make_dataset(max_sample_num)
    
    # Train, Test, Talkenizer ロード
    train, test, tokenizer = load_dataset()
    
    # training Seq2seq
    #train_Seq2Seq(train['X_enc'], train['X_dec'], train['Y_dec'], tokenizer, test['X_enc'], test['X_dec'], test['Y_dec'])
    
    # test Seq2Seq
    test_Seq2Seq(test['X_enc'], test['X_dec'], test['Y_dec'], tokenizer)

