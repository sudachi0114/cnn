
# cnn project について

## ディレクトリ構造

    ```

    .
    ├── architecture
    ├── binary_classifer
    ├── checker
    ├── data_argumentation
    ├── director
    ├── dogs_vs_cats_mid300
    ├── dogs_vs_cats_smaller
    ├── opcheck
    ├── preserve
    ├── test1
    ├── train
    ├── vis_cnn_activation
    └── vis_cnn_filter

    ```


## architecture

* プログラムを書く際の設計に関して色々試すためのディレクトリ。
    - 現状, class を用いて書くか、複数のファイルに分割して書くか迷っている。

## binary_classifer
* 2クラス分類に関して色々な手法を試してみた。
    - そのまま学習する (binary_classify)
    - 学習画像の水増しをする (pad_)
    - 学習済みモデルを用いる (VGG16 model)
        + 転移学習
            * 特徴抽出 (feature extraction)
            * 詳細調整 (fine tuning)

## checker
* data に関するあれこれをチェックするものを突っ込んであるディレクトリ。
* 主に何か (画像などを) 描画するプログラムが多い。
    - `logFormatter` の見本がここに置いてあるので、各プロジェクトの log と同じ位置に置いて実行すると対話的に log を整形できる。

## data_argumentation
* Data Augmentarion (データの水増し) に関しての実験用ディレクトリ
* ごちゃっとしているので、近々整理する予定。
    - 期限は未定..

## director
* 訓練用のデータセットを作るプログラム。
    - 現在は以下のサイズに関して用意がある。
        + small ( `dogs_vs_cats_smaller/` )
            * train 100枚 (各class 50枚 * 2class)
            * validation 50枚 (各class 25枚 * 2class)
            * test 50枚 (各class 25枚 * 2class)
        + middle ( `dogs_vs_cats_mid300/` )
            * train 300枚 (各class 150枚 * 2class)
            * validation 100枚 (各class 50枚 * 2class)
            * test 100枚 (各class 50枚 * 2class)
        + full ( `dogs_vs_cats_full/` )
            * train 20000枚 (各class 10000枚 * 2class)
            * validation 2500枚 (各class 1250枚 * 2class)
            * test 2500枚 (各class 1250枚 * 2class)


## opcheck
* 動作確認用のプログラムが置いてある。
    - 想定使用シーンは新しいサーバに環境を置いた時にきちんと動くか確かめる用など。

## preserve
* 画像をなんども読み込むので読み込んだ結果を np配列として保存しておけばいいのでは、と思い立ち作った。
    - 保存方法は 2つ
        + npz (numpy 固有のデータの保存方法)
        + pickler (python 固有のデータの保存方法)
    - 読み込み元の画像ディレクトリに上記のどちらかのファイル形式で保存することにしている。

## vis_cnn_activation
* CNN (VGG16 の imagenet のものだった気がする) のアクティベーションに関して描画するプログラム。
    - ある層の出力特徴マップを描画する。

## vis_cnn_filter
* CNN (VGG16 の imagenet のものだった気がする) のフィルタに関して描画するプログラム。
    - **注意!!** フィルタそのものではなく、フィルタの一番応答するパターンを入力画像に刷り込んでそれを描画する。


## train / test data の位置について
* cnn 直下に配置することを想定
    - split しなおしたものも同様。


## datasets
* しばらくは Kaggle の Dogs vs Cats で遊ぶ予定。
    - train : dog / cat 12500 枚 => 計 25000枚
        + train.zip
        + 紛らわしいので、dogs_vs_cats_origin と名前を変更して使う。
    - test1 : dog / cat 12500 枚 (random?)
        + 使わないので削除。
    - 数枚ピックアップして使う。

* binary classifer が完成してきたら 自分の取ってきたデータでやる。
    - crawler というディレクトリを切って scraping program を配置予定
        + program は部分的に記載済み。

