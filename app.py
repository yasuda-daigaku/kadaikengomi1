import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# モデルとクラスの読み込み
model = load_model('keras_model.h5')
class_names = ["ペットボトル", "ビニール袋", "段ボール", "カイロ", "紙パック"]

# ゴミの捨て方とリサイクル方法の関数

def get_disposal_method(file_path, start_delimiter, end_delimiter):
    with open(file_path, 'r', encoding='utf-8') as file:
        inside_target = False
        extracted_text = ""

        for line in file:
            if start_delimiter in line:
                inside_target = True
                # デリミタより前の部分を抽出
                extracted_text += line.split(start_delimiter, 1)[1].strip()
            elif end_delimiter in line:
                inside_target = False
                # デリミタより前の部分を抽出
                extracted_text += line.split(end_delimiter, 1)[0].strip()
                break  # end_delimiterが見つかったら終了
            elif inside_target:
                # デリミタがない場合はその行全体を抽出
                extracted_text += line.strip()

    return extracted_text

def get_recycle_method(class_name):
    recycle_method = {
        "ペットボトル": "・リサイクル工場で洗浄後細かく砕かれフレークという原料になります。このフレークからペットボトルや様々なものに加工れます、例えば食品トレーや卵パック、衣料品などになります。",
        "ビニール袋": "プラスチック製品として分類されほかの製品と一緒に溶かされ再生プラスチックとして活用されます。",
        "段ボール": "回収された段ボールは水につけ砂やプラスチックなどの異質を取り除き乾燥させ最後にプレス機にかけたら新しい段ボールとして再利用されます。",
        "カイロ": "カイロは、不燃ごみとして捨てられて、リサイクルされません。",
        "紙パック": "回収された紙パックは水を加えながら溶かしゴミなどを取り除きます。"
    }
    return recycle_method.get(class_name, "特定できるゴミのリサイクル方法がありません。")

# Streamlitアプリの設定
st.set_option("deprecation.showfileUploaderEncoding", False)
st.sidebar.title("ゴミ分別アプリ")
st.sidebar.write("Teachable Machineの学習モデルを使って画像判定を行います。")

# 画像の選択方法
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))

# 画像のアップロードまたはカメラでの撮影
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

# 画像が選択された場合の処理
if img_file is not None:
    with st.spinner("推定中..."):
        # 画像の前処理
        image = Image.open(img_file)
        size = (224, 224)
        image = ImageOps.fit(image, size, method=0, bleed=0.0, centering=(0.5, 0.5))
        st.image(image, caption="対象の画像", width=480)
        st.write("")
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        try:
            data[0] = normalized_image_array
        except (ValueError, IndexError) as e:
            st.write(f"画像データの設定エラー: {e}")
        else:
            prediction = model.predict(data)

        # プロットの初期化
        fig, ax = plt.subplots(figsize=(10, 6))

        # 棒グラフのプロット
        bar_labels = ["PET Bottle", "Plastic Bag", "Cardboard", "Hand Warmer", "Paper Pack"]
        bar_probs = prediction[0]

        ax.bar(bar_labels, bar_probs, label='Probability')

        # 軸ラベルとタイトルの設定
        ax.set_xlabel('Class', fontsize=16)
        ax.set_ylabel('Probability', fontsize=16)

        # タイトルもフォントサイズを指定して設定
        ax.set_title("Prediction Probability", fontsize=20)
        
        # グリッドを表示
        ax.grid(True)

        st.pyplot(fig)

        # 一覧表の表示
        st.subheader('一覧表')
        df_prob = pd.DataFrame(prediction, columns=class_names)
        df_prob = df_prob.T.reset_index()
        df_prob.columns = ['クラス', '確率']
        df_prob['確率'] = df_prob['確率'].apply(lambda x: f"{x*100:.3f}%")
        st.table(df_prob)

        detected_classes = [class_names[idx] for idx, prob in enumerate(prediction[0]) if prob >= 0.6]

        start_delimiter = 'start_' + 'detected_classes'
        end_delimiter = 'end_' + 'detected_classes'

        # 説明文の表示
        st.subheader('ゴミの捨て方の説明')
        for class_name in detected_classes:
            st.subheader(f"{class_name}の説明:")
            st.write(f"{class_name}が60%以上の確率で検出されました。")
            st.write(f"ゴミの捨て方の説明: {get_disposal_method('disposal_methods.txt', start_delimiter, end_delimiter)}")
            st.write("")

        # ゴミのリサイクル過程
        st.subheader('リサイクル過程')
        for class_name in detected_classes:
            st.subheader(f"{class_name}のリサイクル過程:")
            st.write(f"ゴミのリサイクル方法の説明: {get_recycle_method(class_name)}")
