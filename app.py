import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# フォントの設定
sns.set(font_scale=1.5)
font_dict = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 12}

# 色の設定
colors = sns.color_palette("pastel")

# モデルとクラスの読み込み
model = load_model('keras_model.h5')
class_names = ["ペットボトル", "ビニール袋", "段ボール", "カイロ", "紙パック"]

# ゴミの捨て方とリサイクル方法の関数
def get_disposal_method(class_name):
    disposal_methods = {"ペットボトル": "リサイクルしてください。",
                        "ビニール袋": "資源ごみとして分別してください。",
                        "段ボール": "資源ごみとして分別してください。",
                        "カイロ": "可燃ごみとして捨ててください。",
                        "紙パック": "資源ごみとして分別してください。"}
    return disposal_methods.get(class_name, "特定できるゴミの捨て方がありません。")

def get_recycle_method(class_name):
    recycle_method = {"ペットボトル": "ペットボトルのリサイクルは...",
                      "ビニール袋": "ビニール袋のリサイクルは...",
                      "段ボール": "段ボールのリサイクルは...",
                      "カイロ": "カイロのリサイクルは...",
                      "紙パック": "紙パックのリサイクルは..."}
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

        # グラフの表示
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 棒グラフ
        st.subheader('棒グラフと折れ線グラフ')
        bar_labels = class_names
        bar_probs = prediction[0]
        sns.barplot(x=bar_probs, y=bar_labels, palette=colors, ax=ax, label="棒グラフ")
        
        # 折れ線グラフ
        line_labels = class_names
        line_probs = prediction[0]
        ax2 = ax.twiny()
        sns.lineplot(x=line_probs, y=line_labels, marker="o", ax=ax2, sort=False, color=colors[1], label="折れ線グラフ")
        
        # 軸ラベルの設定
        ax.set_xlabel('確率', fontdict=font_dict)
        ax2.set_xlabel('確率', fontdict=font_dict)
        
        # 凡例の表示
        ax.legend(loc='upper right')
        ax2.legend(loc='upper left')
        
        st.pyplot()

        # 一覧表の表示
        st.subheader('一覧表')
        df_prob = pd.DataFrame(bar_probs, bar_labels, columns=['確率'])
        df_prob['確率'] = df_prob['確率'].apply(lambda x: f"{x*100:.3f}%")
        st.write(df_prob)

        # 説明文の表示
        st.subheader('ゴミの捨て方の説明')
        detected_classes = [class_names[idx] for idx, prob in enumerate(prediction[0]) if prob >= 0.6]
        for class_name in detected_classes:
            st.subheader(f"{class_name}の説明:")
            st.write(f"{class_name}が60%以上の確率で検出されました。")
            st.write(f"ゴミの捨て方の説明: {get_disposal_method(class_name)}")
            st.write("")

        # ゴミのリサイクル過程
        st.subheader('リサイクル過程')
        for class_name in detected_classes:
            st.subheader(f"{class_name}のリサイクル過程:")
            st.write(f"ゴミのリサイクル方法の説明: {get_recycle_method(class_name)}
