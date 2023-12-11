import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

model = load_model('keras_model.h5')
class_names = ["ペットボトル", "ビニール袋", "段ボール", "カイロ", "紙パック"]

def get_disposal_method(class_name):
    # ゴミの捨て方の説明を返す関数
    disposal_methods = {
        "ペットボトル": "リサイクルしてください。",
        "ビニール袋": "資源ごみとして分別してください。",
        "段ボール": "資源ごみとして分別してください。",
        "カイロ": "可燃ごみとして捨ててください。",
        "紙パック": "資源ごみとして分別してください。",
    }
    return disposal_methods.get(class_name, "特定できるゴミの捨て方がありません。")

def get_recycle_method(class_name):
    recycle_method = {
        "ペットボトル": "ペットボトルのリサイクルは...",
        "ビニール袋": "ビニール袋のリサイクルは...",
        "段ボール": "段ボールのリサイクルは...",
        "カイロ": "カイロのリサイクルは...",
        "紙パック": "紙パックのリサイクルは...",
    }
    return recycle_method.get(class_name, "特定できるゴミのリサイクル方法がありません。")

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("ゴミ分別アプリ")
st.sidebar.write("Teachable Machineの学習モデルを使って画像判定を行います。")

st.sidebar.write("")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        image = Image.open(img_file)
        size = (224, 224)
        # 画像をセンタリングし指定したsizeに切り出す処理
        image = ImageOps.fit(image, size, method=0, bleed=0.0, centering=(0.5, 0.5))
        st.image(image, caption="対象の画像", width=480)
        st.write("")
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # 例外処理
        try:
            data[0] = normalized_image_array
        except Exception as e:
            st.write(e)
        else:
            prediction = model.predict(data)

        # 説明文の生成
        disposal_explanations = []
        explanations = []
        for idx, prob in enumerate(prediction[0]):
            if prob >= 0.6:
                disposal_explanation = (
                    f"{class_names[idx]}が{prob * 100:.2f}%の確率で検出されました。"
                    f"\nゴミの捨て方の説明: {get_disposal_method(class_names[idx])}"
                )
                recycle_explanation = (
                    f"\nゴミのリサイクル方法の説明: {get_recycle_method(class_names[idx])}"
                )

                disposal_explanations.append(disposal_explanation)
                recycle_explanations.append(recycle_explanation)
                
        if not disposal_explanations:
            disposal_explanations.append("60%以上の確率で検出されたクラスはありませんでした。")
    
        # 円グラフの表示
        pie_labels = class_names
        pie_probs = prediction[0]
        st.subheader('円グラフ')
        fig, ax = plt.subplots()
        wedgeprops = {"width": 0.3, "edgecolor": "white"}
        textprops = {"fontsize": 6}
        ax.pie(pie_probs, labels=None, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)
        
        # 一覧表の表示
        st.subheader('一覧表')
        st.write(pd.DataFrame(pie_probs, pie_labels))

        # 説明文の表示
        st.subheader('ゴミの捨て方の説明')
        for explanation in disposal_explanations:
            st.write(explanation)

        # ゴミのリサイクル過程
        st.subheader('リサイクル過程')
        for explanation in recycle_explanations:
            st.write(explanation)
        
        
        
