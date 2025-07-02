import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Knowledge Distillation - Model Comparison - Weather Phenomena Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

SEED = 42
torch.manual_seed(SEED)

def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0
    return normalized_img.unsqueeze(0)

classes = {0: 'dew',
           1: 'fogsmog',
           2: 'frost',
           3: 'glaze',
           4: 'hail',
           5: 'lightning',
           6: 'rain',
           7: 'rainbow',
           8: 'rime',
           9: 'sandstorm',
           10: 'snow'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.create_layer(
            residual_block, 64, 64, n_blocks_lst[0], 1)
        self.conv3 = self.create_layer(
            residual_block, 64, 128, n_blocks_lst[1], 2)
        self.conv4 = self.create_layer(
            residual_block, 128, 256, n_blocks_lst[2], 2)
        self.conv5 = self.create_layer(
            residual_block, 256, 512, n_blocks_lst[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride=1)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)

        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

n_classes = len(classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_1 = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[2, 2, 2, 2],
    n_classes=n_classes
).to(device)
model_1.load_state_dict(torch.load(
    "./model/kdpretraindata_wt.pt", map_location=device))
model_1.eval()

model_2 = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[3, 4, 6, 3],
    n_classes=n_classes
).to(device)
model_2.load_state_dict(torch.load(
    "./model/teacher_wt.pt", map_location=device))
model_2.eval()


st.title("Weather Phenomena Prediction - Model Comparison")

image_sets = {
    "Set 5 images": "./static/set5",
    "Set 10 images": "./static/set10",
    "Set 15 images": "./static/set15" 
}
st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)

st.subheader("Select Predefined Image Set or Upload Your Own Images")
use_predefined_set = st.radio("Choose an option:", ["Predefined Set", "Upload Images"])

if use_predefined_set == "Predefined Set":
    selected_set = st.selectbox("Choose a predefined set:", list(image_sets.keys()))
    image_folder = image_sets[selected_set]
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img.verify()  # kiểm tra ảnh có hợp lệ không
            img = Image.open(img_path)  # cần mở lại sau khi verify
            images.append(img)
        except Exception as e:
            st.warning(f"⚠️ Không thể mở ảnh: {img_path} ({str(e)})")
else:
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    images = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                img = Image.open(file)
                img.verify()
                file.seek(0)  # reset con trỏ sau verify()
                img = Image.open(file)
                images.append(img)
            except Exception as e:
                st.warning(f"⚠️ Ảnh tải lên bị lỗi: {file.name} ({str(e)})")


st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
if images:
    st.write(f"**Number of images:** {len(images)}")
    
    st.subheader("Image Set Preview:")
    cols = st.columns(5)
    for idx, img in enumerate(images):
        with cols[idx % 5]:
            if img is None:
                st.warning(f"⚠️ Không thể đọc ảnh số {idx + 1}")
            elif not isinstance(img, (np.ndarray, str, Image.Image, bytes)):
                st.error(f"❌ Ảnh số {idx + 1} không đúng định dạng: {type(img)}")
            else:
                st.image(img, use_container_width=True, caption=f"Image {idx + 1}")

    results = []
    total_time_1, total_time_2 = 0, 0

    with st.spinner("Running Predictions..."):
        for idx, img in enumerate(images):
            input_tensor = transform(img).to(device)

            
            start_time_1 = time.time()
            with torch.no_grad():
                output_1 = model_1(input_tensor)
                _, predicted_class_1 = torch.max(output_1, 1)
            end_time_1 = time.time()
            predicted_label_1 = classes[predicted_class_1.item()]
            time_1 = end_time_1 - start_time_1
            total_time_1 += time_1

            
            start_time_2 = time.time()
            with torch.no_grad():
                output_2 = model_2(input_tensor)
                _, predicted_class_2 = torch.max(output_2, 1)
            end_time_2 = time.time()
            predicted_label_2 = classes[predicted_class_2.item()]
            time_2 = end_time_2 - start_time_2
            total_time_2 += time_2

            
            results.append({
                "Image": f"Image {idx + 1}",
                "Model": "KD Student",
                "Prediction": predicted_label_1,
                "Time Taken (s)": time_1
            })
            results.append({
                "Image": f"Image {idx + 1}",
                "Model": "Teacher",
                "Prediction": predicted_label_2,
                "Time Taken (s)": time_2
            })

    # Create a DataFrame for results
    results_df = pd.DataFrame(results)

    st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
    # Display results in a table
    st.subheader("Prediction Results:")
    results_kd = results_df[results_df["Model"] == "KD Student"]
    results_teacher = results_df[results_df["Model"] == "Teacher"]

    # Chia cột
    col1, col2 = st.columns([0.5, 0.5])  # Tỷ lệ cột 50-50, tùy chỉnh nếu cần

    # Hiển thị bảng trong từng cột
    with col1:
        st.write("**KD Student Results**")
        st.dataframe(results_kd)

    with col2:
        st.write("**Teacher Results**")
        st.dataframe(results_teacher)

    st.markdown("<hr style='border: 1px solid #ccc; margin: 20px 0;'>", unsafe_allow_html=True)
    # Charts
    st.subheader("Comparison Charts:")
    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        # Total time chart
        total_times_df = pd.DataFrame({
            "Model": ["KD Student", "Teacher"],
            "Total Time (s)": [total_time_1, total_time_2]
        })
        total_fig = px.bar(
            total_times_df,
            x="Model",
            y="Total Time (s)",
            text="Total Time (s)",
            title="Total Time Comparison",
            labels={"Total Time (s)": "Time (seconds)"}
        )
        total_fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        st.plotly_chart(total_fig, use_container_width=True)

    with col2:
        # Per-image time chart
        fig = px.bar(
            results_df,
            x="Image",
            y="Time Taken (s)",
            color="Model",
            barmode="group",
            title="Time Comparison for Each Image",
            labels={"Time Taken (s)": "Time (seconds)"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
        Made by <a>NamNguyen27</a>
    </div>
    """,
    unsafe_allow_html=True
)