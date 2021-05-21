import numpy as np
import images as images
import torch
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision import transforms
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from random import randint
import SessionState
import pickle

# provide excel download
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Results', float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="result.xlsx">Download result as Excel file</a>'

def main():
    st.title('Predict concentration')

    session_state = SessionState.get(state=0)

    if st.button('Clear uploaded files', help="Clears list of uploaded files"):
        session_state.state = str(randint(1000, 100000000))

    files = st.file_uploader(label="Upload files for prediction", accept_multiple_files=True, type=["png"], help="Select file(s) to perform prediction on", key=session_state.state)



    #if selection made
    if len(files) == 0:
        st.write("Upload a file to get started")
        return

    #prepare for possible multiple files
    data_files = []
    for file in files:
        data_files.append(file.name)

    print(len(data_files))

    x_all = np.ndarray(shape=(len(data_files), images.target_height, images.target_width, images.channels), dtype=np.uint8)
    x_all_xgboost = np.ndarray(shape=(len(data_files), images.target_height* images.target_width* images.channels), dtype=np.uint8)

    i = 0
    no_file = len(data_files)
    my_bar = st.progress(0.0)
    file_tracker = st.empty()
    for file in files:
        #print(i,'/', no_file, ' File:',str(file))
        file_tracker.text('Uploading (' + str(i+1)+'/'+ str(no_file)+'): '+ str(file.name))
        my_bar.progress((i+1)/no_file)
        #extract data scaled down to 224x224
        cur_image = images.preprocess_image(file)
        x_all[i] = np.array(cur_image)
        x_all_xgboost[i] = np.array(np.ravel(cur_image))
        #extract required output
        i+=1

    file_tracker.text('Number of file uploaded: ' + str(no_file) )
    my_bar.empty()
    #rotate x_all to be in CHW format (channel first)
    #print(x_all.shape)
    #x_all = np.moveaxis(x_all, 3, 1)
    #print(x_all.shape)

    #make pytorch tensors
    x_t = torch.tensor(np.moveaxis(x_all, 3, 1), dtype=torch.float32)
    normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    x_t = normalizer(x_t)

    #RESNET18
    resnet = resnet18(pretrained=False, num_classes=5)
    resnet.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

    resnet.eval()
    y_hat = torch.clamp(resnet(x_t), 0, 100)

    y_pred_np = y_hat.cpu().detach().numpy()

    pred_df = pd.DataFrame(y_pred_np, columns=images.value_names, index=data_files)
    st.header('Resnet18 results')
    st.dataframe(pred_df)

    st.markdown(get_table_download_link(pred_df), unsafe_allow_html=True)

    # RESNET34
    resnet = resnet34(pretrained=False, num_classes=5)
    resnet.load_state_dict(torch.load('model34.pt', map_location=torch.device('cpu')))

    resnet.eval()
    y_hat = torch.clamp(resnet(x_t), 0, 100)

    y_pred_np = y_hat.cpu().detach().numpy()

    pred_df = pd.DataFrame(y_pred_np, columns=images.value_names, index=data_files)
    st.header('Resnet34 results')
    st.dataframe(pred_df)

    st.markdown(get_table_download_link(pred_df), unsafe_allow_html=True)

    # RESNET50
    resnet = resnet50(pretrained=False, num_classes=5)
    resnet.load_state_dict(torch.load('model50.pt', map_location=torch.device('cpu')))

    resnet.eval()
    y_hat = torch.clamp(resnet(x_t), 0, 100)

    y_pred_np = y_hat.cpu().detach().numpy()

    pred_df = pd.DataFrame(y_pred_np, columns=images.value_names, index=data_files)
    st.header('Resnet50 results')
    st.dataframe(pred_df)

    st.markdown(get_table_download_link(pred_df), unsafe_allow_html=True)

    # XGBOOST
    loaded_model = pickle.load(open("xgboost.dat", "rb"))
    y_pred = np.clip(loaded_model.predict(x_all_xgboost), 0, 100)
    pred_xg_df = pd.DataFrame(y_pred, columns=images.value_names, index=data_files)
    st.header('XGBoost results')
    st.dataframe(pred_xg_df)
    st.markdown(get_table_download_link(pred_xg_df), unsafe_allow_html=True)

if __name__ == '__main__':
    main()