import streamlit as st
import requests
st.title('Vehicle Damage Detection')

uploaded_file = st.file_uploader('Upload the File',type = ['jpg','png','jpeg'])

if uploaded_file:
    st.image(uploaded_file,caption='Uploaded File',use_container_width =True)

if st.button('Get Prediction'):
    file = {'file':(uploaded_file.name,uploaded_file.getbuffer(),uploaded_file.type)}

    response = requests.post(
    "http://127.0.0.1:8000/predict",
    files=file,
    timeout=30
)


    if response.status_code == 200:
        prediction = response.json()
        message = prediction['prediction']
        idx = message.index('_')
        message = message[idx+1:]
        if message == 'Crushed' or message == 'Normal':
            st.success(f'Your Vehicle Is {message}')
        else :
            st.success(f'Your Vehicle Has A {message}')
    else :
        st.error(f'Response is : {response.text}')
