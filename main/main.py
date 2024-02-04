import time
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle
import os
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
st.set_page_config(page_title="Cryptonite")
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://64.media.tumblr.com/b29afc099269a74c0a2c0169545327bd/tumblr_ndv49dBypR1qjmwryo1_400.gifv");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)


from langchain_openai import OpenAI


def lung_cancer_survey():
    st.title("Lung Cancer Survey Form")

    parameters = [
        "Air Pollution",
        "Alcohol use",
        "Dust Allergy",
        "Occupational Hazards",
        "Genetic Risk",
        "Chronic Lung Disease",
        "Balanced Diet",
        "Obesity",
        "Smoking",
        "Passive Smoker",
        "Chest Pain",
        "Coughing of Blood",
        "Fatigue",
        "Weight Loss",
        "Shortness of Breath",
        "Wheezing",
        "Swallowing Difficulty",
        "Clubbing of Finger Nails",
        "Frequent Cold",
        "Dry Cough",
        "Snoring",
    ]

    age = st.number_input("Enter your age:", min_value=1, step=1)

    input_values = {}
    for param in parameters:
        input_values[param] = st.slider(
            f"Rate your {param} on a scale of 1 to 10",
            min_value=1,
            max_value=10,
            value=5,
        )

    submit_button = st.button("Submit")

    if submit_button:
        data = {"Age": age, **input_values}

        df = pd.DataFrame([data])
        model = load_model(r"/home/gurmann/Downloads/model_cancer1.h5")
        pred=model.predict(df)

        pred=pred.flatten().tolist()
        print(pred)

        if(pred[0]>pred[1] and pred[0] > pred[2]):
            st.write(f"High risk of Lung Cancer")
        if (pred[1] > pred[0] and pred[1] > pred[2]):
            st.write(f"Low risk of Lung Cancer")
        if (pred[2] > pred[1] and pred[2] > pred[0]):
            st.write(f"Medium risk of Lung Cancer")

def feedback():
    st.header("Feedback Form")
    feedback = st.text_area("Please provide your feedback:")

    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
    else:
        st.warning("Please enter your feedback before submitting.")


# Gradcam functions
def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def softmax(x):
    f = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f


def ScoreCam(model, img_array, layer_name, input_shape, max_N=-1):
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0, :, :, k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:, :, :, max_N_indices]

    act_map_resized_list = [cv2.resize(act_map_array[0, :, :, k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k
                            in range(act_map_array.shape[3])]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0, :, :, k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    weights = pred_from_masked_input_array[:, cls]
    cam = np.dot(act_map_array[0, :, :, :], weights)
    cam = np.maximum(0, cam)
    cam /= np.max(cam)
    return cam


def superimpose(uploaded_image, cam, emphasize=False):
    image = Image.open(uploaded_image)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


def read_and_preprocess_img(path, size=(224, 224)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Generate a heatmap from the given unprocessed image
def heatmapgen(image):
    img_array = np.array(image)
    image_df = pd.DataFrame(img_array, columns=range(img_array.shape[1]))
    fig, ax = plt.subplots()
    sns.heatmap(image_df, cmap='YlGnBu', annot=False)
    return fig


def gengradcambox(score_cam_superimposed):
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 100, 100], dtype=np.uint8)
    # Create a mask that identifies the red area
    mask = np.all((score_cam_superimposed >= lower_red) & (score_cam_superimposed <= upper_red), axis=-1)
    # Find the largest connected component
    label_image = label(mask, connectivity=2)
    regions = regionprops(label_image)
    # Find the largest component
    largest_component = max(regions, key=lambda region: region.area)
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    ax.imshow(score_cam_superimposed, cmap='jet')
    # Calculate the bounding box for the largest component
    minr, minc, maxr, maxc = largest_component.bbox
    # Draw a bounding box around the largest component
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, color='black', linewidth=1)
    ax.add_patch(rect)
    # Add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cmap = plt.get_cmap('jet')
    norm = colors.Normalize(0, 100)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Severity (%)", orientation='vertical')
    # Display the plot
    ax.axis('off')
    plt.title("Using Gradcam")
    return plt
os.environ['OPENAI_API_KEY'] = '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'

with st.sidebar:
    st.image("https://i.postimg.cc/GmTwzJ3n/productathon-logo-removebg-preview.png")
    st.title('Medical Assistant')
    selected_model = st.sidebar.selectbox('Choose a model', ['Pneumonia', 'Lung Cancer', 'MedAssistant', "Feedback"],key='selected_model')
if (selected_model=="Pneumonia"):
    st.header("Welcome to the X-RAY Analysis Page!")
    st.subheader("This section of the site empowers you to upload X-ray images for comprehensive analysis. Simply upload your images, and let our system generate meaningful analytics to support your diagnostic journey.")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="pneu")
    if uploaded_image is not None:
        st.toast("Image uploaded successfully", icon='👩‍⚕️')
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Load model with GPU

        model = load_model(r"/media/gurmann/GURMANN/Hackathon/PNEU_MULTICLASS/model.h5")
        # Prediction
        if st.button("Predict"):
            # Image Pre-processing
            p_img = image.convert("RGB")
            p_img = p_img.resize((150, 150))
            p_img = np.array(p_img) / 255.0
            p_img = np.expand_dims(p_img, axis=0)

            # Prediction loading
            with st.spinner("Predicting..."):
                y_prob = model.predict(p_img)
                y_pred = y_prob.argmax(axis=-1)
                # time.sleep(2)

            st.toast("Prediction complete", icon='👩‍⚕️')
            total = y_prob[0][1] + y_prob[0][2]
            st.write(f"<h4>You have a <b>{total * 100:.2f}% </b> chance of having Pneumonia</h4>",
                     unsafe_allow_html=True)
            st.write(
                f"<ul><li><b>Bacterial Pneumonia:</b> {y_prob[0][1] * 100:.2f}% chance</li><li><b>Viral Pneumonia:</b> {y_prob[0][2] * 100:.2f}% chance</li></ul>",
                unsafe_allow_html=True)

            # Heatmap
            st.subheader("Heatmap")
            fig = heatmapgen(image)
            st.pyplot(fig)

            # Gradcam
            st.subheader("Severity Map")
            layer_name = 'conv2d_1'
            input_shape = (150, 150, 3)
            with st.spinner("Generating gradcam..."):
                score_cam = ScoreCam(model, p_img, layer_name, input_shape)
            st.toast("Severity Map Generated", icon='👩‍⚕️')
            score_cam_superimposed = superimpose(uploaded_image, score_cam)
            plt = gengradcambox(score_cam_superimposed)
            st.pyplot(plt)
            st.markdown(f"The box is the region with highest amount of pathogens", unsafe_allow_html=True)

            # Calculate percentage
            size = score_cam_superimposed.size
            RED_MIN = np.array([0, 0, 128], np.uint8)
            RED_MAX = np.array([250, 250, 255], np.uint8)
            dstr = cv2.inRange(score_cam_superimposed, RED_MIN, RED_MAX)
            no_red = cv2.countNonZero(dstr)
            frac_red = np.divide(float(no_red), int(size))
            percent_red = np.multiply(float(frac_red), 100)
            BLU_MIN = np.array([128, 0, 0], np.uint8)
            BLU_MAX = np.array([255, 250, 250], np.uint8)
            dstr2 = cv2.inRange(score_cam_superimposed, BLU_MIN, BLU_MAX)
            no_blu = cv2.countNonZero(dstr2)
            frac_blu = np.divide(float(no_blu), int(size))
            percent_blu = np.multiply(float(frac_blu), 100)
            percent_yel = 100 - percent_red - percent_blu
            st.write(
                f"<ul><li>High Severity: {percent_red:.2f}%</li><li>Medium Severity: {percent_yel:.2f}%</li><li>Low Severity: {percent_blu:.2f}%</li></ul>",
                unsafe_allow_html=True)

            st.toast("Please fill the feedback form", icon='👩‍⚕️')

        del model
if (selected_model=="MedAssistant"):
    st.title("MedAssistant")
    st.write("Personalised chatbot to help with your queries.")
    llm = OpenAI()
    data = st.text_input("Enter question to be asked: ")
    submitted = st.button("Enter")
    if submitted:
        response = llm.invoke(data)
        st.write(response)
if (selected_model=="Feedback"):
    feedback()
if(selected_model=="Lung Cancer"):
    lung_cancer_survey()