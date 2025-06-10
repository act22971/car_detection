import streamlit as st

def home_page():
    st.title("Welcome to Car Crash Detection and Tracking App")
    st.write("This is a Streamlit app using to represent car crash detection and tracking model.")
    st.write("Here is the example result of the model")
    
    # Replace the image with a video player
    video_url = "https://cdn.discordapp.com/attachments/1180164108809666682/1310646968267182193/model_result_best_2.mp4?ex=6745fa9b&is=6744a91b&hm=4f85d8629ffff576e56d4eb7a9432b477965bc20d0193fb33aaae0765efdb28e&"  # You can replace this with your own video URL
    st.video(video_url)

    st.markdown(
        """
        ### Features:
        - Detecting car crash.
        - Count How many cars in the video.
        """
    )
