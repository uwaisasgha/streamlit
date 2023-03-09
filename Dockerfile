From python:3.8

COPY . .
RUN pip install 

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev


# Set the environment variable for Streamlit
# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

EXPOSE 8501

ENTRYPOINT ["streamlit", "run","./newcoe.py"]
