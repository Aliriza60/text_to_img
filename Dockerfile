FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /text_to_image
COPY text_to_image_1.py /text_to_image/
COPY requirements.txt /text_to_image/
RUN pip install scipy
RUN pip install -r requirements.txt
CMD ["python","text_to_image.py"]
