FROM ubuntu:18.04

RUN apt upgrade && apt update && apt install -y \
    software-properties-common \
    unzip \
    git \
    python3-pip \
    python3-dev  \
    python3-venv \
    wget

RUN pip3 install --upgrade pip
ADD . compare-text-project/
RUN cd compare-text-project && pip3 install  --no-cache -r requirements.txt

ENTRYPOINT [ "sh", "-c" ]

CMD ["python3 compare-text-project/main.py"]
