# Variables
IMAGE_NAME = sentiment-analysis-app
CONTAINER_NAME = sentiment-analysis-container

# Targets
.PHONY: build run stop clean

build:
    docker build -t $(IMAGE_NAME) .

run:
    docker run -d -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

stop:
    docker stop $(CONTAINER_NAME)
    docker rm $(CONTAINER_NAME)

clean: stop
    docker rmi $(IMAGE_NAME)
