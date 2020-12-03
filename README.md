# 🌈A Server doing Doodle Classification

This repo is a server I create to support my thesis project, Magical Pencil, in NYU ITP.

My game will send the picture that the user draws to this server, and acquire a prediction.

<image src="docs/assets/MagicalPencil-Cover.gif" />

It runs a Keras model trained with Google Quick Draw dataset and ImageNet.

The accuracy of the prediction is 0.89.

## Run the server with docker

Make sure you have installed docker. [ref](https://docs.docker.com/get-docker/)

```bash
docker-compose up
```

## Try to get a prediction

*After starting the server*, visiting the sketch, click run and draw something:

<https://editor.p5js.org/yangyang/sketches/DgHa-bMDT>

## Develop the server on your machine

### Install

Before install, it's suggested to create a venv with your favorite tool

```bash
make install
```

### Start the server

```bash
make start
```

## APIs

### <span style="background-color:#49cc90; color:white">POST</span> `/api/predict`

Post the Base64 encoded image to this API within the form[`data`]. The API will return the prediction in a JSON array.

### Other

The other APIs are designed for my game, so I'll skip documenting them.
