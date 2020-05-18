# 🌈A Server doing Doodle Classification

This repo is a server I create to support my thesis project, Magical Pencil, in NYU ITP.

My game will send the picture that the user draws to this server, and acquire a prediction.

<image src="docs/assets/MagicalPencil-Cover.gif" />

It runs a Keras model trained with Google Quick Draw dataset and ImageNet.

The accuracy of the prediction is 0.89.

run `doodlePrediction_Main.py` to create this server;

APIs:

```
/api/doodlePredict
```

Post the Base64 encoded image to this API within the form[`data`]. The API will return the prediction in a JSON array.

```
/api/askForSprite
```

Post the Base64 encoded image to this API within the form[`data`]. The API will return a link to the processed png image with the white margin around the contour removed.

I designed other APIs only intended for my game. I'll skip introducing them.
