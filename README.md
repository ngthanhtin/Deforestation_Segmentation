# Indonesia Deforestation Segmentation
`Top 1 solution` for Indonesia Deforestation Segmentation Challenge. `(0.379 Dice Score on LB)` <br>
![1](./readme_imgs/output.png) ![2](./readme_imgs/output2.png)
## Additional Data link
In this work, I use a public dataset and add it into the provided dataset. You can download the data here: [ForestNet dataset](https://stanfordmlgroup.github.io/projects/forestnet/) <br>
The additional data contain the mask as `forest_loss_region.pkl` files, so I processed to `png` files in `eda2.py`. After running this python code, you will get the `png mask-image files` as well as the `csv file` containing the annontation of them.


# Pretrained Weights
You can find the pretrained model here: [link](https://drive.google.com/drive/folders/12HQBT4S2-dOSrLTwbLfwjVbIF15c8mBo?usp=sharing)

## Model Used in the Challenge
I used UNetPlusPlus with NFNet backbone.

## TODO
1. Heavy TTA. (zoom-in, rotation more)
2. Cutmix.
3. Postprocess (Morphology, Connected Components)


## Reference
[Challenge website](https://datameka.com/competition/632c46a4-9c05-4911-a8c2-08226d2fb4e4?tabIndex=7) <br>
[ForestNet](https://stanfordmlgroup.github.io/projects/forestnet/)

## Author
`Thanh-Tin Nguyen`