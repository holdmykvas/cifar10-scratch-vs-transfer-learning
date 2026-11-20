# Training from Scratch vs. Transfer Learning on CIFAR-10

When is it worth training your own CNN, and when should you use a pretrained network instead? This project compares three approaches to image classification on CIFAR-10 and measures how the answer changes as the amount of training data grows from 500 to 45,000 images.

||Approach|What is trained|
|-|-|-|
|**A**|Custom CNN from scratch|the whole network, starting from random weights|
|**B**|Feature extraction|MobileNetV2 (ImageNet weights) is frozen; only a new classification head is trained|
|**C**|Partial fine-tuning|starts from B, then the last 20 backbone layers are unfrozen and trained with a small learning rate|

## Key findings

!\[Test accuracy vs. training-set size](images/accuracy\_vs\_size.png)

* **With little data, pretraining wins by a wide margin.** At 500 training images, feature extraction reaches 76% test accuracy against 41% for the CNN trained from scratch, while training 22× fewer parameters.
* **The advantage disappears with enough data.** The gap shrinks from 35 percentage points (500 images) to 21, then 7, and at 45,000 images the custom CNN matches the pretrained models (85.9% vs. 85.7%).
* **Fine-tuning barely helped.** It added at most 0.6 points over feature extraction, at about 1.7× the training time. With small data it started overfitting after the first epoch.
* **The custom CNN is the fastest at inference:** about 1.6 ms per image vs. about 5.2 ms for MobileNetV2, and it is roughly 8× smaller in total.

## Results

Test accuracy on the official 10,000-image CIFAR-10 test set.

|Training images|Custom CNN (A)|Feature extraction (B)|Fine-tuning (C)|
|-:|-:|-:|-:|
|500|40.8%|76.0%|76.1%|
|2,000|59.3%|80.7%|80.6%|
|10,000|76.6%|83.5%|83.8%|
|45,000|**85.9%**|85.7%|**86.3%**|

Cost of each approach (training time is for the full 45,000-image run on a Colab T4 GPU):

||Trainable params|Total params|Training time (45k)|Latency per image|
|-|-:|-:|-:|-:|
|Custom CNN (A)|289,194|\~290k|452 s|\~1.6 ms|
|Feature extraction (B)|12,810|\~2.27M|501 s|\~5.2 ms|
|Fine-tuning (C)|1,065,290|\~2.27M|868 s (incl. B)|\~5.2 ms|

Macro-F1 is within 0.01 of accuracy in every run, as expected on a balanced test set. B and C run the same network at inference time, so they share one latency figure.

### Where the best model makes mistakes

!\[Confusion matrix, fine-tuning, 45,000 training images](images/confusion\_matrix.png)

The most confused pair is **cat ↔ dog** (208 of 1,373 errors). Vehicles (car, truck, ship) are recognised best. Some errors come from the background rather than the object: airplanes are mistaken for ships, both usually shown against blue sky or water.

## Recommendations

* **Use feature extraction** as the default when you have up to a few thousand images of everyday objects. It is accurate, cheap to train and stable between runs.
* **Use partial fine-tuning** only when you have plenty of data and need every last point of accuracy, or when your images differ strongly from ImageNet photos. Train the head first and use a low learning rate.
* **Train your own CNN** when you have a large dataset, or when inference speed and model size matter more than a small accuracy difference. Avoid it with little data.

## Method

**Data.** CIFAR-10 (32×32 colour images, 10 classes). From the 50,000 training images, 500 per class are held out as a fixed validation set. The remaining 4,500 per class form a training pool, from which stratified, nested subsets of 500 / 2,000 / 10,000 / 45,000 images are taken (each smaller set is contained in the larger ones). The official 10,000-image test set is used only for the final evaluation.

**Models.**

* *Custom CNN:* three blocks of Conv–BatchNorm–ReLU ×2 → MaxPool → Dropout (32, 64 and 128 filters), then global average pooling, dropout and a dense layer.
* *MobileNetV2:* inputs upscaled from 32 to 128 px and scaled to \[−1, 1]; ImageNet weights; global average pooling to 1,280 features; new 10-class dense head. The backbone always runs in inference mode so its BatchNorm statistics stay fixed.

**Training.** AdamW, batch size 32, up to 50 epochs, early stopping on validation loss (patience 10, best weights restored) and learning-rate halving on plateau. Learning rate 1e-3 for A and B, 1e-5 for fine-tuning. The random seed is reset to 42 before each model.

**Measurements.** Test accuracy, macro-F1, trainable parameters, training time (C includes the time of stage B) and single-image latency (median of 30 runs after warm-up, compiled with `tf.function`).

## Limitations

* Each configuration was trained with a single seed; small differences (under about 1 point) are within run-to-run noise.
* The custom CNN reached the 50-epoch limit at 2,000 and 10,000 images and might improve slightly with longer training.
* MobileNetV2 receives blurry, upscaled 32 px images rather than the sharp photos it was trained on.
* On 500 images, the default BatchNorm momentum (0.99) left the CNN's validation accuracy stuck at 10%; momentum 0.9 was needed.
* Data augmentation is implemented but was not used in the main comparison.

## Run it yourself

1. Open the notebook in Google Colab (badge above) and select a GPU runtime: *Runtime → Change runtime type → T4 GPU*.
2. Download [CIFAR-10 (Python version)](https://www.cs.toronto.edu/~kriz/cifar.html), extract it, and put the `cifar-10-batches-py` folder in your Google Drive under `MyDrive/archive/`.
3. Run all cells. The full experiment takes roughly an hour on a T4; results are saved to Drive after each training size.

Built with TensorFlow/Keras, scikit-learn, NumPy, pandas and Matplotlib.

## Repository contents

```
├── UGP\\\_Proj2.ipynb      # full notebook with code, outputs and analysis
├── results.csv          # results table from the experiment
├── images/              # figures used in this README
└── README.md
```

## Author

Nikita Pohrebniak, BSc Computer Science (AI specialization), Polish-Japanese Academy of Information Technology (PJATK), Warsaw. Project for the *Deep Learning in Python* course.

