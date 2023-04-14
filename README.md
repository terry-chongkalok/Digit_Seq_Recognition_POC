# POC on Digit Sequence Recognition from Images

This digit sequence recognition model is a CRNN model using CTCLoss [[1]](https://distill.pub/2017/ctc/) built from scratch using Pytorch.

During the early stage in model development, it is found that the model failed to learn even after a long training time, and can only return blank sequences. It seemed that it was not a uncommon issue when training CRNN model with CTCLoss [[2]](https://discuss.pytorch.org/t/training-crnn-with-ctcloss-returns-only-blank-labels-after-few-iterations/72581) [[3]](https://stackoverflow.com/questions/49483394/ctc-loss-goes-down-and-stops) [[4]](http://www.tbluche.com/ctc_and_blank.html).

In order to tackle this issue, the following measures were adopted:
1. Pretrained a CNN model using single digit images, and freezed its parameters when transferring it to the CNN part of the CRNN model, so the features of digits could be extracted properly.

2. Used Cyclic learning rate instead of Adam in order to prevent the model from getting stuck in local minima (that can only predict blank labels)

3. Used a larger model with more nodes in the RNN (GRU) part, more image data and higher number of epochs for training to prevent from underfitting

Based on the above rationale, the complete workflow of the project is as follows:
1. Notebook `1_Gen_Images` created 3 types of digit sequence images using fonts (downloaded from online) under folder `font`:
    - 2,000 single-digit images under folder `img/single_digit` (for pretraining CNN model)
    - 20,000 five-digit images under folder `img/five_digits` (actual images for training the CRNN model)
    - 100 12-digit images under folder `img/12_digits` (a few testing images to test the transferabiblity of the model on longer-sequence images)
    
2. Notebook `2_Pretrain_CNN` trained a CNN model by data in folder `img/single_digit`, and the model was saved to folder `Model/Pretrained_CNN`

3. Notebook `3_CRNN` trained a CNRN model by data in folder `img/five_digits` based on the CNN model pretrained in step 2

And the performance of the final model:
- Accuracy on successfully recognized all digits correctly from 1,000 testing images of 5-digit sequences: **83.5%**
- Accuracy on successfully recognized all digits correctly from 100 testing images of 12-digit sequences: **52%**

## Project Folder Structure:

    Digit_Seq_Recognition_POC/
    ├─ font/
    ├─ img/
    │   ├─ single_digit/
    │   ├─ five_digits/
    │   └─ 12_digits/
    ├─ Model/
    │   ├─ pretrained_CNN/
    │   └─ CRNN/
    ├─ 1_Gen_Images.ipynb
    ├─ 2_Pretrain_CNN.ipynb
    ├─ 3_CRNN.ipynb
    ├─ __init__.py
    ├─ early_stop.py
    ├─ random_seed.py
    ├─ requirement.txt
    └─ README.md