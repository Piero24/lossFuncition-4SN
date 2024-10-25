
<div id="top"></div>

<br/>
<br/>
<br/>

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/9828/9828886.png" width="100" height="100">
</p>
<h1 align="center">
    <a href="https://github.com/Piero24/lossFuncition-4SN">HSNet+: Enhancing Polyp Segmentation with Region-wise Loss</a>
</h1>
<p align="center">
    <a href="https://github.com/Piero24/lossFuncition-4SN/commits/master">
    <img src="https://img.shields.io/github/last-commit/piero24/lossFuncition-4SN">
    </a>
    <a href="https://github.com/Piero24/twitch-stream-viewer/issues">
    <img src="https://img.shields.io/github/issues/piero24/lossFuncition-4SN">
    </a>
    <a href="https://github.com/Piero24/lossFuncition-4SN/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/piero24/lossFuncition-4SN">
    </a>
</p>
<p align="center">
    Department of Engineering Information, University of Padua
    <br/>
    <a href="https://github.com/Piero24">Pietrobon Andrea</a>
    •
    <a href="https://github.com/nicovii">Biffis Nicola</a>
    <br/>
    <a href="https://github.com/Piero24/lossFuncition-4SN/blob/master/Pietrobon_Biffis.pdf"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://colab.research.google.com/drive/1TEcddIvATuULZxx9QQD3hXsm5ahs0iQv?usp=sharing">View Demo on Google Colab</a>
    •
    <a href="https://github.com/Piero24/lossFuncition-4SN/issues">Report Bug</a>
</p>

---

<br/><br/>
<h3 align="center">⚠️ <strong>IMPORTANT NOTE BEFORE START</strong> ⚠️</h3>
<p align="center">
    This project was tested on an <strong>NVIDIA GeForce GTX 1050 Ti</strong> with <strong>4GB</strong> of memory, as well as with the <strong>Google Colab</strong> free plan. Consequently, it had limited usage of a GPU. This allowed us to achieve the following results, which, unfortunately, in our opinion, are not entirely satisfactory. However, we are confident that with the use of a more powerful GPU, better results can be obtained. For this reason, we encourage anyone who wants to try using GPUs with significantly more memory available.
</p>

<br/>
<h2 id="itroduction">📔  Introduction</h2>
<p>
    This research will show an innovative method useful in the segmentation of polyps during the screening phases of colonoscopies with the aim of concretely helping doctors in this task. To do this we have adopted a new approach which consists in <strong>merging the hybrid semantic network (HSNet) architecture model with the Reagion-wise(RW) as a loss function</strong> for the backpropagation process. In this way the bottleneck problems that arise in the systems currently used in this area are solved, since thanks to the HSNet it is possible to exploit the advantages of both the Transformers and the convolutional neural networks and thanks to the RW loss function its capacity is exploited to work efficiently with biomedical images. Since both the architecture and the loss function chosen by us have shown that they can achieve performances comparable to those of the state of the art working individually, in this research a dataset divided into 5 subsets will be used to demonstrate their effectiveness by using them together .
</p>
<br/>
<table>
  <tr  align="center">
    <th><strong>Original Image</strong></th>
    <th><strong>True Mask</strong></th> 
    <th><strong>Output</strong></th> 
  </tr>
  <tr  align="center">
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/148.png?token=GHSAT0AAAAAACJ7YHTZNSNPDVD7R2SY2YXCZMTGB2A" alt="Original Image"></th>
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/148-t.png?token=GHSAT0AAAAAACJ7YHTZ6JAF4WO6XQVP3ONQZMTGG6Q" alt="True Mask"></th> 
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/148-p.png?token=GHSAT0AAAAAACJ7YHTZWWS52FKTUKEIZZGKZMTGHRA" alt="Output"></th> 
  </tr>
  <tr  align="center">
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/165.png?token=GHSAT0AAAAAACJ7YHTZIDKY5AIBP7GLMIJWZMTGI4A" alt="Original Image"></th>
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/165-t.png?token=GHSAT0AAAAAACJ7YHTZGHT22Q2H4TIQDJ6SZMTGJGQ" alt="True Mask"></th> 
    <th><img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/165-p.png?token=GHSAT0AAAAAACJ7YHTYPK3WAJPOIRLRD4O2ZMTGJTA" alt="Output"></th> 
  </tr>
</table>
<div align="center">
  <img src="https://raw.githubusercontent.com/Piero24/lossFuncition-4SN/master/LaTeX/Figures/50-plot.png?token=GHSAT0AAAAAACJ7YHTZJAVWRRH25MHRRT3OZMTGJ5A" alt="Plot of the Training">
</div>
<br/>
<br/>
<h2 id="made-in"><br/>🛠  Built in</h2>
<p>
    This project was built using a variety of technologies like: <a href="https://www.python.org/downloads/release/python-390/"><strong>Python 3.9</strong></a>, renowned for its versatility and readability; <a href="https://pytorch.org"><strong>PyTorch</strong></a>, revered for its prowess in deep learning applications; <a href="https://opencv.org"><strong>OpenCV</strong></a>, an indispensable tool for computer vision tasks; <a href="https://numpy.org"><strong>Numpy</strong></a>, revered for its array computing capabilities; along with a plethora of other third party libraries like <a href="https://github.com/MIC-DKFZ/nnUNet#"><strong>nnUNet</strong></a>, that collectively contribute to the project's robustness and functionality. 
</p>
<br/>

<p align="right"><a href="#top">⇧</a></p>


<h2 id="index"><br/>📋  Index</h2>
<ul>
    <li><h4><a href="#documentation">Documentation</a></h4></li>
    <li><h4><a href="#prerequisites">Prerequisites</a></h4></li>
    <li><h4><a href="#how-to-start">How to Start</a></h4></li>
    <li><h4><a href="#structure-of-the-project">Structure of the Project</a></h4></li>
    <li><h4><a href="#responsible-disclosure">Responsible Disclosure</a></h4></li>
    <li><h4><a href="#license">License</a></h4></li>
    <li><h4><a href="#third-party-licenses">Third Party Licenses</a></h4></li>
</ul>

<p align="right"><a href="#top">⇧</a></p>


<h2 id="documentation"><br/><br/>📚  Documentation</h2>
<p>
    In the field of biomedicine, quantitative analysis requires a crucial step: image segmentation. Manual segmentation is a time-consuming and subjective process, as demonstrated by the considerable discrepancy between segmentations performed by different annotators. Consequently, there is a strong interest in developing reliable tools for automatic segmentation of medical images.
    <br/>
    <br/>
    The use of neural networks to automate polyp segmentation can provide physicians with an effective tool for identifying such formations or areas of interest during clinical practice. However, there are two important challenges that limit the effectiveness of this segmentation:
</p>
<p>
    <ol>
        <li>Polyps can vary significantly in size, orientation, and illumination, making accurate segmentation difficult to achieve.</li>
        <li>Current approaches often overlook significant details such as textures.</li>
    </ol>
</p>
<p>
    To obtain precise segmentations in medical imagesegmentation, it is crucial to consider class imbalance and the importance of individual pixels. By pixel importance, we refer to the phenomenon where the severity of classification errors depends on the position of such errors.
    <br/>
    <br/>
    Current approaches for polyp segmentation primarily rely on convolutional neural networks (CNN)
    or Transformers, so to overcome the mentioned challenges, this research proposes the use of a hybrid semantic network (HSNet) that combines the advantages of Transformer networks and convolutional neural networks (CNN), along with regional loss (RW). Thanks to this loss function we can simultaneously takes into account class imbalance and pixel importance, without requiring additional hyperparameters or functions, in order to improve polyp segmentation.
    <br/>
    <br/>
    In this study, we examine how the implementation of regional loss (RW) affects polyp segmentation by applying it to a hybrid semantic network (HSNet).
</p>
<p>
    Read the paper for a more detailed explanation: <a href="https://github.com/Piero24/lossFuncition-4SN/blob/master/Pietrobon_Biffis.pdf">Documentation »</a>
</p>


<p align="right"><a href="#top">⇧</a></p>


<h2 id="prerequisites"><br/>🧰  Prerequisites</h2>
<p>
    This are the dependencies you need to test the project. It's strongly suggested to use python 3.9 (or higher).
</p>

```bash
pip3 install thop
pip3 install libtiff
pip3 install timm
pip3 install opencv-python
pip3 install scipy
pip3 install numpy
pip3 install nibabel
pip3 install nnunet
pip3 install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

<br/>
<h3 align="center">
    You can download the dataset from this <a href="https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing">Google Drive Link</a>
</h3>
<p align="right"><a href="#top">⇧</a></p>


<h2 id="how-to-start"><br/>⚙️  How to Start</h2>
<p>
    Those are the steps to follow to test the code. Note that you need a NVIDIA GPU to run the code. If when you start the code you get an error like this:
</p>

```sh
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0;
4.00 GiB total capacity; 3.36 GiB already allocated; 0 bytes free; 3.48 GiB reserved
in total by PyTorch) If reserved memory is >> allocated memory try setting
max_split_size_mb to avoid fragmentation.See documentation for Memory Management
and PYTORCH_CUDA_ALLOC_CONF
```
<p>
    It means that you don't have enough memory on your GPU. For this reason a GPU with at least 8GB of memory is recommended. Alternatively you can test it on <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/2560px-Google_Colaboratory_SVG_Logo.svg.png"  width="35" height="auto" align="center"> <strong>Google Colab</strong> from <a href="https://colab.research.google.com/drive/1TEcddIvATuULZxx9QQD3hXsm5ahs0iQv?usp=sharing"><strong>HERE</strong></a>.
</p>  
<br/>

1. Clone the repo
  
    ```sh
    git clone https://github.com/Piero24/lossFuncition-4SN.git
    ```

2. Download the dataset from <a href="https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing">Google Drive</a>

3. Change the paths of the dataset in `folder_path.py`

4. Check the hyperparameters.

    The hyperparameters used in our tests are the followings:

    a) **learning rate:** 0.0001 \
    b) **training size:** 160 \
    c) **batch size:** 24 \
    d) **number of epochs:** 50

    > **⚠️ ATTENTION:** Be careful when changing the training size and batch size. For an unclear problem (maybe internal to pytorch) we can use multiples of 8 up to 352 for the training size and the batch size must be a dividend of the training size.


5. Start the net with `python main/Train.py`

     > **NOTE:** At the end of each epoch the test will be carried out.
     
     > The model for each test will be saved.Every 10 epochs the dice plot for the test phase is saved (Each plot starts from the very first epoch up to the current number, for example 0-10, 0-20 ... 0-100). At the same time a table is saved with the actual dice value for the last 10 test sets (e.g. 0-10, 10-20 ... 90-100). 
     
     > Once this phase has been completed, each saved model will be tested on a portion of the test set and will save the relative masks in order to be able to make a visual comparison.


<p align="right"><a href="#top">⇧</a></p>


---

<h3 id="responsible-disclosure"><br/>📮  Responsible Disclosure</h3>
<p>
    We assume no responsibility for an improper use of this code and everything related to it. We do not assume any responsibility for damage caused to people and / or objects in the use of the code.
</p>
<strong>
    By using this code even in a small part, the developers are declined from any responsibility.
</strong>
<br/>
<br/>
<p>
    It is possible to have more information by viewing the following links: 
    <a href="#code-of-conduct"><strong>Code of conduct</strong></a>
     • 
    <a href="https://github.com/Piero24/lossFuncition-4SN/blob/master/LICENSE"><strong>License</strong></a>
</p>

<p align="right"><a href="#top">⇧</a></p>


--- 

<h2 id="license"><br/>🔍  License</h2>
<strong>MIT License</strong>
<br/>
<i>Copyright (c) 2023 Andrea Pietrobon</i>
<br/>
<br/>
<i>Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:</i>
<br/>
<br/>
<i>The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.</i>
<br/>
<br/>
<a href="https://github.com/Piero24/lossFuncition-4SN/LICENSE"><strong>License Documentation »</strong></a>
<br/>
<br/>

<h3 id="third-party-licenses"><br/>📌  Third Party Licenses</h3>

The individual licenses are indicated in the following section.
<br/>
<br/>
<strong>Software list:</strong>
<br/>
<table>
  <tr  align="center">
    <th>Software</th>
    <th>License owner</th> 
    <th>License type</th> 
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/baiboat/HSNet">HSNet</a></td>
    <td><a href="https://github.com/baiboat">baiboat</a></td> 
    <td>*</td>
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/james128333/HarDNet-MSEG">HarDNet-MSEG</a></td>
    <td><a href="https://github.com/james128333">james128333</a></td> 
    <td>Apache-2.0</td>
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/DengPingFan/Polyp-PVT">Polyp-PVT</a></td>
    <td><a href="https://github.com/DengPingFan">DengPingFan</a></td>
    <td>*</td>
  </tr>
  <tr  align="center">
    <td><a href="https://arxiv.org/abs/2108.01405">Region-wise Loss for Biomedical Image Segmentation</a></td>
    <td>Juan Miguel Valverde, Jussi Tohka</td>
    <td></td>
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/huggingface/pytorch-image-models">pytorch-image-models</a></td>
    <td><a href="https://github.com/huggingface">huggingface</a></td>
    <td>Apache-2.0</td>
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/Lyken17/pytorch-OpCounter">thop</a></td>
    <td><a href="https://github.com/Lyken17">Lyken17</a></td>
    <td>MIT</td>
  </tr>
  <tr  align="center">
    <td><a href="https://github.com/MIC-DKFZ/nnUNet#">nnUNet</a></td>
    <td><a href="https://github.com/MIC-DKFZ">MIC-DKFZ</a></td>
    <td>Apache-2.0</td>
  </tr>
</table>

<p align="right"><a href="#top">⇧</a></p>


---
> *<p align="center"> Copyrright (C) by Pietrobon Andrea <br/> Released date: July-2023*
