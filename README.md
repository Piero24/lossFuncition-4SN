
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
    <a href="https://github.com/Piero24/lossFuncition-4SN">
    <img src="https://img.shields.io/badge/Maintained-yes-green.svg">
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
    <a href="#index"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/Piero24/lossFuncition-4SN">View Demo</a>
    •
    <a href="https://github.com/Piero24/lossFuncition-4SN/issues">Report Bug</a>
</p>


---

<br/><br/>
<h2 id="itroduction">📔  Itroduction</h2>
<p>
    This research will show an innovative method useful in the segmentation of polyps during the screening phases of colonoscopies with the aim of concretely helping doctors in this task. To do this we have adopted a new approach which consists in <strong>merging the hybrid semantic network (HSNet) architecture model with the Reagion-wise(RW) as a loss function</strong> for the backpropagation process. In this way the bottleneck problems that arise in the systems currently used in this area are solved, since thanks to the HSNet it is possible to exploit the advantages of both the Transformers and the convolutional neural networks and thanks to the RW loss function its capacity is exploited to work efficiently with biomedical images. Since both the architecture and the loss function chosen by us have shown that they can achieve performances comparable to those of the state of the art working individually, in this research a dataset divided into 5 subsets will be used to demonstrate their effectiveness by using them together .
</p>
<br/>
<img src="https://github.githubassets.com/images/modules/site/social-cards/github-social.png">
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
    Read the paper for a more detailed explanation: <a href="https://shields.io/">Documentation »</a>
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
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 3.36 GiB already allocated; 0 bytes free; 3.48 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
<p>
    It means that you don't have enough memory on your GPU. For this reason a GPU with at least 8GB of memory is recommended. Alternatively you can test it on <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/2560px-Google_Colaboratory_SVG_Logo.svg.png"  width="35" height="auto" align="center"> <strong>Google Colab</strong> from <a href="https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing"><strong>HERE</strong></a>.
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
    <a href="#license"><strong>License</strong></a>
</p>

<p align="right"><a href="#top">⇧</a></p>


--- 

<h2 id="license"><br/>🔍  License</h2>
<strong>GNU GENERAL PUBLIC LICENSE</strong>
<br/>
<i>Version 3, 29 June 2007</i>
<br/>
<br/>
<i>Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/> Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.</i>
<br/>
<br/>
<i>Preamble</i>
<br/>
<i>The GNU General Public License is a free, copyleft license for software and other kinds of works.</i>
<br/>
<a href="https://github.com/Piero24/Template-README/blob/main/LICENSE"><strong>License Documentation »</strong></a>
<br/>
<br/>


<h3 id="authors-and-copyright"><br/>✏️  Authors and Copyright</h3>
<br/>
<p>
    👨🏽‍💻: <strong>Pietrobon Andrea</strong>
    <br/>
    🌐: <a href="https://www.pietrobonandrea.com">pietrobonandrea.com</a>
    <br/>
    <img src="https://assets.stickpng.com/thumbs/580b57fcd9996e24bc43c53e.png" width="30" height="30" align="center">:
    <a href="https://twitter.com/pietrobonandrea">@PietrobonAndrea</a>
    <br/>
    🗄: <a href="https://github.com/Piero24/lossFuncition-4SN">HSNet+: Enhancing Polyp Segmentation with Region-wise Loss</a>
</p>
<br/>
<p>
    My name is <strong>Pietrobon Andrea</strong>, a computer engineering student at the 
    <img src="https://upload.wikimedia.org/wikipedia/it/thumb/5/53/Logo_Università_Padova.svg/800px-Logo_Università_Padova.svg.png"  width="26" height="26" align="center"> 
    University of Padua (🇮🇹).
</p>
<p>
    My passion turns towards <strong>AI</strong> and <strong>ML</strong>.
    I have learned and worked in different sectors that have allowed me to gain skills in different fields, such as IT and industrial design.
    To find out more, visit my <a href="https://www.pietrobonandrea.com">
    <strong>website »</strong></a>
</p>
<p>
    <strong>The Copyright (C) of this project is held exclusively by my person.</strong>
</p>


<p align="right"><a href="#top">⇧</a></p>


<h3 id="third-party-licenses"><br/>📌  Third Party Licenses</h3>

In the event that the software uses third-party components for its operation, 
<br/>
the individual licenses are indicated in the following section.
<br/>
<br/>
<strong>Software list:</strong>
<br/>
<table>
  <tr  align="center">
    <th>Software</th>
    <th>License owner</th> 
    <th>License type</th> 
    <th>Link</th>
    <th>Note</th>
  </tr>
  <tr  align="center">
    <td>None</td>
    <td></td> 
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

<p align="right"><a href="#top">⇧</a></p>


---
> *<p align="center"> Copyrright (C) by Pietrobon Andrea <br/> Released date: July-2023*
