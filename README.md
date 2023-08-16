<p align="center">
  <h1 align="center"><ins>LightGlue ⚡️</ins><br>Local Feature Matching at Light Speed</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p>
<!-- <p align="center">
    <img src="assets/larchitecture.svg" alt="Logo" height="40">
</p> -->
  <!-- <h2 align="center">PrePrint 2023</h2> -->
  <h2 align="center"><p>
    <a href="https://arxiv.org/pdf/2306.13643.pdf" align="center">Paper</a> | 
    <a href="https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb" align="center">Colab</a>
  </p></h2>
  <div align="center"></div>
</p>
<p align="center">
    <a href="https://arxiv.org/abs/2306.13643"><img src="assets/easy_hard.jpg" alt="example" width=80%></a>
    <br>
    <em>LightGlue is a deep neural network that matches sparse local features across image pairs.<br>An adaptive mechanism makes it fast for easy pairs (top) and reduces the computational complexity for difficult ones (bottom).</em>
</p>

##

This repository based on the code of LightGlue and open cv to change the processing time of lightglue by using open cv
 - [check out the paper for more details](https://arxiv.org/pdf/2306.13643.pdf).

We release pretrained weights of LightGlue with [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566) local features.
The training end evaluation code will be released in July in a separate repo. To be notified, subscribe to [issue #6](https://github.com/cvg/LightGlue/issues/6).

## Installation and demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb)

Install this repo using pip:

```bash
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

We provide a [demo notebook](demo.ipynb) which shows how to perform feature extraction and matching on an image pair.



# Contribution 
- We used open cv to perform the image processing and use that image processing with the lightglue to check the processing time

<b>To see the implementation run cv_2_lightglue.ipynb file</b>

```python

# Visualize LightGlue results with OpenCV tools
def lightglue2opencv(points0, points1, matches, kpts0, kpts1, img0, img1, **kwargs):
    return {
        "src_pts": points0.numpy().reshape(-1, 1, 2),
        "dst_pts": points1.numpy().reshape(-1, 1, 2),
        "kp0": cv2.KeyPoint_convert(kpts0.numpy()),
        "kp1": cv2.KeyPoint_convert(kpts1.numpy()),
        "matches": tuple(
            cv2.DMatch(matches[i][0].item(), matches[i][1].item(), 0.) 
            for i in range(matches.shape[0])
        ),
        "img0": cv2.cvtColor((255 * img0).numpy().astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2GRAY),
        "img1": cv2.cvtColor((255 * img1).numpy().astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    }


%time results_lightglue_opencv = lightglue2opencv(**results_lightglue)
homography_lightglue = visualize_opencv(**results_lightglue_opencv, cfg=cfg.lightglue["homography"], title="LightGlue")


```

## Other links
- [hloc - the visual localization toolbox](https://github.com/cvg/Hierarchical-Localization/): run LightGlue for Structure-from-Motion and visual localization.
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX): export LightGlue to the Open Neural Network Exchange format.
- [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui): a web GUI to easily compare different matchers, including LightGlue.
- [kornia](https://kornia.readthedocs.io) now exposes LightGlue via the interfaces [`LightGlue`](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LightGlue) and [`LightGlueMatcher`](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LightGlueMatcher).




## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}
```



## License
The pre-trained weights of LightGlue and the code provided in this repository are released under the [Apache-2.0 license](./LICENSE). [DISK](https://github.com/cvlab-epfl/disk) follows this license as well but SuperPoint follows [a different, restrictive license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) (this includes its pre-trained weights and its [inference file](./lightglue/superpoint.py)).
