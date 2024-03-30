# Python library : torch_hilbert

2D Hilbert Transformations on PyTorch - Code example based on research article


# Infos :

This example of Python code is the implementation of the article below.

# Reference :

Lorenzo-Ginori, Juan. (2007). An Approach to the 2D Hilbert Transform for Image Processing Applications. Lecture Notes in Computer Science. 157-165. 10.1007/978-3-540-74260-9_14.

https://www.researchgate.net/publication/221472037_An_Approach_to_the_2D_Hilbert_Transform_for_Image_Processing_Applications


<hr />


# Install library



```bash
%%bash
if !python -c "import torch_hilbert" 2>/dev/null; then
    pip install https://github.com/Simon-Bertrand/2DHilbertTransformations-PyTorch/archive/main.zip
fi
```

# Import library



```python
import torch_hilbert
```

# Load data



```python
!pip install -q torchvision requests matplotlib
import torchvision
import torch.nn.functional as F
import torch
im = torchvision.io.read_image("./imgs/example.png", torchvision.io.ImageReadMode.GRAY).divide(255)

```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


# Compute Hilbert transformations



```python
import torch_hilbert
import matplotlib.pyplot as plt


for mode in ["AS", "HT"]:
    for method in ["basic", "combined", "single_orthant", "directional"]:

        plt.imshow(
            torch_hilbert.HilbertTransformations(method, mode)(im)
            .abs()
            .moveaxis(0, -1)
        )
        plt.title(f"{method=} {mode=}")
        plt.colorbar()
        plt.show()
```


    
![png](figs/README_10_0.png)
    



    
![png](figs/README_10_1.png)
    



    
![png](figs/README_10_2.png)
    



    
![png](figs/README_10_3.png)
    



    
![png](figs/README_10_4.png)
    



    
![png](figs/README_10_5.png)
    



    
![png](figs/README_10_6.png)
    



    
![png](figs/README_10_7.png)
    



```python

```
