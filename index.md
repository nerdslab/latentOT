## Making transport more robust and interpretable by moving data through a small number of anchor points

## Abstract
Optimal transport (OT) is a widely used technique for distribution alignment, with applications throughout the machine learning, graphics, and vision communities. Without any additional structural assumptions on transport, however, OT can be fragile to outliers or noise, especially in high dimensions. Here, we introduce a new form of structured OT that simultaneously learns low-dimensional structure in data while leveraging this structure to solve the alignment task. Compared with OT, the resulting transport plan has better structural interpretability, highlighting the connections between individual data points and local geometry, and is more robust to noise and sampling. We apply the method to synthetic as well as real datasets, where we show that our method can facilitate alignment in noisy settings and can be used to both correct and interpret domain shift.

## Method
The main idea behind LOT is to factorize the transport plan into three components, where mass is moved: (i) from individual source points to source anchors, (ii) from the source anchors to target anchors, and (iii) from target anchors to individual target points. This is shown in Figure 1 along side other optimal transport methods.

The factorized form is written as follows:

![](https://latex.codecogs.com/svg.latex?\mathbf{P}=\mathbf{P}_{x}%20\operatorname{diag}\left(\mathbf{u}_{z}^{-1}\right)%20\mathbf{P}_{z}%20\operatorname{diag}\left(\mathbf{v}_{z}^{-1}\right)%20\mathbf{P}_{y})

![](imgs/figure1)
**Figure 1: Comparison of transport plans obtained for different methods applied to clustered data**

## Results

### Domain shift
![](imgs/figure3)
**Figure 3: Visualization of transport of handwritten digits in the unbalanced case**

## Citation

```bibtex
@misc{lin2020making,
      title={Making transport more robust and interpretable by moving data through a small number of anchor points}, 
      author={Chi-Heng Lin and Mehdi Azabou and Eva L. Dyer},
      year={2020},
      eprint={2012.11589},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
