# Joint image restoration and contour detection using Discrete Mumford-Shah with Ambrosio-Tortorelli penalization
> **Hoang Trieu Vy Le, [Nelly Pustelnik](https://perso.ens-lyon.fr/nelly.pustelnik/), [Marion Foare](https://perso.ens-lyon.fr/marion.foare/),**
*Proximal based strategies for solving Discrete Mumford-Shah with Ambrosio-Tortorelli penalization on edges,* IEEE Signal Processing Letters, vol. 29, pp. 952- 956, 2022. [Download](https://ieeexplore.ieee.org/abstract/document/9723590)

## <div align="center">Quick start Examples </div>

<details open>
<summary>Install</summary>

Clone the repository and install [requirements.txt]() in a
[**Python>=3.7.0**](https://www.python.org/) environment,


```bash
git clone https://github.com/HoangTrieuVy/GGS-DMS # clone
cd your_repo
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Denoisers</summary>
Denoising with discrete Mumford-Shah functional:

 $$
\begin{equation}
 \min_{x,e} \frac{1}{2} \Vert A x - z \Vert_2^2 + \beta \Vert (1-e)\odot Dx \Vert_2^2 + \lambda h(e)
\end{equation}
 $$
 
 Remarks:
 * Input degraded images can be in standard image format (.png, .jpeg,...)  or .mat file (where the noisy image shuld be denoted by 'z').
 * As an option, the original image can be specified allowing to find the optimal $\lambda$ and $\beta$ hyperparameters minimizing the PSNR.
 * The penalization over the edges variable can be either the $\ell_1$-norm or Ambrosio-Tortorelli penalization.
 * The default method is PALM with $h = \Vert \cdot \Vert_1$.
 * The default parameter for Ambrosio-Tortorelli is  eps = 0.02.
 * SLPAM with Ambrosio-Tortorelli penalization is numerically costly.

```bash
optional arguments:
  -h, --help   show this help message and exit
  --z Z        noisy image path
  --x X        original image path
  --b B        beta
  --l L        lambda
  --algo ALGO  PALM, SLPAM,PALM-eps-descent,SLPAM-eps-descent
  --norm NORM  l1, AT
  --eps EPS    epsilon in the Ambrosio-Tortorelli penalization
  --eps EPS    minimum epsilon when using AT eps decreasing to 0 
  --it IT      number of iteration
 ```
 
```bash
cd examples

python dms_denoiser.py --z 10081_noisy.jpg

python dms_denoiser.py --z 10081_noisy.jpg  --algo PALM --norm AT --b 4 --l 1e-2 --eps 0.02

python dms_denoiser.py --z 10081_noisy.jpg  --algo PALM-eps-descent --norm AT --b 4 --l 3e-3 --eps 0.02 --eps_AT_min 0.002

 ```
  
 
<img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/examples/results_SLPAM_l1_10081_noisy.jpg" >

</details>




## <div align="center">Reproducing IEEE SPL results </div>

<details open>
<summary>Comparison of several DMS strategies on a synthetic noisy image degraded with white Gaussian noise </summary>
 
Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python

```bash
cd SPL-fig4
python dms_spl_fig4.py
```
  
Running [Home et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig4
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_ggs ; exit"
```
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig4/Screenshot%202022-10-18%20032114.png" >
</details>

<details open>
<summary>Comparison of several DMS strategies on several realization of synthetic noisy and blur images. </summary>
 
Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python

```bash
cd SPL-fig5
python dms_ggs # running different schemes on dms
python trof_ggs # TV and T-ROF 
```
  
Running [Hohm et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig5
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_ggs ; exit"
```
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig5/Screenshot%202022-10-18%20032056.png" >
</details>

<details open>
  
  
<summary>Comparison of several DMS strategies on real images from BSDS500 dataset.</summary>

Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python
```bash
cd SPL-fig6
python dms_real_std_0_05
python trof_ggs_real_std_0_05
```
Running [Hohm et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig6
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_figure6_std_0_05 ; exit"
```
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig6/Screenshot%202022-10-18%20031023.png" >

</details>



