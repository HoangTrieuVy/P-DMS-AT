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

python dms_denoiser.py --z ..\notebooks\10081_noisy.jpg --algo SLPAM  --b 4 --l 1e-2 

python dms_denoiser.py --z ..\notebooks\10081_noisy.jpg  --algo SLPAM--norm AT --b 4 --l 1e-2 --eps 0.02 --eps_AT_min 0.002

python dms_denoiser.py --z ..\notebooks\10081_noisy.jpg  --algo SLPAMeps-descent --norm AT --b 4 --l 3e-3 --eps 0.02 --eps_AT_min 0.002

 ```
  