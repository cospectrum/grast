# grast

Automatic differentiation of generic fields for Python

## Install

```sh
pip install grast
```

## Usage

```py
from grast import var

x = var('x')
y = var('y')

f = x/y + y**x
df = f.grad()

df_dx = df['x']
df_dy = df['y']

args = dict(x=3, y=-5)
f(args)
df_dx(args)
df_dy(args)
```

## References

1. F. Krawiec, S. Peyton Jones, N. Krishnaswami, T. Ellis, R. A. Eisenberg, A. Fitzgibbon. 2022. 
Provably correct, asymptotically efficient, higher-order reverse-mode automatic differentiation. 
Proc. ACM Program. Lang., 6, POPL (2022), 1–30. <https://doi.org/10.1145/3498710>

2. Jerzy Karczmarczuk. 1998. Functional Differentiation of Computer Programs. 
In Proceedings of the Third ACM SIGPLAN International Conference on Functional 
Programming (Baltimore, Maryland, USA) (ICFP ’98). Association for Computing 
Machinery, New York, NY, USA, 195-203. <https://doi.org/10.1145/289423.289442>
