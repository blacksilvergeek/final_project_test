# Test for Wave Equation(1D)
## Aim 

Test accuracy of idrlnet

## Why Wave Equation

1. Known answer
2. Not complicated, easy to compute

## Modeling Process

Consider the 1d wave equation:
$$
\begin{equation}
\frac{\partial^2u}{\partial t^2}=c\frac{\partial^2u}{\partial x^2},
\end{equation}
$$
where $c>0$​​​ is unknown and is to be estimated. Three groups of data pairs $\{x_i, t_i, u_i\}_{i=1,2,\cdot,N_1}$​​​  $\{x_j, t_j, {u_x}_j\}_{j=1,2,\cdot,N_2}$​​​ $\{x_k, t_k, {u_t}_k\}_{k=1,2,\cdot,N_3}$​​​  is observed.

To simplify the question, we adapt $c=1$ and the expected solution is $sin(x - t)$​

Assume we have known Dirichlet boundary conditions and the region is $[0,1]\times[0,1]$.

### To optimization problem

Then the problem is formulated as:
$$
\min_{u,c}\quad \omega_1\sum_{i=1,2,\cdots,N_1} \|u(x_i, t_i)-u_i\|^2+        \omega_2 \sum_{j=1,2,\cdots,N_2} \|u_x(x_i, t_j)-{u_x}_j\|^2          \omega_3 +\sum_{k=1,2,\cdots,N_3} \|u_t(x_k, t_k)-{u_t}_k\|^2\\
s.t. \frac{\partial^2u}{\partial t^2}=c\frac{\partial^2u}{\partial x^2}
$$



## Work on code

For simplicity, we write t as y. In this way the initial value condition can be seen as providing a boundary condition for a square region

### Define symbols and region

```{python}
x, y = symbols("x y")
u = Function('u')(x, y)

geo = sc.Rectangle((0, 0), (1, 1))
```

### Define boundary condition

```python
@sc.datanode(name="left_right", sigma=100)
def left_right():
    points = geo.sample_boundary(100, sieve=(y > 0) & (y < 1))
    constraints = sc.Variables({"u__x": cos(x-y),"u": sin(x-y)})
    return points, constraints

def up_down_domain():
    points = geo.sample_interior(100, sieve=(x > 0) & (x < 1))
    constraints = sc.Variables({"u__y": -cos(x-y),"u": sin(x-y)})
    return points, constraints
```

### Define PDE

```{python}
pde = sc.ExpressionNode(u.diff(x,2) - u.diff(y,2), name='pde')
@sc.datanode(name="interior")
def interior():
    points = geo.sample_interior(10000, low_discrepancy=True)
    constraints = {"pde": 0}
    return points, constraints
```

### Define solver

```{python}
s = sc.Solver(
    sample_domains=(interior(), left_right(), up_down_domain()),
    netnodes=[net],
    pdes=[pde],
    max_iter=3000,
)
s.solve()
```

## Result analysis

### Max accuracy loss

```{python}
coord = s.infer_step({"interior": ["x", "y", "u"]})
num_x = coord["interior"]["x"].cpu().detach().numpy().ravel()
num_y = coord["interior"]["y"].cpu().detach().numpy().ravel()
num_u = coord["interior"]["u"].cpu().detach().numpy().ravel()
num_u_real = np.sin(num_x-num_y)
print(max(np.abs(num_u - num_u_real)))

```

after 1000 iteration, this error is 0.009

![image-20211224135530649](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20211224135530649.png)



### Scatter Plot

```{python}
plt.rcParams['figure.figsize'] = (8.0, 8.0)
ax = plt.axes(projection='3d')

xdata=num_x
ydata=num_y
zdata=num_u
zdata2=num_u_real
ax.set_zlim(-0.35, 0.35)
ax.scatter3D(xdata, ydata, zdata-zdata,s=0.1,label="predict")
ax.scatter3D(xdata, ydata, zdata,s=0.1,c="g",label="error")
ax.legend()
```

![output](E:\Desktop\jpt_wd\output.png)

