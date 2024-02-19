# FF_rays

This is a first attempt at trying to design Flying Focus optics for Plasma Wakefield Accelerators by considering a ray tracing problem of different conic sections of mirrors and individually moving each point of the mirror to find the optimal shape. The target shape is a parabola which has the unique property that all rays will end at a singular point along the optical axis. 

There are two folders that actually execute the gradient descent algorithm: `optimizing_k` and `optimitizing_mirror_shape`. 
* `optimizing_k` contains a Jupyter notebook with plots that showcase how the algorithm works as well as `raytrace_k.py` which contains the functions for the algorithm and `run_grad_desc_k.py` which will actually run the algorithm to find the optimal conic number

*`optimizing_mirror_shape` also contains the `raytrace_mirror.py` which contains similar functions to `raytrace_k.py`, but now the functions have been altered such that each point of the optic moves independently of one another and the algorithm tries to converge to the optimal shape. Similarly, `run_grad_desc_mirror.py` will actually run the algorithm with the given initial parameters. An example run folder `dz-5e-07_k--0.50_eps-1.0e-07_lr-1.0e-02_No-20_Nr-20_N-50` has been saved here to show what the algorithm outputs when it is ran. To create a movie using the plots within the `raytrace` folder, one can use `create_gif.py`.

The other two folders are just for reference: `test_cases` and `ray_tracing`
* The `test_cases` folder has multiple jupyter notebooks that test different aspects of the code. 
* The `ray_tracing` folder shows notebooks of the raytracing algorithm 