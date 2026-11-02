"""
Implementations of the corresponding Autograd functions
"""

import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

try:
    import finufft

    FINUFFT_AVAIL = True
except ImportError:
    FINUFFT_AVAIL = False

try:
    import cufinufft

    if cufinufft.__version__.startswith("1."):
        warnings.warn("pytorch-finufft does not support cufinufft v1.x.x")
    else:
        CUFINUFFT_AVAIL = True
except ImportError:
    CUFINUFFT_AVAIL = False

if not (FINUFFT_AVAIL or CUFINUFFT_AVAIL):
    raise ImportError(
        "No FINUFFT implementation available. "
        "Install either finufft or cufinufft and ensure they are importable."
    )

import pytorch_finufft.checks as checks

newaxis = None


def get_nufft_func(
    dim: int, nufft_type: int, device: torch.device
) -> Callable[..., torch.Tensor]:
    if device.type == "cuda":
        if not CUFINUFFT_AVAIL:
            raise RuntimeError("CUDA device requested but cufinufft failed to import")
        # note: in the future, cufinufft may figure out gpu_device_id on its own
        # see: https://github.com/flatironinstitute/finufft/issues/420
        return functools.partial(
            getattr(cufinufft, f"nufft{dim}d{nufft_type}"), gpu_device_id=device.index
        )

    if not FINUFFT_AVAIL:
        raise RuntimeError("CPU device requested but finufft failed to import")
    # CPU needs extra work to go to/from torch and numpy
    finufft_func = getattr(finufft, f"nufft{dim}d{nufft_type}")

    def f(*args, **kwargs):
        new_args = [arg for arg in args]
        for i in range(len(new_args)):
            if isinstance(new_args[i], torch.Tensor):
                new_args[i] = new_args[i].data.numpy()

        return torch.from_numpy(finufft_func(*new_args, **kwargs))

    return f


def coordinate_ramps(shape, device):
    start_points = -(torch.tensor(shape, device=device) // 2)
    end_points = start_points + torch.tensor(shape, device=device)
    coord_ramps = torch.stack(
        torch.meshgrid(
            *(
                torch.arange(start, end, device=device)
                for start, end in zip(start_points, end_points)
            ),
            indexing="ij",
        )
    )

    return coord_ramps[newaxis]


def batch_fftshift(x: torch.Tensor, n_shifted_dims: int) -> torch.Tensor:
    """fftshift only over the final n_shifted_dims dimensions"""
    out: torch.Tensor = torch.fft.fftshift(x, dim=tuple(range(-n_shifted_dims, 0)))
    return out


def batch_ifftshift(x: torch.Tensor, n_shifted_dims: int) -> torch.Tensor:
    """ifftshift only over the final n_shifted_dims dimensions"""
    out: torch.Tensor = torch.fft.ifftshift(x, dim=tuple(range(-n_shifted_dims, 0)))
    return out

class FinufftPlan:

    def __init__(
        self,
        type: int,
        dims: Union[int, Tuple[int, ...]],
        n_trans: int = 1,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.complex64,
        isign: int = 1,
        modeord: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        self.type = int(type)
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.n_trans = int(n_trans)
        self.eps = float(eps)
        self.dtype = dtype
        self.isign = int(isign)
        self.modeord = int(modeord)
        self.device = torch.device(device)

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        if self.device.type == "cuda":
            if not CUFINUFFT_AVAIL:
                raise RuntimeError("CUDA requested but cufinufft not available")

            torch.cuda.set_device(self.device.index)

            kwargs.pop("modeord", None)
            kwargs.setdefault("upsampfac", 1.25)

            import numpy as np
            np_dtype = np.complex64 if dtype == torch.complex64 else np.complex128

            self._plan = cufinufft.Plan(
                self.type,
                self.dims,
                n_trans=self.n_trans,
                eps=self.eps,
                dtype=np_dtype,
                isign=self.isign,
                gpu_device_id=self.device.index,
                **kwargs,
            )
            self._backend = "cufinufft"
        else:
            if not FINUFFT_AVAIL:
                raise RuntimeError("CPU requested but finufft not available")

            import numpy as np
            np_dtype = np.complex64 if dtype == torch.complex64 else np.complex128

            self._plan = finufft.Plan(
                self.type,
                self.dims,
                n_trans=self.n_trans,
                eps=self.eps,
                dtype=np_dtype,
                isign=self.isign,
                modeord=self.modeord,
                **kwargs,
            )
            self._backend = "finufft"
    
    def setpts(self, *points: torch.Tensor):
        """
        Set the non-uniform points.
        
        Parameters
        ----------
        *points : torch.Tensor
            Coordinates for each dimension (x, y, z, ...)
            Each should be a 1D tensor of length N (number of points)
        """
        # Validate points
        if len(points) != len(self.dims):
            raise ValueError(
                f"Expected {len(self.dims)} point arrays, got {len(points)}"
            )
        
        n_pts = points[0].shape[0]
        for i, p in enumerate(points):
            if p.shape[0] != n_pts:
                raise ValueError(
                    f"All point arrays must have same length, got {p.shape[0]} vs {n_pts}"
                )
        
        if self._backend == "cufinufft":
            # cuFINUFFT expects separate arrays for each dimension
            self._plan.setpts(*points)
        else:  # finufft
            # finufft expects numpy arrays
            import numpy as np
            numpy_points = [p.cpu().numpy() for p in points]
            self._plan.setpts(*numpy_points)
    
    def execute(self, data: torch.Tensor) -> torch.Tensor:
        if self._backend == "cufinufft":
            batch_shape = data.shape[:-1] if self.type == 1 else data.shape[:-len(self.dims)]
            data_flat = data.reshape(-1, data.shape[-1] if self.type == 1 else data.shape[-len(self.dims):])
            n_batch = data_flat.shape[0]
            
            if n_batch == 1 or self.n_trans == n_batch:
                out_flat = self._plan.execute(data_flat)
            else:
                out_flat = torch.cat([self._plan.execute(data_flat[i:i+1]) for i in range(n_batch)], dim=0)
            
            if self.type == 1:
                return out_flat.reshape(*batch_shape, *self.dims)
            return out_flat.reshape(*batch_shape, -1)
            
        else:
            batch_shape = data.shape[:-1] if self.type == 1 else data.shape[:-len(self.dims)]
            data_flat = data.reshape(-1, data.shape[-1] if self.type == 1 else data.shape[-len(self.dims):])
            
            out_np = self._plan.execute(data_flat.cpu().numpy())
            out_flat = torch.from_numpy(out_np).to(self.device)
            
            if self.type == 1:
                return out_flat.reshape(*batch_shape, *self.dims)
            return out_flat.reshape(*batch_shape, -1)
    
    def destroy(self):
        """Explicitly destroy the plan to free resources."""
        if hasattr(self._plan, 'destroy'):
            self._plan.destroy()
        self._plan = None
    
    def __del__(self):
        self.destroy()

class FinufftType1WithPlan(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, plan: FinufftPlan, plan_adj: FinufftPlan, points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        checks.check_devices(values, points)
        checks.check_dtypes(values, points, "Values")
        checks.check_sizes_t1(values, points)
        
        points = torch.atleast_2d(points)
        plan.setpts(*[points[i] for i in range(points.shape[0])])
        
        batch_dims = values.shape[:-1]
        values_flat = values.reshape(-1, values.shape[-1])
        finufft_out = plan.execute(values_flat)
        finufft_out = finufft_out.reshape(*batch_dims, *plan.dims)
        
        if plan.modeord:
            finufft_out = batch_ifftshift(finufft_out, points.shape[0])
        
        ctx.save_for_backward(points)
        ctx.plan_adj = plan_adj
        return finufft_out
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        points, = ctx.saved_tensors
        plan_adj = ctx.plan_adj
        
        if plan_adj.modeord:
            grad_output = batch_fftshift(grad_output, points.shape[0])
        
        plan_adj.setpts(*[points[i] for i in range(points.shape[0])])
        
        batch_dims = grad_output.shape[:-len(plan_adj.dims)]
        grad_output_flat = grad_output.reshape(-1, *plan_adj.dims)
        grad_values = plan_adj.execute(grad_output_flat)
        grad_values = grad_values.reshape(*batch_dims, -1)
        
        return None, None, None, grad_values


class FinufftType2WithPlan(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, plan: FinufftPlan, plan_adj: FinufftPlan, points: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        checks.check_devices(targets, points)
        checks.check_dtypes(targets, points, "Targets")
        checks.check_sizes_t2(targets, points)
        
        points = torch.atleast_2d(points)
        
        if plan.modeord:
            targets = batch_fftshift(targets, points.shape[0])
        
        plan.setpts(*[points[i] for i in range(points.shape[0])])
        
        batch_dims = targets.shape[:-points.shape[0]]
        targets_flat = targets.reshape(-1, *targets.shape[-points.shape[0]:])
        finufft_out = plan.execute(targets_flat)
        finufft_out = finufft_out.reshape(*batch_dims, points.shape[1])
        
        ctx.save_for_backward(points)
        ctx.plan_adj = plan_adj
        return finufft_out
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        points, = ctx.saved_tensors
        plan_adj = ctx.plan_adj
        
        plan_adj.setpts(*[points[i] for i in range(points.shape[0])])
        
        batch_dims = grad_output.shape[:-1]
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        grad_targets = plan_adj.execute(grad_output_flat)
        grad_targets = grad_targets.reshape(*batch_dims, *plan_adj.dims)
        
        if plan_adj.modeord:
            grad_targets = batch_ifftshift(grad_targets, points.shape[0])
        
        return None, None, None, grad_targets



class FinufftType1(torch.autograd.Function):
    """
    FINUFFT problem type 1
    """

    ISIGN_DEFAULT = -1  # note: FINUFFT default is 1
    MODEORD_DEFAULT = 1  # note: FINUFFT default is 0

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: Tuple[
            torch.Tensor, torch.Tensor, Any, Optional[Dict[str, Union[int, float]]]
        ],
        output: Any,
    ) -> None:
        points, values, _, finufftkwargs = inputs
        ctx.save_for_backward(points, values)

        if finufftkwargs is None:
            finufftkwargs = {}
        else:  # copy to avoid mutating caller's dictionary
            finufftkwargs = finufftkwargs.copy()
        ctx.isign = finufftkwargs.pop("isign", FinufftType1.ISIGN_DEFAULT)
        ctx.mode_ordering = finufftkwargs.pop("modeord", FinufftType1.MODEORD_DEFAULT)
        ctx.finufftkwargs = finufftkwargs

    @staticmethod
    def forward(  # type: ignore
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> torch.Tensor:
        checks.check_devices(values, points)
        checks.check_dtypes(values, points, "Values")
        checks.check_sizes_t1(values, points)
        points = torch.atleast_2d(points)
        ndim = points.shape[0]
        checks.check_output_shape(ndim, output_shape)

        if finufftkwargs is None:
            finufftkwargs = dict()
        else:  # copy to avoid mutating caller's dictionary
            finufftkwargs = finufftkwargs.copy()

        finufftkwargs.setdefault("isign", FinufftType1.ISIGN_DEFAULT)
        # pop because cufinufft doesn't support modeord
        modeord = finufftkwargs.pop("modeord", FinufftType1.MODEORD_DEFAULT)

        nufft_func = get_nufft_func(ndim, 1, points.device)

        batch_dims = values.shape[:-1]
        finufft_out = nufft_func(
            *points,
            values.reshape(-1, values.shape[-1]),
            output_shape,
            **finufftkwargs,
        )
        finufft_out = finufft_out.reshape(*batch_dims, *output_shape)

        if modeord:
            finufft_out = batch_ifftshift(finufft_out, ndim)

        return finufft_out

    @staticmethod
    def vmap(  # type: ignore[override]
        info: Any,
        in_dims: Tuple[Optional[int], ...],
        points: torch.Tensor,
        values: torch.Tensor,
        output_shape: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        batch_points, batch_values, *_ = in_dims
        if batch_values is not None:
            values = values.movedim(batch_values, 0)

        if batch_points is not None:
            # need a for-loop here
            points = points.movedim(batch_points, 0)
            if batch_values is not None:
                output = torch.stack(
                    [
                        FinufftType1.apply(
                            points[i],
                            values[i],
                            output_shape,
                            finufftkwargs,
                        )
                        for i in range(info.batch_size)
                    ],
                    dim=0,
                )
            else:
                output = torch.stack(
                    [
                        FinufftType1.apply(
                            points[i],
                            values,
                            output_shape,
                            finufftkwargs,
                        )
                        for i in range(info.batch_size)
                    ],
                    dim=0,
                )
        else:
            output = FinufftType1.apply(points, values, output_shape, finufftkwargs)

        return output, 0

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        _i_sign = -1 * ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, values = ctx.saved_tensors
        points = torch.atleast_2d(points)

        device = points.device
        ndim = points.shape[0]

        grads_points = None
        grad_values = None

        nufft_func = get_nufft_func(ndim, 2, device)

        if any(ctx.needs_input_grad):
            if _mode_ordering:
                grad_output = batch_fftshift(grad_output, ndim)

            # group together batched dimensions, if any
            shape = grad_output.shape[-ndim:]
            batch_dims = grad_output.shape[:-ndim]
            batched_grad_output = grad_output.reshape(-1, 1, *shape)
            nbatch = batched_grad_output.shape[0]

        if ctx.needs_input_grad[0]:
            # wrt points
            coord_ramps = coordinate_ramps(shape, device)

            # nbatch x ndims x ...
            batched_values = values.reshape(nbatch, 1, values.shape[-1])

            ramped_grad_output = (
                coord_ramps * batched_grad_output * 1j * _i_sign
            ).reshape(-1, *shape)

            backprop_ramp = (
                nufft_func(*points, ramped_grad_output, isign=_i_sign, **finufftkwargs)
                .conj()
                .reshape(nbatch, ndim, -1)
            )

            grads_points = (backprop_ramp * batched_values).real.sum(dim=0)

        if ctx.needs_input_grad[1]:
            grad_values = nufft_func(
                *points,
                batched_grad_output.squeeze(),
                isign=_i_sign,
                **finufftkwargs,
            ).reshape(*batch_dims, -1)

        return (
            grads_points,
            grad_values,
            None,
            None,
            None,
            None,
        )


class FinufftType2(torch.autograd.Function):
    """
    FINUFFT problem type 2
    """

    ISIGN_DEFAULT = -1  # note: FINUFFT default is -1
    MODEORD_DEFAULT = 1  # note: FINUFFT default is 0

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: Tuple[
            torch.Tensor, torch.Tensor, Optional[Dict[str, Union[int, float]]]
        ],
        output: Any,
    ) -> None:
        points, targets, finufftkwargs = inputs
        if finufftkwargs is None:
            finufftkwargs = {}
        else:  # copy to avoid mutating caller's dictionary
            finufftkwargs = finufftkwargs.copy()
        ctx.save_for_backward(points, targets)
        ctx.isign = finufftkwargs.pop("isign", FinufftType2.ISIGN_DEFAULT)
        ctx.mode_ordering = finufftkwargs.pop("modeord", FinufftType2.MODEORD_DEFAULT)
        ctx.finufftkwargs = finufftkwargs

    @staticmethod
    def forward(  # type: ignore
        points: torch.Tensor,
        targets: torch.Tensor,
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> torch.Tensor:
        checks.check_devices(targets, points)
        checks.check_dtypes(targets, points, "Targets")
        checks.check_sizes_t2(targets, points)

        if finufftkwargs is None:
            finufftkwargs = dict()
        else:
            finufftkwargs = finufftkwargs.copy()

        finufftkwargs.setdefault("isign", FinufftType2.ISIGN_DEFAULT)

        modeord = finufftkwargs.pop("modeord", FinufftType2.MODEORD_DEFAULT)

        points = torch.atleast_2d(points)
        ndim = points.shape[0]
        npoints = points.shape[1]
        if modeord:
            targets = batch_fftshift(targets, ndim)

        nufft_func = get_nufft_func(ndim, 2, points.device)
        batch_dims = targets.shape[:-ndim]
        shape = targets.shape[-ndim:]
        finufft_out = nufft_func(
            *points,
            targets.reshape(-1, *shape),
            **finufftkwargs,
        )
        finufft_out = finufft_out.reshape(*batch_dims, npoints)

        return finufft_out

    @staticmethod
    def vmap(  # type: ignore[override]
        info: Any,
        in_dims: Tuple[Optional[int], ...],
        points: torch.Tensor,
        targets: torch.Tensor,
        finufftkwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        batch_points, batch_targets, *_ = in_dims

        if batch_targets is not None:
            targets = targets.movedim(batch_targets, 0)

        if batch_points is not None:
            # need a for-loop here
            # potential opportunity for CUDA streams
            points = points.movedim(batch_points, 0)
            if batch_targets is not None:
                output = torch.stack(
                    [
                        FinufftType2.apply(
                            points[i],
                            targets[i],  # inner product
                            finufftkwargs,
                        )
                        for i in range(info.batch_size)
                    ],
                    dim=0,
                )
            else:
                output = torch.stack(
                    [
                        FinufftType2.apply(
                            points[i],
                            targets,
                            finufftkwargs,
                        )
                        for i in range(info.batch_size)
                    ],
                    dim=0,
                )
        else:
            output = FinufftType2.apply(points, targets, finufftkwargs)

        return output, 0

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, None],
        Union[torch.Tensor, None],
        None,
        None,
        None,
    ]:
        _i_sign = ctx.isign
        _mode_ordering = ctx.mode_ordering
        finufftkwargs = ctx.finufftkwargs

        points, targets = ctx.saved_tensors
        points = torch.atleast_2d(points)
        device = points.device
        ndim = points.shape[0]

        grad_points = None
        grad_targets = None

        if any(ctx.needs_input_grad):
            if _mode_ordering:
                # TODO this was also computed in forward
                targets = batch_fftshift(targets, ndim)

            batch_dims = targets.shape[:-ndim]
            shape = targets.shape[-ndim:]
            batched_targets = targets.reshape(-1, 1, *shape)
            nbatch = batched_targets.shape[0]
            batched_outputs = grad_output.reshape(nbatch, 1, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            # wrt. points
            nufft_func = get_nufft_func(ndim, 2, points.device)

            coord_ramps = coordinate_ramps(shape, device)

            ramped_targets = (coord_ramps * batched_targets * 1j * _i_sign).reshape(
                -1, *shape
            )

            backprop_ramp = (
                nufft_func(*points, ramped_targets, isign=_i_sign, **finufftkwargs)
                .conj()  # Why can't this `conj` be replaced with a flipped isign
                .reshape(nbatch, ndim, -1)
            )

            grad_points = (backprop_ramp * batched_outputs).real.sum(dim=0)

        if ctx.needs_input_grad[1]:
            # wrt. targets
            nufft_func = get_nufft_func(ndim, 1, points.device)

            grad_targets = nufft_func(
                *points,
                batched_outputs.squeeze(),
                shape,
                isign=-_i_sign,
                **finufftkwargs,
            ).reshape(*batch_dims, *shape)

            if _mode_ordering:
                grad_targets = batch_ifftshift(grad_targets, ndim)

        return (
            grad_points,
            grad_targets,
            None,
            None,
            None,
        )

def create_plan(
    type: int,
    dims: Union[int, Tuple[int, ...]],
    n_trans: int = 1,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.complex64,
    isign: int = 1,
    modeord: int = 0,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> FinufftPlan:
    
    return FinufftPlan(
        type=type,
        dims=dims,
        n_trans=n_trans,
        eps=eps,
        dtype=dtype,
        isign=isign,
        modeord=modeord,
        device=device,
        **kwargs,
    )

def finufft_type1_plan(
    plan: FinufftPlan,
    plan_adj: FinufftPlan,
    points: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return FinufftType1WithPlan.apply(plan, plan_adj, points, values)


def finufft_type2_plan(
    plan: FinufftPlan,
    plan_adj: FinufftPlan,
    points: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return FinufftType2WithPlan.apply(plan, plan_adj, points, targets)


def finufft_type1(
    points: torch.Tensor,
    values: torch.Tensor,
    output_shape: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    **finufftkwargs: Union[int, float],
) -> torch.Tensor:
    """
    Evaluates the Type 1 (nonuniform-to-uniform) NUFFT on the inputs.

    This is a wrapper around :func:`finufft.nufft1d1`, :func:`finufft.nufft2d1`, and
    :func:`finufft.nufft3d1` on CPU, and :func:`cufinufft.nufft1d1`,
    :func:`cufinufft.nufft2d1`, and :func:`cufinufft.nufft3d1` on GPU.

    Parameters
    ----------
    points : torch.Tensor
        DxN tensor of locations of the non-uniform points.
        Points should lie in the range ``[-pi, pi]``, values outside will be folded.
    values : torch.Tensor
        Complex-valued tensor of values at the non-uniform points.
        All dimensions except the final dimension are treated as batch
        dimensions. The final dimension must have size ``N``.
    output_shape : int | tuple(int, ...)
        Requested output shape of Fourier modes. Must be a tuple of length D or
        an integer (1D only).
    **finufftkwargs : int | float
        Additional keyword arguments are forwarded to the underlying
        FINUFFT functions. A few notable options are

        - ``eps``: precision requested (default: ``1e-6``)
        - ``modeord``: 0 for FINUFFT default, 1 for Pytorch default (default: ``1``)
        - ``isign``: Sign of the exponent in the Fourier transform (default: ``-1``)

    Returns
    -------
    torch.Tensor
        Tensor with shape ``*[batch], *output_shape`` containing the Fourier
        transform of the values.
    """
    res: torch.Tensor = FinufftType1.apply(points, values, output_shape, finufftkwargs)
    return res


def finufft_type2(
    points: torch.Tensor,
    targets: torch.Tensor,
    **finufftkwargs: Union[int, float],
) -> torch.Tensor:
    """
    Evaluates the Type 2 (uniform-to-nonuniform) NUFFT on the inputs.

    This is a wrapper around :func:`finufft.nufft1d2`, :func:`finufft.nufft2d2`, and
    :func:`finufft.nufft3d2` on CPU, and :func:`cufinufft.nufft1d2`,
    :func:`cufinufft.nufft2d2`, and :func:`cufinufft.nufft3d2` on GPU.

    Parameters
    ----------
    points : torch.Tensor
        DxN tensor of locations of the non-uniform points.
        Points should lie in the range ``[-pi, pi]``, values outside will be folded
    targets : torch.Tensor
        Complex-valued tensor of Fourier modes to evaluate at the points.
        The final D dimensions must contain the Fourier modes, and any
        preceding dimensions are treated as batch dimensions.
    **finufftkwargs : int | float
        Additional keyword arguments are forwarded to the underlying
        FINUFFT functions. A few notable options are

        - ``eps``: precision requested (default: ``1e-6``)
        - ``modeord``: 0 for FINUFFT default, 1 for Pytorch default (default: ``1``)
        - ``isign``: Sign of the exponent in the Fourier transform (default: ``-1``)

    Returns
    -------
    torch.Tensor
        A ``[batch]xDxN`` tensor of values at the non-uniform points.
    """
    res: torch.Tensor = FinufftType2.apply(points, targets, finufftkwargs)
    return res
