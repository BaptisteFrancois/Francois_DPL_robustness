
import torch
import torch.nn.functional as F
from typing import Tuple, Union





def snow_melt_torch(
        prcp: torch.Tensor,
        tmin: torch.Tensor,
        snow_params: tuple,
        snowpack_initial: float = 0.0,
):
    
    """
    Snow accumulation & melt in PyTorch.

    Args:
        prcp             (Tensor[B?, T]) daily precipitation [mm]
        tmin             (Tensor[B?, T]) daily minimum temperature [°C]
        snow_params      (m, rain_thresh, snow_thresh)
        snowpack_initial float or Tensor[B?] initial snowpack

    Returns:
        rain      Tensor[B?, T]
        snow      Tensor[B?, T]
        snowmelt  Tensor[B?, T]
        snowpack  Tensor[B?, T]
    """


    # unpack & cast to tensors on same device/dtype
    m, rain_thr, snow_thr = snow_params
    device, dtype = prcp.device, prcp.dtype
    
    m        = torch.as_tensor(m, device=device, dtype=dtype)
    rain_thr = torch.as_tensor(rain_thr, device=device, dtype=dtype)
    snow_thr = torch.as_tensor(snow_thr, device=device, dtype=dtype)

    batch_dims, T = prcp.shape[:-1], prcp.shape[-1] 

    # init state
    if isinstance(snowpack_initial, torch.Tensor):
        S_prev = snowpack_initial.to(device=device, dtype=dtype)
    else:
        S_prev = torch.full(batch_dims, float(snowpack_initial), device=device, dtype=dtype)

    # output tensors
    rain     = prcp.new_zeros(*batch_dims, T)
    snow     = prcp.new_zeros(*batch_dims, T)
    snowmelt = prcp.new_zeros(*batch_dims, T)
    snowpack = prcp.new_zeros(*batch_dims, T)

    # loop over time
    for t in range(T):
        p  = prcp[..., t]
        tn = tmin[..., t]

        # fractional snow/rain
        frac_snow = torch.where(
            tn < snow_thr,    # cold -> all snow
            1.0,
            torch.where(
                tn <= rain_thr, # mild -> mix of snow and rain
                (rain_thr - tn) / (rain_thr - snow_thr),
                0.0            # warm -> all rain
            )
        )
        frac_snow = torch.clamp(frac_snow, min=0.0, max=1.0)

        # snowmelt_coefficient
        melt_coeff = torch.where(
            tn < snow_thr,  # too cold -> no melt
            0.0,
            torch.where(
                tn <= rain_thr,  # mild -> some melt
                m * (rain_thr - tn) / (rain_thr - snow_thr), 
                m            # warm -> full melt
            )
        )
        melt_coeff = torch.clamp(melt_coeff, min=0.0, max=1.0)

        # snow, rain and melt coefficient
        snow_t     = frac_snow * p
        rain_t     = (1.0 - frac_snow) * p

        # update snowpack
        total_snow = S_prev + snow_t
        melt_t     = melt_coeff * total_snow
        pack_t     = torch.clamp(total_snow - melt_t, min=0.0)

        # update outputs
        rain[..., t]     = rain_t
        snow[..., t]     = snow_t
        snowmelt[..., t] = melt_t
        snowpack[..., t] = pack_t

        # update state for next iteration
        S_prev = pack_t

    return rain, snow, snowmelt, snowpack



def pet_hargreaves_torch(
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        day_of_year: torch.Tensor,
        latitude: Union[float, torch.Tensor]
):
    
    """
    Daily PET via Hargreaves in PyTorch.

    Args:
        tmin         Tensor[B?, T] daily min temp [°C]
        tmax         Tensor[B?, T] daily max temp [°C]
        day_of_year  Tensor[B?, T] integers 1–365
        latitude     float or Tensor[B?] latitude in degrees

    Returns:
        pet Tensor[B?, T] [mm/day]
    """

    device, dtype = tmin.device, tmin.dtype

    # constants
    GSC = 0.0820  # solar constant [MJ m-2 min-1]
    pi = torch.pi
    batch_dims = tmin.shape[:-1]

    # unpack & cast to tensors on same device/dtype
    if isinstance(latitude, torch.Tensor):
        lat = latitude.to(device=device, dtype=dtype)
    else:
        lat = torch.full(batch_dims, float(latitude), device=device, dtype=dtype)

    # If lat is per‐basin (B,), we need (B,1) to broadcast over time
    # so that ops like lat * day_of_year broadcast correctly
    lat = lat.unsqueeze(-1)      # now shape (..., 1,)

    # Compute φ in radians
    # torch.radians() exists in newer PyTorch; else multiply by π/180
    phi = lat * (pi / 180.0)

    # Compute Δ (day length) in radians
    delta = 0.409 * torch.sin(2 * pi * (day_of_year - 81) / 365.0)

    # Compute dr (inverse relative distance Earth-Sun)
    dr = 1.0 + 0.033 * torch.cos(2 * pi * (day_of_year) / 365.0) 

    cos_w = -torch.tan(phi) * torch.tan(delta)
    cos_w = torch.clamp(cos_w, min=-1.0, max=1.0)
    w     = torch.acos(cos_w)  # twilight angle in radians

    et_rad = (24.0 * 60.0 / pi) * GSC * dr * (
        (w * torch.sin(phi) * torch.sin(delta)) +
        (torch.cos(phi) * torch.cos(delta) * torch.sin(w))
    )

    tmean = (tmin + tmax) / 2.0
    tdiff = torch.clamp(tmax - tmin, min=0.0)
    pet   = 0.0023 * (tmean + 17.8) * torch.sqrt(tdiff) * 0.408 * et_rad 
    
    return pet


def abcd_torch(
        total_prcp: torch.Tensor,
        pet:        torch.Tensor,
        abcd_pars:  Tuple[Union[float, torch.Tensor],
                          Union[float, torch.Tensor],
                          Union[float, torch.Tensor],
                          Union[float, torch.Tensor]],
        uz_initial: float = 0.0,
        lz_initial: float = 0.0
):
    
    """
    ABCD water-balance in PyTorch.

    Args:
        total_prcp  Tensor[B?, T]  precipitation [mm/day]
        pet         Tensor[B?, T]  potential ET   [mm/day]
        abcd_pars   4-tuple        (a, b, c, d)
        uz_initial  float or Tensor[B?]  initial upper zone storage [mm]
        lz_initial  float or Tensor[B?]  initial lower zone storage [mm]

    Returns:
        Qd    Tensor[B?, T]  direct runoff
        Qb    Tensor[B?, T]  baseflow
        uz    Tensor[B?, T]  upper‐zone storage
        lz    Tensor[B?, T]  lower‐zone storage
        E     Tensor[B?, T]  evaporation
    """

    device, dtype = total_prcp.device, total_prcp.dtype
    batch_dims, T = total_prcp.shape[:-1], total_prcp.shape[-1]

    # unpack & cast to tensors on same device/dtype
    a, b, c, d = abcd_pars
    a = torch.as_tensor(a, device=device, dtype=dtype)
    b = torch.as_tensor(b, device=device, dtype=dtype)
    c = torch.as_tensor(c, device=device, dtype=dtype)
    d = torch.as_tensor(d, device=device, dtype=dtype)

    # enforce a > 1e-6 for numerical stability
    a = torch.clamp(a, min=1e-6)

    # initial states (per-batch or scalar)
    if torch.is_tensor(uz_initial):
        uz_prev = uz_initial.to(device=device, dtype=dtype)
    else:
        uz_prev = torch.full(batch_dims, 
                             float(uz_initial),
                             device=device, dtype=dtype)

    if torch.is_tensor(lz_initial):
        lz_prev = lz_initial.to(device=device, dtype=dtype)
    else:
        lz_prev = torch.full(batch_dims, 
                             float(lz_initial), 
                             device=device, dtype=dtype)
        
    # output tensors
    Qd = total_prcp.new_zeros(*batch_dims, T)  # direct runoff
    Qb = total_prcp.new_zeros(*batch_dims, T)  # baseflow
    uz = total_prcp.new_zeros(*batch_dims, T)  # upper zone storage
    lz = total_prcp.new_zeros(*batch_dims, T)  # lower zone storage
    E  = total_prcp.new_zeros(*batch_dims, T)  # evaporation

    # loop over time
    for t in range(T):
        P   = total_prcp[..., t]
        PET = pet[..., t]

        # Water Available for evaporation
        WA = P + uz_prev

        # partitioning into evaporation-eligible storage EO
        temp = (WA + b) / (2.0 *a)
        disc = temp*temp - (WA * b) / a
        disc = torch.where(disc < 0, torch.zeros_like(disc), disc)
        # EO
        EO   = temp - torch.sqrt(disc)
        EO   = torch.where(EO < 0,  torch.zeros_like(EO),
                torch.where(EO > WA, WA, EO))
        # E_t
        rawE = EO * (1 - torch.exp(-PET / b))
        E_t  = torch.where(rawE < 0,      torch.zeros_like(rawE),
                torch.where(rawE > EO,    EO, rawE))

        # direct runoff vs. recharge
        Qd_t = (1.0 - c) * (WA - EO)
        R_t  = c * (WA - EO)

        # update storages
        uz_curr = uz_prev + P - E_t - R_t
        lz_curr = (lz_prev + R_t) / (1.0 + d)

        # output direct runoff, baseflow, and storages
        Qd[..., t] = Qd_t  # direct runoff
        Qb[..., t] = lz_curr * d  # baseflow
        uz[..., t] = uz_curr  # upper zone storage
        lz[..., t] = lz_curr  # lower zone storage
        E[..., t]  = E_t  # evaporation

        # update state for next iteration
        uz_prev = uz_curr
        lz_prev = lz_curr

    return Qd, Qb, uz, lz, E


def generate_HRU_UH_torch(params, KE=12, device: Union[None, str, torch.device] = None):

    """
    Generate HRU unit hydrographs in PyTorch.

    Args:
        params : tuple of (shape, rate) - gamma distribution parameters.
                 Each can be a Python float or a Tensor of shape (B?,).
        KE     : int - number of days to span the HRU response.

    Returns:
        UH_direct : Tensor[B?, KE] or (KE,)  - direct runoff UH.
        UH_base   : Tensor[B?, KE] or (KE,)  - delta UH for baseflow.
    """

    # Setup device and convert scalar parameters to tensors
    device = torch.device(device or 'cpu')

    # unpack & cast to tensors on same device/dtype
    shape, rate = params

    # Turn shape and rate into tensors on the same device/dtype
    if torch.is_tensor(shape):
        device, dtype = shape.device, shape.dtype
        alpha = shape.clamp(min=1e-6)
    else:
        alpha = torch.tensor(shape, dtype=torch.get_default_dtype())
        device, dtype = alpha.device, alpha.dtype

    if torch.is_tensor(rate):
        rate = rate.clamp(min=1e-6).to(device=device, dtype=dtype)
    else:
        rate = torch.tensor(rate, dtype=dtype, device=device).clamp(min=1e-6)

    # Compute scale = 1 / rate
    scale = 1.0 / rate

    # Build time grid: M = 1000*KE points from 0...24*KE hours
    M = 1000 * KE + 1  # 1000 points per hour, plus one for zero
    x = torch.linspace(0.0, 24.0 * KE, M, device=device, dtype=dtype)
    dx = x[1] - x[0]  # time step in hours

    # Prepare for broadcasting: alpha and scale -> (..., 1)
    alpha_exp = alpha.unsqueeze(-1)  # shape (..., 1)
    scale_exp = scale.unsqueeze(-1)  # shape (..., 1)

    # Small time step for numerical stability
    eps = 1e-12

    # Compute the unit hydrograph using the gamma PDF
    num   = (x + eps).pow(alpha_exp - 1.0) * torch.exp(-x / scale_exp)
    denom = (scale_exp.pow(alpha_exp) * torch.exp(torch.lgamma(alpha_exp)))
    pdf   = num / denom # shape (..., M)

    # Integrate into daily bins by summing over 1000-point blocks
    pdf_trunc  = pdf[..., : KE * 1000]  # truncate to KE days
    prefix = pdf_trunc.shape[:-1] # unpack batch dimensions
    pdf_blocks = pdf_trunc.view(*prefix, KE, 1000) # shape (..., KE, 1000)
    UH_direct  = pdf_blocks.sum(dim=-1) * dx  # shape (..., KE)

    # Create baseflow UH: delta at t=0
    out_shape = UH_direct.shape
    UH_base   = torch.zeros(out_shape, device=device, dtype=dtype)
    UH_base[...,0] = 1.0

    return UH_direct, UH_base


def generate_channel_UH_torch(
        flowlen, velo, diff,
        UH_DAY: int = 96,
        DT: int     = 3600,
        LE: int     = 2400,
        device=None,
):
    
    """
    PyTorch version of the Green's‐function channel unit‐hydrograph.

    Returns:
        UH_river : (UH_DAY,) torch.Tensor of daily unit hydrograph
    """

    # Setup device and convert scalar parameters to tensors
    device = torch.device(device or 'cpu')

    flowlen = torch.as_tensor(flowlen, device=device, dtype=torch.float32)
    velo    = torch.tensor(velo, device=device, dtype=torch.float32)
    diff    = torch.tensor(diff, device=device, dtype=torch.float32)
    
    # Build fine-scale time grid (seconds)
    t_grid = torch.arange(1, LE+1, dtype=torch.float32, device=device) * DT

    # Evaluate Green's function H(t)
    pot = ((velo * t_grid - flowlen) **2) / (4.0 * diff * t_grid)
    H = torch.where(
        pot <= 69.0,
        flowlen / (2 * t_grid * torch.sqrt(torch.pi * t_grid * diff)) * torch.exp(-pot),
        torch.zeros_like(t_grid)
    )

    # Normalize to unit-area UH (UHM)
    H_sum = H.sum() # sum stays a tensor

    # build the 'zero-sum' fallback UH: all zeros except a 1 at t=0
    fallback = torch.zeros_like(H)
    fallback[0] = 1.0

    # create a boolean mask tensor: true if sum>0, false otherwise
    mask = H_sum > 0.0

    # use torch.where to pick the normalized H or the fallback, per-element
    UHM = torch.where(
        mask,       # scalar mask will broadcast to H's shape
        H / H_sum,  # if mask is true, normalize H
        fallback    # if mask is false, use the fallback UH
    )

    # Build hourly finite response FR(t)
    TMAX = UH_DAY * 24  # total hours in UH_DAY days
    FR = torch.zeros(TMAX+24, device=device, dtype=torch.float32)
    FR[:24] = 1.0 / 24.0  # first 24 hours are flat at 1/24

    # Convolve FR with UHM (first 24 h of UHM)
    for t in range(24, TMAX):
        window = FR[t - 24:t].flip(0) # reverse the last 24 hours
        FR[t] = torch.dot(window, UHM[:24]) # inner product

    # Downsample FR to daily sums -> UH_river
    # reshape (UH_DAY * 24), then sum each day
    FR_days = FR[:TMAX].view(UH_DAY, 24)
    UH_river = FR_days.sum(dim=1)  # sum over the 24

    return UH_river


def routing_lohman_torch(
        inflow_direct: torch.Tensor, 
        inflow_base: torch.Tensor,
        UH_HRU_direct: torch.Tensor,
        UH_HRU_base: torch.Tensor,
        UH_river: torch.Tensor
):
    
    """
    Route flows for a single basin using precomputed unit hydrographs.

    All inputs are 1-D torch.Tensors:
      - inflow_direct, inflow_base: length n_time
      - UH_HRU_direct, UH_HRU_base: length n_hru
      - UH_river: length n_uh

    Returns two 1-D tensors of length n_time:
      - directflow, baseflow
    """

    # Ensure common device and dtype
    device = inflow_direct.device
    dtype = inflow_direct.dtype

    inflow_direct = inflow_direct.to(device, dtype).view(1,1,-1)
    inflow_base   = inflow_base.  to(device, dtype).view(1,1,-1)
    UH_HRU_direct = UH_HRU_direct.to(device, dtype).view(1,1,-1)
    UH_HRU_base   = UH_HRU_base.  to(device, dtype).view(1,1,-1)
    UH_river      = UH_river.     to(device, dtype).view(1,1,-1)

    n_hru  = UH_HRU_direct.size(-1)
    n_uh   = UH_river.size(-1)
    UH_len = n_hru + n_uh - 1

    # Build combined UH via 1D convolution
    UH_river_rev = UH_river.flip(-1).view(1,1,-1)
    UH_direct = F.conv1d(UH_HRU_direct, UH_river_rev, padding = n_uh-1).view(UH_len)
    UH_base   = F.conv1d(UH_HRU_base, UH_river_rev, padding = n_uh-1).view(UH_len)

    # Normalize th unit-area (keep graph: use where)
    sum_d = UH_direct.sum()
    sum_b = UH_base.sum()

    # default fallback: Dirac at t=0    
    fallback = torch.zeros_like(UH_direct)
    fallback[0] = 1.0

    UH_direct = torch.where(
        sum_d > 0.0,
        UH_direct / sum_d,  # normalize if sum > 0
        fallback            # else use fallback UH
    )

    UH_base = torch.where(
        sum_b > 0.0,
        UH_base / sum_b,  # normalize if sum > 0
        fallback          # else use fallback UH
    )

    # Route inflow via discrete convolution
    pad = UH_len -1
    in_d_pad = F.pad(inflow_direct, (pad, 0), mode='constant', value=0.0)
    in_b_pad = F.pad(inflow_base, (pad, 0), mode='constant', value=0.0)

    # reverse UH for causal convolution
    k_direct = UH_direct.flip(0).view(1, 1, -1)
    k_base   = UH_base.flip(0).view(1, 1, -1)

    # Route inflow via discrete convolution
    directflow = F.conv1d(in_d_pad, k_direct).view(-1)
    baseflow   = F.conv1d(in_b_pad, k_base).view(-1)

    return directflow, baseflow