

import torch
import math

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
    m = torch.as_tensor(m, device=device, dtype=dtype)
    rain_thr = torch.as_tensor(rain_thr, device=device, dtype=dtype)
    snow_thr = torch.as_tensor(snow_thr, device=device, dtype=dtype)

    batch_dims, T = prcp.shape[:-1], prcp.shape[-1] 

    # init state
    if isinstance(snowpack_initial, torch.Tensor):
        S_prev = snowpack_initial.to(device=device, dtype=dtype)
    else:
        S_prev = torch.full(batch_dims, float(snowpack_initial), device=device, dtype=dtype)

    # output tensors
    rain = prcp.new_zeros(*batch_dims, T)
    snow = prcp.new_zeros(*batch_dims, T)
    snowmelt = prcp.new_zeros(*batch_dims, T)
    snowpack = prcp.new_zeros(*batch_dims, T)

    # loop over time
    for t in range(T):
        p = prcp[..., t]
        tn = tmin[..., t]

        # fractional snow/rain
        frac = (rain_thr - tn) / (rain_thr - snow_thr)
        frac = torch.clamp(frac, min=0.0, max=1.0)

        # snow, rain and melt coefficient
        snow_t = frac * p
        rain_t = (1.0 - frac) * p
        melt_coeff = m * frac

        # update snowpack
        total_snow = S_prev + snow_t
        melt_t = melt_coeff * total_snow
        pack_t = torch.clamp(total_snow - melt_t, min=0.0)

        # update outputs
        rain[..., t] = rain_t
        snow[..., t] = snow_t
        snowmelt[..., t] = melt_t
        snowpack[..., t] = pack_t

        # update state for next iteration
        S_prev = pack_t

    return rain, snow, snowmelt, snowpack



def pet_hargreaves_torch(
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        day_of_year: torch.Tensor,
        latitude: torch.Tensor
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
    pi = math.pi
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
    w = torch.acos(cos_w)  # twilight angle in radians

    et_rad = (24.0 * 60.0 / pi) * GSC * dr * (
        (w * torch.sin(phi) * torch.sin(delta)) +
        (torch.cos(phi) * torch.cos(delta) * torch.sin(w))
    )

    tmean = (tmin + tmax) / 2.0
    tdiff = torch.clamp(tmax - tmin, min=0.0)
    pet = 0.0023 * (tmean + 17.8) * torch.sqrt(tdiff) * 0.408 * et_rad 
    
    return pet


def abcd_torch(
        total_prcp: torch.Tensor,
        pet:        torch.Tensor,
        abcd_pars,  # (a, b, c, d) floats or Tensor[B?]
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
    E = total_prcp.new_zeros(*batch_dims, T)   # evaporation

    # loop over time
    for t in range(T):
        P = total_prcp[..., t]
        PET = pet[..., t]

        # Water Available for evaporation
        WA = P + uz_prev

        # partitioning into evaporation-eligible storage EO
        temp = (WA + b) / (2.0 *a)
        disc = temp*temp - (WA * b) / a
        disc = torch.clamp(disc, min=0.0) # ensure non-negative discriminant
        EO = temp - torch.sqrt(disc)
        EO = torch.clamp(EO, min=0.0, max=WA) # [0, WA]

        # evaporation
        E_t = EO * (1.0 - torch.exp(-PET / b))
        E_t = torch.clamp(E_t, min=0.0, max=EO)  # ensure non-negative

        # direct runoff vs. recharge
        Qd_t = (1.0 - c) * (WA - EO)
        R_t = c * (WA - EO)

        # update storages
        uz_curr = uz_prev + P - E_t - Qd_t
        lz_curr = (lz_prev + R_t) * (1.0 - d)

        # output direct runoff, baseflow, and storages
        Qd[..., t] = Qd_t  # direct runoff
        Qb[..., t] = lz_prev * d  # baseflow
        uz[..., t] = uz_curr  # upper zone storage
        lz[..., t] = lz_curr  # lower zone storage
        E[..., t] = E_t  # evaporation

        # update state for next iteration
        uz_prev = uz_curr
        lz_prev = lz_curr

    return Qd, Qb, uz, lz, E


def generate_HRU_UH_torch(params, KE=12):

    """
    Generate HRU unit hydrographs in PyTorch.

    Args:
        params : tuple of (shape, rate) – gamma distribution parameters.
                 Each can be a Python float or a Tensor of shape (B?,).
        KE     : int – number of days to span the HRU response.

    Returns:
        UH_direct : Tensor[B?, KE] or (KE,)  – direct runoff UH.
        UH_base   : Tensor[B?, KE] or (KE,)  – delta UH for baseflow.
    """

    # unpack & cast to tensors on same device/dtype
    shape, rate = params
    device = None
    dtype = None

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
    M = 1000 * KE
    x = torch.linspace(0.0, 24.0 * KE, M, device=device, dtype=dtype)
    dx = x[1] - x[0]  # time step in hours

    # Prepare for broadcasting: alpha and scale -> (..., 1)
    alpha_exp = alpha.unsqueeze(-1)  # shape (..., 1)
    scale_exp = scale.unsqueeze(-1)  # shape (..., 1)

    # Small time step for numerical stability
    eps = 1e-6

    # Compute the unit hydrograph using the gamma PDF
    num = (x + eps).pow(alpha_exp - 1.0) * torch.exp(-x / scale_exp)
    denom = (scale_exp.pow(alpha_exp) * torch.exp(torch.lgamma(alpha_exp)))
    pdf = num / denom # shape (..., M)

    # Integrate into daily bins by summing over 1000-point blocks
    # Discard the final extra points if M is not a multiple of 1000, reshape & sum:
    pdf_trunc = pdf[..., :-1]
    pdf_blocks = pdf_trunc.view(*pdf_trunc.shape[:-1], KE, 1000)
    UH_direct = pdf_blocks.sum(dim=-1) * dx  # shape (..., KE)

    # Create baseflow UH: delta at t=0
    out_shape = UH_direct.shape
    UH_base = torch.zeros(out_shape, device=device, dtype=dtype)
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
    flowlen = torch.tensor(flowlen, device=device, dtype=torch.float32)
    velo = torch.tensor(velo, device=device, dtype=torch.float32)
    diff = torch.tensor(diff, device=device, dtype=torch.float32)
    
    # Build fine-scale time grid (seconds)
    t_grid = torch.arange(1, LE+1, dtype.float32, device=device) * DT

    # Evaluate Green's function H(t)
    pot = ((velo * t_grid - flowlen) **2) / (4.0 * diff * t_grid)
    H = torch.where(
        pot <= 69.0,
        
    )