# -*- coding: utf-8 -*-
"""
Normalization Model of Attention Implementation

Description:
    This script implements a dynamic normalization model of attention based on 
    Reynolds & Heeger (2009). It simulates the interaction between voluntary 
    (endogenous) and involuntary (exogenous) attention across multiple layers 
    (sensory, attention, decision). The code includes procedures for model 
    simulation, parameter fitting using BADS (Bayesian Adaptive Direct Search), 
    and visualization of d-prime results.

Date: 2025-04-11
Map:
    - Class 'parameters': Defines model constants and time-courses.
    - Function 'n_model': Core layer-by-layer processing loop.
    - Function 'run_model': Execution of experimental conditions.
    - Main Execution: Data loading, parameter optimization, and plotting.
"""

import numpy as np
from scipy import stats
import sys
import math
import matplotlib.pyplot as plt
import os, time, datetime, pickle
import pandas as pd 
from pybads import BADS  # Requires: pip install pybads

# %% Parameter Configuration
class parameters():
    def __init__(self, opt=None):
        self.opt = opt if opt is not None else {}
        # Initialize parameters with defaults if opt is empty
        if not self.opt:
            self.set_parameters()

    def set_parameters(self):
        # --- Temporal Parameters ---
        self.dt = 2                    # Time step (ms)
        self.T = 2.1 * 1000            # Total duration (ms)
        self.nt = int(self.T/self.dt + 1) # Number of time points
        self.tlist = np.arange(0, self.T+self.dt, self.dt)
        self.soa = 300                 # Stimulus Onset Asynchrony (time between cue/validity and target)

        # --- Spatial and Feature Space ---
        self.x = 0
        self.nx = np.size(self.x)
        self.ntheta = 12               # Number of orientation channels/receptive fields

        # --- Nonlinearity Exponent ---
        self.p = 1.5                   # Exponent for power-law nonlinearity

        # --- Sensory Layer Constants (Time constant & Semisaturation) ---
        # Layer 1 (S1)
        self.tau1 = 52
        self.sigma1 = 1.4

        # Layer 2 (S2)
        self.tau2 = 100
        self.sigma2 = 0.1

        # Layer 3 (S3) - unused in current logic but reserved
        self.tau3 = 2
        self.sigma3 = 0.3

        # --- Attention Parameters ---
        self.aA = 10                   # Attention amplitude (general)
        self.pn = 0.5                  # Exponent scaling for rhythm
        
        # Time constants for Attention Layers
        self.tauAI = 2                 # Involuntary (Exogenous) Attention
        self.tauAV = 50                # Voluntary (Endogenous) Attention

        # Voluntary Attention (AV)
        self.aAV = 40                  # Amplitude
        self.gamPSh = 2.2              # Gamma Positive Shape
        self.gamPSc = 0.023            # Gamma Positive Scale

        # Involuntary Attention (AI)
        self.aAI = 8.5                 # Amplitude
        self.aANeg = 0                 # Negative dip amplitude
        self.gamNSh = None             # Gamma Negative Shape
        self.gamNSc = None             # Gamma Negative Scale
        
        self.sigmaA = 20               # Normalization constant for attention layers

        # Temporal filter for involuntary attention (Gamma distribution based)
        self.h0 = makePrefilter(np.arange(0, 0.8+self.dt/1000, self.dt/1000),
                                posShape=self.gamPSh,
                                posScale=self.gamPSc,
                                negShape=self.gamNSh,
                                negScale=self.gamNSc,
                                ampNeg=self.aANeg)
        self.h = np.tile(self.h0, (self.ntheta, 1)) # Replicate across orientations

        self.attn = 2                  # Base attention scalar

        # --- Stimulus Timing ---
        self.stimOnset = 300
        self.stimDur = 30

        # --- Task Timing ---
        self.AVOnset = -34
        self.AVDur = 124
        self.tR = 918                  # Voluntary attention recovery time (based on Denison et al.)

        # --- Attention Weights ---
        self.distributeVoluntary = 1   # Flag: 1 = limit voluntary attention capacity
        self.wv = 0.2                  # Weight for valid cue
        self.winv = 0.8                # Weight for invalid cue
        self.AVWeights = [self.wv, self.winv]
        self.AVNeutralT1Weight = 0.5
        self.AVProp = 1
        self.disturbuteVoluntary = 1   # Redundant flag check

        # --- Decision Layer ---
        self.sigmaD = 0.7
        self.tauD = 100000             # Long integration window for decision

        # --- Scaling Factors (for fitting d-prime) ---
        self.scaling1 = 1e5
        self.scaling2 = 1e5

        # --- System Info ---
        self.version = sys.version


# %% Helper Functions

def decodeEvidence(response, templateResponse, decisionBias=0):
    """
    Decodes the decision evidence from the neural response.
    
    Calculates the projection of the response onto the difference between 
    two templates (e.g., CW vs CCW).
    
    Parameters:
    - response: Activity of the sensory layer (S2).
    - templateResponse: 2-row array representing ideal responses for the two choices.
    - decisionBias: Bias term (default 0).

    Returns:
    - decisionEvidence: Scalar value representing the evidence (d-prime proxy) at this time point.
      Negative values support Template 1, positive values support Template 2.
    """
    # Calculate weight vector (difference between two templates)
    w = templateResponse[1, :] - templateResponse[0, :]

    # Project response onto the weight vector
    decisionEvidence = np.dot(w, response.T) - decisionBias

    return np.array(decisionEvidence, dtype='double')


def distributeAttention(weight):
    """
    Allocates limited voluntary attention resources between two potential targets.

    Parameters:
    - weight: Proportion of attention allocated to Target 1.

    Returns:
    - attn: Array [attn_T1, attn_T2]. Capped at a total capacity.
    """
    totalAttn = 1
    attn = np.array([0., 0.])

    attn[0] = totalAttn * weight
    attn[1] = totalAttn * (1 - weight)

    # Reallocate extra attention if one channel exceeds capacity
    if attn[0] > 1:
        extra = attn[0] - 1
        attn[0] = 1
        attn[1] = attn[1] + extra
    elif attn[1] > 1:
        extra = attn[1] - 1
        attn[1] = 1
        attn[0] = attn[0] + extra

    return attn


def halfExp(base, n=1):
    """
    Half-wave rectification followed by exponentiation.
    Functions as the neuronal nonlinearity (ReLU-like).

    Output = max(0, base) ^ n
    """
    x = (np.maximum(0, base))**n
    x.astype('double')
    return x


def initTimeSeries(self): 
    """Initialize all time-series arrays for the simulation."""
    # Sensory layer 1
    self.d1 = np.zeros([self.ntheta, self.nt])  # Drive
    self.s1 = np.zeros([self.ntheta, self.nt])  # Suppressive drive (pool)
    self.f1 = np.zeros([self.ntheta, self.nt])  # Normalization factor
    self.r1 = np.zeros([self.ntheta, self.nt])  # Response (firing rate)

    # Sensory layer 2
    self.d2 = np.zeros([self.ntheta, self.nt])
    self.s2 = np.zeros([self.ntheta, self.nt])
    self.f2 = np.zeros([self.ntheta, self.nt])
    self.r2 = np.zeros([self.ntheta, self.nt])

    # Sensory layer 3 (Placeholder)
    self.d3 = np.zeros([self.ntheta, self.nt])
    self.s3 = np.zeros([self.ntheta, self.nt])
    self.f3 = np.zeros([self.ntheta, self.nt])
    self.r3 = np.zeros([self.ntheta, self.nt])

    # Decision Layer
    self.dd = np.zeros([self.nstim, self.nt])
    self.sd = np.zeros([self.nstim, self.nt])
    self.fd = np.zeros([self.nstim, self.nt])
    self.rd = np.zeros([self.nstim, self.nt])

    # Voluntary Attention Layer (Endogenous)
    self.dav = np.zeros([self.ntheta, self.nt])
    self.sav = np.zeros([self.ntheta, self.nt])
    self.fav = np.zeros([self.ntheta, self.nt])
    self.rav = np.zeros([self.ntheta, self.nt])
    self.task = np.zeros([self.ntheta, self.nt])  # Top-down control signal

    # Involuntary Attention Layer (Exogenous)
    self.dai = np.zeros([self.ntheta, self.nt])
    self.sai = np.zeros([self.ntheta, self.nt])
    self.fai = np.zeros([self.ntheta, self.nt])
    self.rai = np.zeros([self.ntheta, self.nt])

    return self


def makeGamma(space, center=None, shape=None, scale=None, height=None):
    """
    Generates a Gamma distribution function, used for temporal filters.
    
    Parameters:
    - space: Domain values (e.g., time points).
    - shape: Shape parameter (k).
    - scale: Scale parameter (theta).
    - center: Peak position adjustment.
    - height: Peak amplitude scaling.
    """
    if shape is None: shape = 12
    if scale is None: scale = 12

    g0 = stats.gamma.pdf(space, a=shape, scale=scale)

    if center is not None:
        peak = np.where(g0 == g0.max())[0][0]
        shift = center - peak
        # Shift the distribution to align the peak with 'center'
        if shift > 0:
            g = np.concatenate([np.zeros(shift-1, dtype='double'), g0[:-shift+1]])
        elif shift < 0:
            g = np.concatenate([g0[np.abs(shift)-1:], np.zeros(np.abs(shift)-1, dtype='double')])
    else:
        g = g0

    if height is not None:
        if height == 0:
            g = np.zeros_like(space)
        else:
            g = g/np.max(g) * height

    return g


def rfResponse(theta, nRF=12):
    """
    Calculates the population response of orientation-selective neurons.

    Parameters:
    - theta: Stimulus orientation (radians).
    - nRF: Number of receptive fields / channels.

    Returns:
    - response: Array of weights for each RF channel.
    """
    m = 2 * nRF - 1 # Tuning width exponent
    response = []
    for iRF in np.arange(1, nRF+1, 1):
        response.append(np.abs(np.cos(theta + iRF*np.pi/nRF)**m))
    return np.array(response, dtype='double')


def makePrefilter(x, posShape, posScale, negShape, negScale, ampNeg):
    """
    Creates a difference-of-gammas filter for involuntary attention dynamics.
    Combines a positive gamma (excitation) and a negative gamma (inhibition).
    """
    hPos = makeGamma(x, None, posShape, posScale, 1)
    hNeg = makeGamma(x, None, negShape, negScale, ampNeg)
    h = hPos - hNeg
    return h


def prefilter(x, w, n, dt, idx):
    """
    Applies the temporal prefilter to the input drive for involuntary attention.
    
    Parameters:
    - x: Input time series.
    - w: Filter weights (the impulse response).
    - idx: Current time index.
    """
    # Phase weights (on/off transients)
    phw = np.array([[1], [-1]])

    # Extract relevant history from x
    if idx > np.shape(w)[1]-1:
        start_idx = idx - np.shape(w)[1]
        end_idx = idx
        y = x[:, start_idx:end_idx]
    elif idx <= np.shape(w)[1]-1:
        y = np.full(w.shape, np.nan)
        y[:, -idx:] = x[:, 0:idx]

    # Convolve
    inp = []
    for iPh in [0, 1]:
        inp0 = y * np.fliplr(w * phw[iPh])
        end = inp0.shape[1]-1
        # Integrate
        inp1 = np.sum(inp0[:, np.maximum(end - idx + 2, 1):end], axis=1) * dt
        inp.append(halfExp(inp1, n))

    inp = np.array(inp)
    z = np.dot(inp.T, phw)
    return z


def setStim(p):
    """
    Generates the stimulus time course matrix.
    
    Populates p.stim with 1s where stimulus is present based on 
    stimulus onset, duration, SOA, and orientation sequence.
    """
    stimStart = p.stimOnset
    stimEnd = p.stimOnset + p.stimDur

    # Create time series: Rows = orientations, Columns = time points
    timeSeries = np.zeros((p.norient, p.nt))
    
    # Set Stimulus 1 (T1)
    idx_t1 = (np.unique(np.round((np.arange(stimStart, stimEnd+p.dt, p.dt)/p.dt)))-1).astype('int')
    timeSeries[p.stimseq[0], idx_t1] = 1
    
    # Set Stimulus 2 (T2) - offset by SOA
    idx_t2 = (np.unique(np.round((np.arange(stimStart, stimEnd+p.dt, p.dt)+p.soa)/p.dt))-1).astype('int')
    timeSeries[p.stimseq[1], idx_t2] = 1

    p.stim = timeSeries
    return p


def setDecisionWindows(p):
    """
    Defines the temporal window over which evidence is accumulated.
    For T1: From T1 onset until T2 onset (or fixed duration).
    """
    condname = p.condname
    if 'valid' == condname:
        desStart = p.stimOnset + p.soa
    elif 'invalid' == condname:
        desStart = p.stimOnset

    decisionOnsets = np.array([desStart])
    desicionDur = 800 # Duration of integration

    p.decisionWindows = np.zeros([p.nt])
    temp = np.arange(decisionOnsets[0]/p.dt-1,
                     (decisionOnsets[0]+desicionDur)/p.dt-1)

    idx = np.round(temp).astype('int')
    p.decisionWindows[idx] = 1
    return p


def setTask(p):
    """
    Configures the voluntary attention control signal based on cue validity.
    
    Modulates attention weights based on rhythm conditions using a log function.
    """
    condname = p.condname
    iRhythms = p.iRhythms
    
    # Logarithmic scaling of attention weights based on rhythm
    w = p.AVWeights * math.log(np.power(iRhythms+1, p.attn, dtype='float64'), p.pn)

    timeSeries = np.zeros([p.ntheta, p.nt, 2])
    
    # Set attention weights for valid vs invalid conditions
    if 'valid' == condname:
        attStart = p.stimOnset + p.AVOnset + p.soa
        attEnd = p.stimOnset + p.AVOnset + p.AVDur + p.soa
        attWeights = w
        idx = np.unique(np.round(np.arange(attStart, attEnd + p.dt, p.dt, dtype='int') / p.dt)).astype('int')
        timeSeries[:, idx-1] = attWeights
        
    elif 'invalid' == condname:
        attStart = p.stimOnset + p.AVOnset
        attEnd = p.stimOnset + p.AVOnset + p.AVDur
        attWeights = w
        idx = np.unique(np.round(np.arange(attStart, attEnd + p.dt, p.dt, dtype='int') / p.dt)).astype('int')
        timeSeries[:, idx-1] = attWeights
    else:
        raise ValueError('Attention condition not recognized')

    # Max over features (assuming spatial attention spreads)
    timeSeries = np.max(timeSeries, axis=2)
    p.task = timeSeries
    return p


# %% Core Normalization Functions

def n_core(d, sigma, p, r_prev, tau, dt):
    """
    The Reynolds & Heeger (2009) Normalization Equation.
    Calculates the response for a single time step.

    Parameters:
    - d: Input drive (excitatory).
    - sigma: Semisaturation constant (prevents division by zero).
    - p: Exponent for rectification.
    - r_prev: Response at previous time step (t-1).
    - tau: Time constant.
    - dt: Time step size.

    Returns:
    - r: Updated firing rate.
    - f: Normalization result (Drive / Suppression).
    - s: Suppressive drive (sum of pool).
    """
    # Suppression pool: Sum across feature space (broad inhibition)
    pool = np.abs(d)
    s = np.sum(pool)

    # Normalization step
    f = d / (s + halfExp(sigma, p))

    # Update firing rates (Leaky integrator)
    r = r_prev + (dt/tau) * (-r_prev + f)

    return r, f, s


def n_model(p):
    """
    Main Model Loop: Iterates through time and layers.
    Layers: S1 -> S2 -> Decision, plus parallel Attention Layers.
    """
    idx = 0
    # Time loop
    for t in np.arange(p.dt, p.T+p.dt, p.dt):
        idx = idx + 1

        # === Sensory Layer 1 (S1) ===
        inp = p.stim[:, idx]

        # Calculate Excitatory Drive
        if np.any(inp):
            drive = p.rfresp[inp.astype(bool), :] * p.contrast
            drive = halfExp(drive, p.p)
        else:
            drive = np.zeros([p.ntheta, 1])

        # Apply Attention Gain (Endogenous * Exogenous)
        # rav/rai are responses from the attention layers at t-1
        attGain1 = halfExp(1 + p.rav[:, idx-1] * p.aAV)
        attGain2 = halfExp(1 + p.rai[:, idx-1] * p.aAI)
        attGain = attGain1 * attGain2
        
        p.d1[:, idx] = attGain * drive[0]

        # S1 Normalization
        p.r1[:, idx], p.f1[:, idx], p.s1[:, idx] = n_core(
            p.d1[:, idx], p.sigma1, p.p, p.r1[:, idx-1], p.tau1, p.dt
        )

        # === Sensory Layer 2 (S2) ===
        # Input comes from S1 response
        drive = halfExp(p.r1[:, idx], p.p)
        p.d2[:, idx] = drive
        
        # S2 Normalization
        p.r2[:, idx], p.f2[:, idx], p.s2[:, idx] = n_core(
            p.d2[:, idx], p.sigma2, p.p, p.r2[:, idx-1], p.tau2, p.dt
        )

        # === Decision Layer ===
        # Decode response (Template matching)
        response = p.r2
        rfresp = np.zeros((2, p.rfresp.shape[1], 1))
        rfresp[:, :, 0] = p.rfresp[0:2, :]
        
        evidence = decodeEvidence(response[:, idx].T, rfresp[:, :, 0])
        evidence = evidence * p.decisionWindows[idx]
        evidence = np.array(evidence)
        evidence[np.abs(evidence) < 1e-3] = 0
        
        p.dd[0, idx] = evidence

        # Decision Normalization (Accumulation)
        p.rd[:, idx], p.fd[:, idx], p.sd[:, idx] = n_core(
            p.dd[:, idx], p.sigmaD, p.p, p.rd[:, idx-1], p.tauD, p.dt
        )

        # === Voluntary Attention Layer ===
        inp = p.task[:, idx]
        drive = halfExp(inp, p.p)
        p.dav[:, idx] = np.sum(drive)
        
        p.rav[:, idx], p.fav[:, idx], p.sav[:, idx] = n_core(
            p.dav[:, idx], p.sigmaA, p.p, p.rav[:, idx-1], p.tauAV, p.dt
        )

        # === Involuntary Attention Layer ===
        # Drive is prefiltered S1 response
        drive = prefilter(p.r1, p.h, p.p, p.dt, idx)
        p.dai[:, idx] = np.sum(drive)
        
        p.rai[:, idx], p.fai[:, idx], p.sai[:, idx] = n_core(
            p.dai[:, idx], p.sigmaA, p.p, p.rai[:, idx-1], p.tauAI, p.dt
        )

    return p


# %% Simulation Functions

def run_model(p):
    """
    Full simulation routine covering multiple conditions (Demo mode).
    Note: This function appears to be for testing/debugging logic.
    """
    condnames = ['invalid', 'valid']
    
    # Orientations setup (Degrees -> Radians)
    tilt = 2 * np.pi / 180
    orientations = np.array([np.pi/2+tilt, np.pi/2-tilt])
    p.norient = len(orientations)

    contrasts = 0.64
    seqs = 1
    Rhythms = np.array(np.arange(1, 6, seqs))
    stimseqs = [[1, 0]]

    ncond = len(condnames)
    rnRhythms = np.arange(0, len(Rhythms), 1)
    nRhythms = len(Rhythms)

    # Initialize RF responses
    prfresp = []
    for iO in np.arange(0, p.norient, 1):
        prfresp.append(rfResponse(orientations[iO], p.ntheta))
    p.rfresp = np.array(prfresp)

    ev = np.zeros((ncond, nRhythms))
    
    for icond in range(ncond):
        condname = condnames[icond]
        for iRhythms in rnRhythms:
            p.condname = condname
            p.iRhythms = Rhythms[iRhythms]
            p.contrast = contrasts
            p.nstim = 1  
            p.stimseq = stimseqs[0]
            p.orientseq = orientations[p.stimseq]

            # Distribute voluntary attention
            if p.disturbuteVoluntary:
                w = p.AVProp
                p.AVWeights = distributeAttention(w)
                if condname == "invalid":
                    p.AVWeights = p.winv
                elif condname == "valid":
                    p.AVWeights = p.wv

            # Initialize and run
            p = initTimeSeries(p)
            p = setStim(p)
            p = setTask(p)
            p = setDecisionWindows(p)
            p = n_model(p)

            # Extract evidence
            p.ev = np.array(p.rd[0, -1])
            # Correct sign based on stimulus identity (+ for CW, - for CCW)
            p.ev1 = p.ev * (-1) ** (np.array(p.stimseq)+1)
            p.ev2 = p.ev1 * np.array([p.scaling1, p.scaling2])
            ev[icond, iRhythms] = np.sum(np.abs(p.ev2))/2

    return ev


def run_model_cond(condname, p):
    """
    Optimized simulation run for a specific condition.
    Used during parameter fitting.
    """
    condnames = ['invalid', 'valid']
    if condname not in condnames:
        print('Error: Invalid condition name')
        return None
    
    tilt = 2 * np.pi / 180
    orientations = np.array([np.pi/2+tilt, np.pi/2-tilt])
    p.norient = len(orientations)

    contrasts = 0.64        
    seqs = 1                
    Rhythms = np.array(np.arange(1, 6, seqs))
    stimseqs = [[1, 0]]
    rnRhythms = np.arange(0, len(Rhythms), 1) 
    nRhythms = len(Rhythms)

    # RF setup
    prfresp = [] 
    for iO in np.arange(0, p.norient, 1):
        prfresp.append(rfResponse(orientations[iO], p.ntheta))
    p.rfresp = np.array(prfresp)

    ev = np.zeros(nRhythms)

    for iRhythms in rnRhythms:
        p.condname = condname
        p.iRhythms = Rhythms[iRhythms]
        p.contrast = contrasts
        p.nstim = 1  
        p.stimseq = stimseqs[0]
        p.orientseq = orientations[p.stimseq]

        # Distribute voluntary attention
        if p.disturbuteVoluntary:
            w = p.AVProp
            p.AVWeights = distributeAttention(w)
            if condname == "invalid":
                p.AVWeights = p.winv
            elif condname == "valid":
                p.AVWeights = p.wv

        # Execution
        p = initTimeSeries(p)
        p = setStim(p)
        p = setTask(p)
        p = setDecisionWindows(p)
        p = n_model(p)

        # Get final evidence
        p.ev = np.array(p.rd[0, -1])
        p.ev2 = p.ev * p.scaling1
        ev[iRhythms] = np.sum(np.abs(p.ev2))/2
        
    return ev


# %% Optimization & Metrics

def sse(y_true, y_pred):
    """Calculates Sum of Squared Errors."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between true and predicted values.")
    return np.sum((y_true - y_pred)**2)


def r_squared(y_true, y_pred):
    """Calculates Coefficient of Determination (R^2)."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between true and predicted values.")

    mean_y_true = np.mean(y_true)
    tss = np.sum((y_true - mean_y_true)**2) # Total Sum of Squares
    sse = np.sum((y_true - y_pred)**2)      # Residual Sum of Squares

    if tss == 0:
        return 1.0 if sse == 0 else np.nan

    return 1 - (sse / tss)


def cm_wsci(df, conf_level=0.95, difference=True):
    """
    Calculates Cousineau-Morey Within-Subject Confidence Intervals (Loftus & Masson, 1994).
    """
    n, k = df.shape
    diff_factor = 2**0.5 / 2 if difference else 1
    
    sub_mean = df.mean(axis=1)
    gra_mean = df.mean().mean()
    # Normalize data to remove between-subject variability
    df_norm = df.apply(lambda x: x - sub_mean + gra_mean)
    
    t_mat = []
    mean_mat = []
    for i in range(k):
        col_data = df_norm.iloc[:, i]
        if np.all(col_data == 0):
            t_val = np.finfo(float).eps
        else:
            t_val = stats.ttest_1samp(col_data, popmean=0)[0]
            if t_val == 0: t_val = np.finfo(float).eps
            
        t_mat.append(t_val)
        mean_mat.append(col_data.mean())
    
    t_mat = np.array(t_mat)
    mean_mat = np.array(mean_mat)
    
    c_factor = (k / (k - 1))**0.5
    moe_mat = mean_mat / t_mat * stats.t.ppf(1 - (1 - conf_level) / 2, n - 1) * c_factor * diff_factor
    return moe_mat


# %% Main Execution Block

if __name__ == "__main__":
    p = parameters()
    
    # ==========================================
    # 1. Data Loading & Preprocessing
    # ==========================================
    # PLACEHOLDER PATH - Update before running
    csv_path = r'../../data/exp2_data.csv' 
    
    if not os.path.exists(csv_path):
        print(f"Warning: Data file not found at {csv_path}. Please update the path.")
        # Create dummy data structure for code validation if file is missing
        # In production, this should raise an error.
    else:
        print(f"Loading data: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        raw_df.columns = raw_df.columns.str.strip()

        # Calculate means for fitting
        mean_df = raw_df.groupby(['emo', 'validity', 'nRhythms'])['correct'].mean().reset_index()

        # Initialize parameter storage: (3 emotions x 2 validities x 9 parameters)
        all_params = np.zeros((3, 2, 9))
        emo_list = ['negative', 'neutral', 'positive']
        validity_list = ['invalid', 'valid']

        # ==========================================
        # 2. Model Fitting Loop (BADS)
        # ==========================================
        try: 
            # Try loading existing fit parameters to save time
            params_path = r'../../data/exp2_model_fit_params_total.csv'
            fit_params = pd.read_csv(params_path)
            emotion_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
            validity_map = {"invalid": 0, "valid": 1}
            feature_columns = ['attn', 'pn', 'tauAV', 'aAV', 'tauAI', 'aAI', 'AVDur', 'base', 'AVWeights']

            for index, row in fit_params.iterrows():
                emotion = emotion_map[row['emo']]
                validity = validity_map[row['validity']]
                all_params[emotion, validity] = row[feature_columns].values
                
            print("Loaded existing parameters from file.")

        except FileNotFoundError:
            # Perform Fitting if no parameter file exists
            print("No parameter file found. Starting model fitting using BADS...")
            bads_options = {"max_fun_evals": 100, "uncertainty_handling": False}

            for emo_idx, emo_name in enumerate(emo_list):
                print(f"\n=== Processing Emotion: {emo_name} ({emo_idx+1}/3) ===")
                
                for valid_idx, valid_cond in enumerate(validity_list):
                    print(f"   >>> Fitting Condition: {valid_cond}")
                    
                    # Get target data
                    current_data = mean_df[
                        (mean_df['emo'] == emo_name) & 
                        (mean_df['validity'] == valid_cond)
                    ].sort_values('nRhythms')['correct'].values
                    
                    target_dat = current_data # Assuming d-prime or accuracy
                    
                    # Define Objective Function for BADS
                    def objective_function(x):
                        p.attn, p.pn, p.tauAV, p.aAV, p.tauAI, p.aAI, p.AVDur, p.base, p.AVWeights = x
                        ev = run_model_cond(valid_cond, p)
                        return 1 - r_squared(target_dat, ev) # Minimize (1 - R^2)

                    # Define Bounds: [attn, pn, tauAV, aAV, tauAI, aAI, AVDur, base, AVWeights]
                    x0 = np.array([3,  6,  9, 45,  5,  5, 100, 100, 0.1]) # Starting Point
                    lb = np.array([1,  1,  1,  1,  1,  1,  10,  10, 0.0]) # Lower Bound
                    ub = np.array([10, 20, 50, 50, 50, 50, 1000, 300, 1.0]) # Upper Bound
                    
                    # Run Optimization
                    bads = BADS(fun=objective_function, x0=x0, lower_bounds=lb, upper_bounds=ub, options=bads_options)
                    res = bads.optimize()
                    
                    all_params[emo_idx, valid_idx, :] = res['x']
                    
                    # Backup save
                    with open(f'{emo_name}_{valid_cond}.pkl', 'wb') as f:
                        pickle.dump(res['x'], f)

            print("\nFitting completed.")    

        # ==========================================
        # 3. Generate Fitted Curves
        # ==========================================
        fit_data_list = []
        for emo_idx, emo_name in enumerate(emo_list):
            for valid_idx, valid_cond in enumerate(validity_list):
                current_params = all_params[emo_idx, valid_idx, :]
                p.attn, p.pn, p.tauAV, p.aAV, p.tauAI, p.aAI, p.AVDur, p.base, p.AVWeights = current_params
                
                ev_values = run_model_cond(valid_cond, p)
                
                for i, val in enumerate(ev_values):
                    fit_data_list.append({
                        'emotion': emo_name.capitalize(),
                        'validity': valid_cond,
                        'strength': i,
                        'value': val
                    })

        df_ev = pd.DataFrame(fit_data_list)
        df_ev.to_csv('ev_fitted.csv', index=False)

        # ==========================================
        # 4. Plotting (APA/Journal Style)
        # ==========================================
        plt.rcParams['font.family'] = 'Arial'
        PALETTE = {'invalid': '#D77071', 'valid': '#6888F5'}
        output_dir = 'pictures'
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # Calculate Error Bars (SEM/Loftus-Masson)
        df_wide = raw_df.pivot(index='sub', columns=['emo', 'validity', 'nRhythms'], values='d')
        d_moe = cm_wsci(df_wide)
        
        summary_stats = raw_df.groupby(['emo', 'nRhythms', 'validity']).agg(d_mean=('d', 'mean')).reset_index()
        summary_stats['d_sem'] = d_moe 

        font_size = 30

        for emo_name in emo_list:
            emotion_condition = emo_name.capitalize()
            output_filename = f'Figure_5points_{emotion_condition}_dprime_fit_const.svg'
            
            # Prepare Data
            # 1. Real Data
            this_emo_real = raw_df[raw_df['emo'] == emo_name].groupby(['validity', 'nRhythms'])['correct'].mean().reset_index()
            this_emo_real['d_plot'] = stats.norm.cdf(this_emo_real['correct']) * 2 # Transformation for plotting
            
            real_inv = this_emo_real[this_emo_real['validity'] == 'invalid'].sort_values('nRhythms')
            real_val = this_emo_real[this_emo_real['validity'] == 'valid'].sort_values('nRhythms')
            
            # 2. Error Bars
            this_emo_sem = summary_stats[summary_stats['emo'] == emo_name]
            yerr_inv = this_emo_sem[this_emo_sem['validity'] == 'invalid'].sort_values('nRhythms')['d_sem'].values / 12
            yerr_val = this_emo_sem[this_emo_sem['validity'] == 'valid'].sort_values('nRhythms')['d_sem'].values / 12
            
            # 3. Fitted Data
            this_emo_fit = df_ev[df_ev['emotion'] == emotion_condition]
            this_emo_fit['value'] = stats.norm.cdf(this_emo_fit['value']) * 2
            
            fit_inv_vals = this_emo_fit[this_emo_fit['validity'] == 'invalid']['value']
            fit_val_vals = this_emo_fit[this_emo_fit['validity'] == 'valid']['value']
            
            # Calculate Goodness of Fit
            r2 = r_squared(this_emo_real['d_plot'], this_emo_fit['value'])
            print(f'{emo_name} R^2: {r2:.3f}')
            
            # --- Plot ---
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            
            x_real = np.arange(1, 6)
            x_fit_plot = np.arange(1, 6) 
            
            # Plot Data Points + Error Bars
            ax.errorbar(x_real, real_inv['d_plot'], yerr=yerr_inv, fmt='s', 
                        color=PALETTE['invalid'], markersize=8, label='Invalid (Data)', 
                        elinewidth=1.5, capsize=3)

            ax.errorbar(x_real, real_val['d_plot'], yerr=yerr_val, fmt='o', 
                        color=PALETTE['valid'], markersize=8, label='Valid (Data)', 
                        elinewidth=1.5, capsize=3)
            
            # Plot Fitted Lines
            ax.plot(x_fit_plot, fit_inv_vals, linestyle='--', color=PALETTE['invalid'], 
                    linewidth=2, label='Invalid (Fit)')
            ax.plot(x_fit_plot, fit_val_vals, linestyle='-', color=PALETTE['valid'], 
                    linewidth=2, label='Valid (Fit)')
            
            # Styling
            ax.set_xlabel('Rhythm', fontsize=font_size)
            ax.set_ylabel("d'", fontsize=font_size)
            ax.spines[['left','bottom']].set_linewidth(1.5)
            ax.spines[['right', 'top']].set_visible(False)
            
            ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=font_size-10)
            ax.set_xticks(x_real)

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 1, 2, 3]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
                      frameon=False, loc='upper left', fontsize=14, 
                      bbox_to_anchor=(1.02, 1.0))
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, output_filename)
            plt.savefig(save_path, format='svg', dpi=600)
            print(f"Figure saved: {save_path}")
            # plt.show()