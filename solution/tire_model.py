"""
Tire degradation model for the F1 race simulation.

This module contains the tire compound parameters and the core
lap time calculation logic. The parameters were derived from
analysing patterns in 30,000 historical race results.

Lap time formula:
    lap_time = base_lap_time + compound_offset + degradation_penalty

Where:
    - compound_offset: inherent speed difference of each tire compound
        SOFT is fastest (negative offset), HARD is slowest (positive offset),
        MEDIUM is the baseline (zero offset)

    - degradation_penalty: increases once tires exceed their "cliff" age
        penalty = base_rate * temp_scale * max(0, tire_age - cliff)
        temp_scale = 1 + temp_coeff[compound] * (track_temp - temp_ref)

    - tire_age starts at 0 when fitted, increments by 1 BEFORE each lap
      so the first lap on a set of tires is driven at age 1

Temperature effect:
    Each compound has its own temperature sensitivity coefficient.
    Higher temperatures accelerate degradation, lower temperatures slow it.
    The relationship is linear around a reference temperature.
"""


# ---- Compound speed offsets (seconds per lap relative to MEDIUM) ----
# SOFT tires are inherently faster per lap, HARD tires are slower
COMPOUND_BASE = {
    "SOFT": -1.0,
    "MEDIUM": 0.0,
    "HARD": 0.8,
}


# ---- Degradation rate (seconds per lap once past the cliff) ----
# SOFT degrades fastest, HARD degrades slowest
DEGRADATION_RATE = {
    "SOFT": 1.475,
    "MEDIUM": 0.75,
    "HARD": 0.375,
}


# ---- Performance cliff (laps of consistent pace before degradation) ----
# After this many laps, the tires start losing grip progressively
CLIFF = {
    "SOFT": 10,
    "MEDIUM": 20,
    "HARD": 30,
}


# ---- Temperature sensitivity coefficients (per compound) ----
# Controls how much track temperature scales the degradation rate
TEMP_COEFFICIENT = {
    "SOFT": 0.025,
    "MEDIUM": 0.025,
    "HARD": 0.026,
}

# Reference temperature — degradation rates are "as-is" at this temperature
TEMP_REFERENCE = 24.0


def get_effective_degradation_rate(compound, track_temp):
    """
    Calculate the effective degradation rate for a compound at a given
    track temperature. The rate scales linearly with temperature difference
    from the reference point.
    """
    delta_temp = track_temp - TEMP_REFERENCE
    base_rate = DEGRADATION_RATE[compound]
    temp_scale = 1.0 + TEMP_COEFFICIENT[compound] * delta_temp
    return base_rate * temp_scale


def calculate_lap_time(base_lap_time, compound, tire_age, track_temp):
    """
    Calculate a single lap time given current tire state.

    Parameters:
        base_lap_time: the track's characteristic lap time (seconds)
        compound: "SOFT", "MEDIUM", or "HARD"
        tire_age: how many laps driven on this set (including current lap)
        track_temp: track temperature in degrees Celsius

    Returns:
        lap time in seconds
    """
    compound_offset = COMPOUND_BASE[compound]
    effective_rate = get_effective_degradation_rate(compound, track_temp)
    cliff = CLIFF[compound]

    # degradation only kicks in after the cliff
    degradation = effective_rate * max(0.0, tire_age - cliff)

    return base_lap_time + compound_offset + degradation
