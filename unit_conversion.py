import numpy as np

def gg_to_cpd(gg, m_total, half_life, molar_m):
    return gg * m_total * 6.022e23 * np.log(2) / molar_m / half_life / 365

def gg_to_bqkg(gg, half_life, molar_m):
    return gg * 1000 * 6.022e23 * np.log(2) / molar_m / half_life / 365 / 24 / 60 / 60

def gg_to_mbqkg(gg, half_life, molar_m):
    return gg * 1000 * 1000 * 6.022e23 * np.log(2) / molar_m / half_life / 365 / 24 / 60 / 60

def gg_to_mbqvolumem3(gg, half_life, molar_m, volume, rho):
    return gg * 1000 * 1000 * 6.022e23 * np.log(2) * rho * volume / molar_m / half_life / 365 / 24 / 60 / 60

def cpd_to_gg(cpd, m_total, half_life, molar_m):
    return  cpd * molar_m * half_life * 365 / m_total /  6.022e23 / np.log(2)

gg = 10e-15 # 'g/g'
half_life = 4.458e9 # 'year'
molar_m = 238.028910 # 'g/mol'
m_total = 16.202e6 # 'g'
volume = 20 # 'm^3'
rho = 860 # 'kg/m^3'

cpd = gg_to_cpd(gg, m_total, half_life, molar_m)
print(cpd)

mbqkg = gg_to_mbqkg(gg, half_life, molar_m)
print(mbqkg)

mbqvolumem3 = gg_to_mbqvolumem3(gg, half_life, molar_m, volume, rho)
print(mbqvolumem3)

gg_ = cpd_to_gg(cpd, m_total, half_life, molar_m)
print(gg_)