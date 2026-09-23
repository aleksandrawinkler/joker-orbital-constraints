#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable
import pymc as pm
import arviz as az
import thejoker as tj
import corner

#input dates/times in UTC 
def load_data():
    dates = ["2020-06-01 12:00:00", "2020-06-11 15:30:00", "2020-06-23 08:45:12", "2020-07-06 02:15:00", "2020-07-19 23:00:00", "2020-07-31 11:10:30"] #times in UTC times
    rv_values = [10.5,  28.4,  34.1,  21.8,  -5.2, -25.2] #km/s
    rv_errors = [0.5, 0.4, 0.5, 0.6, 0.4, 0.5] #km/s
    t = Time(dates, scale='utc').jd   
    rv = np.array(rv_values) * (u.km / u.s)
    rv_err = np.array(rv_errors) * (u.km / u.s)
    
    data = tj.RVData(t=t, rv=rv, rv_err=rv_err)
    return data

def orbit(data, prior_samples_size=500_000, mcmc_draws=1000):

    prior = tj.JokerPrior.default(P_min=2 * u.day, P_max=1000 * u.day, sigma_K0=30 * (u.km / u.s), sigma_v=100 * (u.km / u.s)) #adjust priors as needed
    joker = tj.TheJoker(prior)
    
    prior_samples = prior.sample(size=prior_samples_size)
    joker_samples = joker.rejection_sample(data, prior_samples=prior_samples, max_posterior_samples=256)
    
    print(f"{len(joker_samples)} posteriors.")
    
    with prior.model:
        mcmc_init = joker.setup_mcmc(data, joker_samples)
        trace = pm.sample(draws=mcmc_draws, tune=1000, initvals=mcmc_init, chains=2, cores=2, target_accept=0.90)
        
    mcmc_samples = tj.JokerSamples.from_inference_data(prior, trace, data)
    mcmc_samples = mcmc_samples.wrap_K()
    
    return mcmc_samples, trace

def companion_mass(mcmc_samples, m1_val=1.2, m1_err=0.05):

    P_arr = mcmc_samples['P'].value.flatten()
    e_arr = mcmc_samples['e'].value.flatten()
    K_arr = mcmc_samples['K'].to(u.km/u.s).value.flatten()

    #mass function of a binary system that only uses K, P, and e. Sets a lower limit to the possible mass of the hidden companion. Need inclination to get exact mass. 
    f_M = 1.0361e-7 * (1 - e_arr**2)**1.5 * P_arr * (K_arr**3)

    #dont know exact inclination so this uses primary objects' mass to estimate
    cos_i = np.random.uniform(0, 1, size=len(f_M))
    sin_i = np.sin(np.arccos(cos_i))

    m1_samples = np.random.normal(m1_val, m1_err, size=len(f_M))

    m2_samples = (f_M**(1/3) * m1_samples**(2/3)) / sin_i
    for _ in range(12):
        m2_samples = ((f_M * (m1_samples + m2_samples)**2) ** (1/3)) / sin_i
        
    return f_M, m2_samples


if __name__ == "__main__":
    known_mass = 1.2 
    rv_data = load_data() 
    samples, trace = orbit(rv_data)
    f_M, m2_posterior = companion_mass(samples, m1_val=known_mass)

    summary = az.summary(trace, var_names=["P", "e", "K"])
    print(summary)

    
    q2 = np.percentile(m2_posterior, [16, 50, 84])
    print(f"\nHidden companion mass constraints")
    print(f"Median M2: {q2[1]:.3f} M_sun")
    print(f"68%: {q2[0]:.3f} to {q2[2]:.3f} M_sun")
    
    df_corner = samples.tbl.to_pandas()
    corner.corner(df_corner[["P", "e", "K"]], show_titles=True)
    plt.savefig("orbit_degeneracies.png")
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(m2_posterior, bins=60, density=True, histtype='step', lw=2.5, color='midnightblue', label='Posterior $M_2$')
    plt.axvline(q2[1], color='crimson', linestyle='--', label=f'Median: {q2[1]:.2f} $M_\\odot$')
    plt.axvspan(q2[0], q2[2], color='crimson', alpha=0.15, label='1-Sigma Interval')
    plt.title("Hidden companion mass distribution")
    plt.xlabel("Companion Mass ($M_\\odot$)")
    plt.ylabel("Probability Density")
    plt.legend(frameon=True)
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("companion_mass_constraint.png")
    
    plt.show()


# In[ ]:




