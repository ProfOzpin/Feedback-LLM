import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import StickBreakingTransform  # CORRECT IMPORT
from numpyro.infer import MCMC, NUTS
from pgmpy.factors.discrete import TabularCPD
import os

# Environment setup for Windows
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
numpyro.set_platform("cpu")


def estimate_cpt_with_hdp_simple(data, parent_name, child_name, parent_card, child_card):
    """
    Simplified HDP estimation that actually works with numpyro.
    """
    print(f"🔬 Estimating CPT for P({child_name} | {parent_name}) using HDP...")
    
    # Simple HDP Model using Dirichlet-Multinomial hierarchy
    def simple_hdp_model():
        # Global concentration parameter
        alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0))
        
        # Base distribution (shared across parent states)  
        base_probs = numpyro.sample("base_probs", dist.Dirichlet(jnp.ones(child_card)))
        
        # For each parent state, sample CPT column
        with numpyro.plate("parent_states", parent_card):
            # Each parent state gets its own Dirichlet with concentration = alpha * base_probs
            cpt_probs = numpyro.sample("cpt_probs", dist.Dirichlet(alpha * base_probs))
        
        # Observe the data
        for parent_val in range(parent_card):
            subset = data[data[parent_name] == parent_val]
            if len(subset) > 0:
                child_obs = subset[child_name].values
                with numpyro.plate(f"obs_{parent_val}", len(child_obs)):
                    numpyro.sample(f"likelihood_{parent_val}", 
                                 dist.Categorical(cpt_probs[parent_val]), 
                                 obs=child_obs)

    # Run MCMC
    kernel = NUTS(simple_hdp_model)
    mcmc = MCMC(kernel, num_warmup=300, num_samples=500, num_chains=1, progress_bar=True)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key)
    
    # Extract results
    samples = mcmc.get_samples()
    cpt_values = samples['cpt_probs'].mean(axis=0)
    
    # Create TabularCPD
    cpt = TabularCPD(variable=child_name,
                     variable_card=child_card,
                     values=cpt_values.T.tolist(),
                     evidence=[parent_name],
                     evidence_card=[parent_card])
    
    print("✅ HDP CPT estimation complete.")
    return cpt


# Test it
if __name__ == '__main__':
    sample_data = pd.DataFrame({
        'U_Fraction_Operations': [1, 1, 1, 1, 1, 0, 0, 0],
        'E_Fraction_Error':      [1, 1, 1, 0, 0, 0, 0, 0]
    })
    
    cpt = estimate_cpt_with_hdp_simple(
        data=sample_data,
        parent_name='U_Fraction_Operations',
        child_name='E_Fraction_Error',
        parent_card=2,
        child_card=2
    )
    
    print("\n--- HDP-Estimated CPT ---")
    print(cpt)
