# Introduction

Machine learning emulators of existing physical parameterizations have mostly focused on improving climate model execution time be replacing the slower parameterizations like the radiative transfer code (Chevroiter) or conversely improving model physics for a fixed execution time by emulating expensive yet accurate schemes. For example, Gagne et. al. emulated an expensive bin microphysics scheme. Most studies focus on emulation radiation simply because it is the most expensive climate model parameterization.

Performance issues aside, emulating physical parameterization is of interest on
machine learning grounds alone. Physical parameterizations produce highly
non-gaussian multi-dimensional targets...and regime where traditional ML often
fails.  In this way parameterization emulation has similar challenges to
emulating the sub-grid physics of cloud resolving models.

Perhaps as a result of the focus on performance rather than on solving
challenging ML problems, the literature has few demonstrations of emulators on
*fast* physical processes. It
also doesn't feature any non-local vertical dependence.
By fast, we mean faster than the intrinsic dynamical
*timescales of a coarse-resolution model. Fast physical processes include
*boundary layer turbulence and moist physics.

Gagne et. al. succesfully emulated of the warm-rain
*process, but this problem is only one component of the microphysics scheme. 
The scheme they emulated converts cloud water into rain within a single (x,y,z)
grid box.
Because this process is not directly associated with latent heating it only has
an indirect affect on the dynamics compared to a scheme that predicts full
temperature and humidity tendencies.
Moreover, the work is somewhat specific to the CAM context and choice of
prognostic variables.
We focus on emulating an *entire* microphysical scheme, including the
condensation process, conversion, and fall of hydrometeors.

More importantly, most works have focused on demonstrating the success of their
emulators on their problem of interest...but fail to provide general guidance on
what steps are required for emulators to work more broadly.
By focusing on emulating a particular simple parametization...the Zhao Carr microphysics, we hope to answer the following questions:
- What is the relationship between online and offline skill?
- How much offline skill is "enough" to train a stable emulator?
- What is the most parsimonous architecture that achieves success?


Gagne: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002268