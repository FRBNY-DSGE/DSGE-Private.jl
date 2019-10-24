"""
`init_subspec!(m::AnSchorfheide)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::AnSchorfheide)
    if subspec(m) == "ss0"
        return
    elseif subspec(m) == "ss2"
        return ss2!(m)
    elseif subspec(m) == "ss3"
        return ss3!(m)
    else
        error("This subspec should be a 0")
    end
end


# Used for just plain exponential transformation (with b=0 rather than b=1e5). Experimental
function ss2!(m::AnSchorfheide)
 m <= parameter(:τ, 1.9937, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (1e-20, 1e1), (1e-20, 1e1), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false,

                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-20, 1e5), (1e-20, 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
  m <= parameter(:e_y, 0.20*0.579923, fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0.20*1.470832, fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0.20*2.237937, fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

# Really experimental
function ss3!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (1e-20, 1e1), (1e-20, 1e1), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false,

                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (1e-20, 1-1e-7), (1e-20, 1-1e-7), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-20, 1e2), (1e-20, 1e5), ModelConstructors.Exponential(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0.20*0.579923, fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0.20*1.470832, fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0.20*2.237937, fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end
