"""
`init_subspec!(m::AnSchorfheide)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::AnSchorfheide)
    if subspec(m)     == "ss0"
        return
    elseif subspec(m) == "ss2"
        return ss2!(m)
    elseif subspec(m) == "ss3"
        return ss3!(m)
    elseif subspec(m) == "ss4"
        return ss4!(m)
    elseif subspec(m) == "ss5"
        return ss5!(m)
    elseif subspec(m) == "ss6"
        return ss6!(m)
    elseif subspec(m) == "ss7"
        return ss7!(m)
    elseif subspec(m) == "ss8"
        return ss8!(m)
    elseif subspec(m) == "ss9"
        return ss9!(m)
    elseif subspec(m) == "ss10"
        return ss10!(m)
    elseif subspec(m) == "ss11"
        return ss11!(m)
    else
        error("Invalid subspec. Options are: ss0, ss2, ss3, ss4.")
    end
end


# Used for just plain exponential transformation (with b=0 rather than b=1e5). Experimental
function ss2!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (0., Inf), (0., 0.), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., Inf), (0., 0.), ModelConstructors.Exponential(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., Inf), (0., 0.), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., 0.), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., 0.), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., 0.), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., Inf), (0., 0.), ModelConstructors.Exponential(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., Inf), (0., 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., Inf), (0., 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 0.5), fixed=false,
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

# Uses b = 1e5 for exponential transformation
function ss3!(m::AnSchorfheide)
    b = 1e9
 m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., Inf), (0., b), ModelConstructors.Exponential(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., Inf), (0., b), ModelConstructors.Exponential(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., Inf), (0., b), ModelConstructors.Exponential(), RootInverseGamma(4, 0.5), fixed=false,
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

function ss4!(m::AnSchorfheide)
    #m.parameters = Array{AbstractParameter{Float64},1}()
    b = 0.
    m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=true,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(1.5, 0.25), fixed=true,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=true,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=true,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=true,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=true,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=true,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-5, 10.), (1e-5, 10.), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-5, 10.), (1e-5, 10.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-5, 10.), (1e-5, 10.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss5!(m::AnSchorfheide)
    b = 0.
    m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=true,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., 3.), (0., 3.), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=true,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 3.), (0., 3.), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=true,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=true,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=true,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=true,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss6!(m::AnSchorfheide)
    b = 0.
    m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=true,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0.5, 3.), (0.5, 3.), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 3.), (0., 3.), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=true,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=true,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=true,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=true,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=true,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=true,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=true,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss7!(m::AnSchorfheide)
    b = 0.
    m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=true,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., 1e2), (1e-5, 1e2 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 1e2), (1e-5, 1e2 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=true,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=true,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=true,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=true,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=true,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=true,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=true,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss8!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (1.6, 2.3), (1.6, 2.3), ModelConstructors.SquareRoot(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0.5, 1.), (0.5, 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1.01, 1.5), (1.01, 1.5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0.1, 0.8), (0.1, 0.8), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (1e-4, 0.2), (1e-4, 0.2), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (4., 12.), (4., 12.), ModelConstructors.SquareRoot(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (1., 2.), (1., 2.), ModelConstructors.SquareRoot(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0.2, 0.5), (0.2, 0.5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0.2, 0.5), (0.2, 0.5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0.9, 1. - 1e-5), (0.9, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0.3, 0.8), (0.3, 0.8), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1.2, 1.7), (1.2, 1.7), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0.7, 1.1), (0.7, 1.1), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss6!(m::AnSchorfheide)
    b = 0.
    m <= parameter(:τ, 1.9937, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(2., 0.5), fixed=true,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0.5, 3.), (0.5, 3.), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 3.), (0., 3.), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=true,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., Inf), (0., b), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=true,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., Inf), (0., b), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=true,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=true,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=true,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=true,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=true,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (1e-5, 3.), (1e-5, 3.), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=true,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss9!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1., 1e5), (1. + 1e-5,  1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss10!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1., 20.), (1. + 1e-5,  20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., 1e3), (1e-5, 1e3 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss11!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (1., 20.), (1. + 1e-5,  20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., 20.), (1e-5, 20. - 1e-5), ModelConstructors.SquareRoot(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., 10.), (1e-5, 10. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., 10.), (1e-5, 10. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., 10.), (1e-5, 10. - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end

function ss12!(m::AnSchorfheide)
    m <= parameter(:τ, 1.9937, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(2., 0.5), fixed=false,
                   description="τ: The inverse of the intemporal elasticity of substitution.",
                   tex_label="\\tau")

    m <= parameter(:κ, 0.7306, (0., 1.), (1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="κ: Composite parameter in New Keynesian Phillips Curve.",
                   tex_label="\\kappa")

    m <= parameter(:ψ_1, 1.1434, (0., 1e5), (0. + 1e-5,  1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(1.5, 0.25), fixed=false,
                   description="ψ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\psi_1")
    m <= parameter(:ψ_2, 0.4536, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.25), fixed=false,
                   description="ψ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\psi_2")

    m <= parameter(:rA, 0.0313, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(0.5, 0.5), fixed=false,
                   description="rA: β (discount factor) = 1/(1+ rA/400).",
                   tex_label="rA")

    m <= parameter(:π_star, 8.1508, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), GammaAlt(7., 2.), fixed=false,
                   description="π_star: Target inflation rate.",
                   tex_label="\\pi*")

    m <= parameter(:γ_Q, 1.5, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), Normal(0.40, 0.20), fixed=false,
                   description="γ_Q: Steady state growth rate of technology.",
                   tex_label="\\gamma_Q")

    m <= parameter(:ρ_R, 0.3847, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_R: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_R")

    m <= parameter(:ρ_g, 0.3777, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                   description="ρ_g: AR(1) coefficient on g_t = 1/(1 - ζ_t), where ζ_t is government spending as a fraction of output.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_z, 0.9579, (0., 1.), (0.0 + 1e-5, 1. - 1e-5), ModelConstructors.SquareRoot(), Uniform(0.,1.), fixed=false,
                   description="ρ_z: AR(1) coefficient on shocks to the technology growth rate.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_R, 0.4900, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, .4), fixed=false,
                   description="σ_R: Standard deviation of shocks to the nominal interest rate.",
                   tex_label="\\sigma_R")

    m <= parameter(:σ_g, 1.4594, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 1.), fixed=false,
                   description="σ_g: Standard deviation of shocks to the government spending process.",
                   tex_label="\\sigma_g")

    m <= parameter(:σ_z, 0.9247, (0., 1e5), (1e-5, 1e5 - 1e-5), ModelConstructors.SquareRoot(), RootInverseGamma(4, 0.5), fixed=false,
                   description="σ_z: Standard deviation of shocks to the technology growth rate process.",
                   tex_label="\\sigma_z")
    m <= parameter(:e_y, 0., fixed=true,
                   description="e_y: Measurement error on GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_π, 0., fixed=true,
                   description="e_π: Measurement error on inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_R, 0., fixed=true,
                   description="e_R: Measurement error on the interest rate.",
                   tex_label="e_R")
end
