# Proposed Organization

## Intro Note


The biggest differences will be in the _grouping_ of information about
model structure. The actual optimization, forecast, and computation steps will be more a
less a direct port.

The setup below will mainly focus on developing a more logical,
Julia-friendly definition of a model by focusing on the following
objects that define the model

- __Parameters__: With values, boundaries, fixed-or-not status, priors
- 
- __Model Indices__: Functions that associate an index with a name
  (e.g. "π_t" -> 1)

- __Equilibrium Conditions__: A function that takes parameters and model
  indices, then returns G0, G1, Ψ, and Π

These are enough to define the model structure. _Everything else_ is
essentially a function of these basics, and we can get to a forecast by
this chain:

- (Parameters + Model Indices + Eqcond Function) -> (TTT + RRR)
- 
- (TTT + RRR + Data) -> Estimation
- 
- (Estimation + TTT + RRR + Data) -> Forecast


## Parameters


- A parameter vector is an object of user-defined type "Parameters"
- 
- A parameter vector of type "Parameters" collects more fundamental
  individual objects of type "Para", which have fields

  - `Value`:   Float64
  - `IsFixed`: Logical
  - `Boundaries`
  - `PriorDistribution`
  - `TransformationType`: To go from model space to real line & vice versa

We define the following functions to act on objects of type parameter:

- `PriorDensity`: Look at value of parameter & its prior density; compute
- `Transform`:    To go from model space to real line
- `InvTransform`: To go from real line to model space

Parameter file for a particulare model looks like this, for all parameters:
```
# Define individual parameters as:
#
#   paraname = Param(Value, IsFixed, Bounds, PriorDistribution)
#
α = Param(0.2, false,[-1,1],     Beta(1,0.5))
β = Param(27,  true, [-Inf,Inf], Gamma(1,0.5))

# Collect parameters in vector
θ = Parameters(α, β, ...)
```

## Defining Indices

We have five functions for defining model indices, all collected into a
"ModelInds990" file.

All functions take a name like "π_t" or "rm" and give back an index.
Functions are:

- `endo`: For endogenous states
- `exo`:  Exogenous shocks
- `exp`:  Expectation shocks
- `eq`:   Equation indices
- `obs`:  Indices of named observables to use in measurement equation

They all look like this:
```
  function exo(name)
    names = ["π_sh", "rm_sh", "jerry", "george", "elaine", "kramer"]
    return find(map(nm -> (nm == name), names))
  end
```
- Since we don't care about the number, we only have to define the names.
- In this setup, adding states is easier, because we don't have to
  increment the index numbers of _everything_ when we add states.
- Super-automatic and less error prone; code focuses on the names just
  like we do.

## Equilibrium Conditions

A model-specific function of parameters and indices. Should look very
similar to our current code. Example
```
function eqcond990(θ, endo, exo, exp, eq)

  G0(eq("mp"), endo("R_t")) = 1;
  G1(eq("mp"), endo("R_t")) = θ.ρ;
  G0(eq("mp"), endo("π_t")) = -θ.Ψ_1;
  etc.

  return G0, G1, Ψ, Π

end
```
Measurement Equation will be very similar, taking parameters, model
indices, and data.

## Defining a Model

A model is a user defined type with the following fields

- `Parameters`:   An object of type "Parameters" defined above
- `ModelIndices`: The 5 functions defined above
- `Eqcond` Function
- `Measurement` Function

That's essentially enough to define the entire model structure.

We then build functions to act on a Model Type

- `dsgesolv`: Take a model type, use its parameters, indices, and eqconds to run through gensys and get TTT, RRR, CCC
- `Estimate`: Call dsgesolv, run gibb, etc.


## Building From There

Everything else will look like our current code, and will be more a less
a direct port of functions (Gensys, csminwel, kalman filter, etc.).

The main difference will be that rather than pass "TTT", "RRR", "YY" as
disjoint variables, we pass an object of "Model type" that encodes the
stucture, and _unpack_ that structure, either within functions or when
we pass arguments to a function.

We can also easily query key information about the model (whether
parametrs are fixed, what the prior distributions are, the index of
state "π_t"), since _one, single object_ of type "Model" will contain
all the relevant information.

