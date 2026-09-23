unprime(k::Symbol) = Symbol(replace(string(k), "′" => ""))

@inline function sslogdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
    nt::NamedTuple, ::Val{k}) where {k}
return (length(d[k]) > 1 ? exp.((@view x[d[k]]) + nt[k]) : exp(x[d[k][1]] + nt[k]))
end

@inline function sslogdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
    nt::NamedTuple, ::Val{k}) where {k}
return exp(x[d[k]] + nt[k])
end

#likely can remove
@inline function sslogdeviation2level_unprimekeys(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, <: UnitRange},
                nt::NamedTuple, ::Val{k}) where {k}
kunprime = unprime(k) # keys of nt are assumed to not have primes on them, but we still want to use prime keys for d
return (length(d[k]) > 1 ? exp.((@view x[d[k]]) + nt[kunprime]) : exp(x[d[k][1]] + nt[kunprime]))
end

#likely can remove
@inline function sslogdeviation2level_unprimekeys(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, Int},
                nt::NamedTuple, ::Val{k}) where {k}
kunprime = unprime(k) # keys of nt are assumed to not have primes on them, but we still want to use prime keys for d
return exp(x[d[k]] + nt[kunprime])
end

# overload: ss is vector (StateSS/ControlSS); d can be Union{Int,UnitRange}
@inline function sslogdeviation2level(x::AbstractVector{<:Real}, d::AbstractDict, ss::AbstractVector, ::Val{k}) where {k}
    idx = d[k]
    return idx isa Int ? exp(ss[idx] + x[idx]) : exp.(@view(ss[idx]) .+ @view(x[idx]))
end

#added overload
@inline function sslogdeviation2level_unprimekeys(x::AbstractVector{<:Real}, d::AbstractDict, ss::AbstractVector, ::Val{k}) where {k}
    idx = d[unprime(k)]
    return idx isa Int ? exp(ss[idx] + x[idx]) : exp.(@view(ss[idx]) .+ @view(x[idx]))
end

macro sslogdeviations2levels(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2level($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro sslogdeviations2levels_unprimekeys(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2level_unprimekeys($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end
