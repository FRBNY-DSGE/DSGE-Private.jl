@inline function variable2index2value(x::AbstractVector{S}, d::AbstractDict, ::Val{k}) where {S <: Number, k}
    return length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]
end

macro variables2indices2values(args) # Based on @unpack from UnPack
    varnames, x_idict = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict = x_idict.args
    x_instance = gensym()
    idict_instance = gensym()
    kd = [:( $key = $variable2index2value($x_instance, $idict_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

@inline function sslogdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, UnitRange},
                                      nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? exp.((@view x[d[k]]) + nt[k]) : exp(x[d[k][1]] + nt[k]))
end

@inline function sslogdeviation2log(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, UnitRange},
                                    nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]) + nt[k]
end

@inline function ssdeviation2level(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, UnitRange},
                                   nt::NamedTuple, ::Val{k}) where {k}
    return (length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]) + nt[k]
end

@inline function get_deviation(x::AbstractVector{<: Real}, d::AbstractDict{Symbol, UnitRange}, ::Val{k}) where {k}
    return length(d[k]) > 1 ? (@view x[d[k]]) : x[d[k][1]]
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

macro sslogdeviations2logs(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2log($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro ssdeviations2levels(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $ssdeviation2level($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro sslogdeviations2logs(args) # Based on @unpack from UnPack
    varnames, x_idict_nt = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict, nt = x_idict_nt.args
    x_instance = gensym()
    idict_instance = gensym()
    nt_instance = gensym()
    kd = [:( $key = $sslogdeviation2log($x_instance, $idict_instance, $nt_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        local $nt_instance = $nt # handles if nt_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end

macro get_deviations(args) # Based on @unpack from UnPack
    varnames, x_idict = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, idict = x_idict.args
    x_instance = gensym()
    idict_instance = gensym()
    kd = [:( $key = $get_deviation($x_instance, $idict_instance, Val{$(Expr(:quote, key))}()) ) for key in varnames]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $x_instance = $x # handles if x_instance is not a variable but an expression
        local $idict_instance = $idict # handles if idict_instance is not a variable but an expression
        $kdblock
    end
    esc(expr)
end
