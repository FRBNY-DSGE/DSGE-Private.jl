function get_field(dict, field)
    if dict isa Dict
        # Try String key first (MATLAB default), then Symbol
        if haskey(dict, string(field))
            return dict[string(field)]
        elseif haskey(dict, field)
            return dict[field]
        elseif haskey(dict, Symbol(field))
            return dict[Symbol(field)]
        else
            error("Field $field not found in dict")
        end
    else
        # Assume struct-like access
        return getproperty(dict, field isa Symbol ? field : Symbol(field))
    end
end