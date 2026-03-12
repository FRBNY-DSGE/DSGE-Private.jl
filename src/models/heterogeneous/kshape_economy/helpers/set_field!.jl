# Helper function to set Dict fields
function set_field!(dict, field, value)
    if dict isa Dict
        # Try to match existing key type, default to String
        if !isempty(dict)
            first_key = first(keys(dict))
            if first_key isa String
                dict[string(field)] = value
            else
                dict[field isa Symbol ? field : Symbol(field)] = value
            end
        else
            dict[string(field)] = value
        end
    else
        setproperty!(dict, field isa Symbol ? field : Symbol(field), value)
    end
end