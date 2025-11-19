module TGVDenoise

#= General rules for treating inputs:
  - If a 2D array is provided as input, treat it as raw monochrome image data if the element type is
    a Number, or a color image if a type from ColorTypes.jl is used.
  - If a 3D array is provided as input, try to treat it as a color image
    Check if the last axis has size 3
    If not, check the first axis
    Finally, error
  - Images of different dimensionalities should explicitly have their element types explicitly given
    as a ColorTypes.jl type (even a Gray type if needed)
=#

function dx_plus(u)
    v = zero(u)
    s = size(v,1)
    v[1,:] .= u[1,:]
    v[1:(s-1),:] .= u[2:s,:] .- u[1:(s-1),:]
    return v
end

function dx_minus(u)
    v = copy(u)
    s = size(v,1)
    v[2:(s-1),:] .= u[2:(s-1),:] - u[1:(s-2),:]
    v[s,:] .= -u[s,:]
    return v
end

function dy_plus(u)
    v = zero(u)
    s = size(v,2)
    v[:,1] .= u[:,1]
    v[:,1:(s-1)] .= u[:,2:s] .- u[:,1:(s-1)]
    return v
end

function dy_minus(u)
    v = copy(u)
    s = size(v,2)
    v[:,2:(s-1)] .= u[:,2:(s-1)] - u[:,1:(s-2)]
    v[:,s] .= -u[:,s]
    return v
end

function forward_index(i::CartesianIndex{N}, dim) where N
    return CartesianIndex(ntuple(n -> i[n] + (n == dim), Val(N)))
end

function backward_index(i::CartesianIndex{N}, dim) where N
    return CartesianIndex(ntuple(n -> i[n] - (n == dim), Val(N)))
end

"""
    TGVDenoise.d_central!(out, data, dim)

Performs a central difference approximation of `data` along dimension `dim`, storing the result in
an identically sized destination, `out`, without allocations.
"""
function d_central!(out, data, dim)
    # Check for matching dimensions before doing anything
    size(out) == size(data) || throw(
        DimensionMismatch("Output must have size equal to the input ($(size(data))).")
    )
    @inbounds for i in CartesianIndices(data)
        if i[dim] == firstindex(data, dim)
            out[i] = (data[forward_index(i, dim)] - data[i]) / 2
        elseif i[dim] == lastindex(data, dim)
            out[i] = (data[i] - data[backward_index(i, dim)]) / 2
        else
            out[i] = (data[forward_index(i, dim)] - data[backward_index(i, dim)]) / 2
        end
    end
    return out
end

"""
    d_central!(out, data)

Performs a central difference approximation of `data` along all dimensions, storing the result in
`out` without allocations.
"""
function d_central!(out, data)
    sz = tuple(size(data)..., ndims(data))
    size(out) == sz || throw(
        DimensionMismatch("Output must have size equal to the input ($sz).")
    )
    for d in 1:ndims(data)
        d_central!(view(out, ntuple(Returns(:), Val(ndims(data)))..., d), data, d)
    end
    return out
end

"""
    d_central!(out, data, [dim])

Performs a central difference approximation of `data` along all dimensions, returning a new array.
The central difference is taken along `dim` if provided; if it is not provided the central 
difference is taken along all dimensions, and a larger array (of size
`(size(data)..., ndims(data))`) is returned.
"""
d_central(data, dim) = d_central!(similar(data), data, dim)
d_central(data) = d_central!(similar(data, size(data)..., ndims(data)), data)

reduce_drop(op, itr; dims, kw...) = dropdims(reduce(op, itr; dims, kw...); dims)

#=
function reduce_drop(::typeof(hypot), itr; init = zero(eltype(itr)), dims, kw...)
    sz = ntuple(n -> ifelse(n in dims, 1, size(itr, n)), Val(ndims(itr)))
    result = similar(itr, sz)

    return dropdims(result; dims)
end
=#

#=
"""
    TGVDenoise.d_plus!(out, data, dim)

Performs a forward finite difference approximation of `data` along dimension `dim`, storing the 
result in an identically sized destination, `out`, without allocations.
"""
function d_plus!(out, data, dim)
    # Check for matching dimensions before doing anything
    size(out) == size(data) || throw(
        DimensionMismatch("Output must have size equal to the input ($(size(data))).")
    )
    @inbounds for i in CartesianIndices(data)
        if i[dim] == lastindex(data, dim)
            out[i] = (data[i] - data[backward_index(i, dim)]) / 2
        else
            out[i] = data[forward_index(i, dim)] - data[i]
        end
    end
    return out
end

"""
    d_plus!(out, data)

Performs a forward finite difference approximation of `data` along all dimensions, storing the 
result in `out` without allocations.
"""
function d_plus!(out, data)
    sz = tuple(size(data)..., ndims(data))
    size(out) == sz || throw(
        DimensionMismatch("Output must have size equal to the input ($sz).")
    )
    for d in 1:ndims(data)
        d_plus!(view(out, ntuple(Returns(:), Val(ndims(data)))..., d), data, d)
    end
    return out
end

"""
    d_plus!(out, data, [dim])

Performs a central difference approximation of `data` along all dimensions, returning a new array.
The central difference is taken along `dim` if provided; if it is not provided the central 
difference is taken along all dimensions, and a larger array (of size
`(size(data)..., ndims(data))`) is returned.
"""
d_plus(data, dim) = d_plus!(similar(data), data, dim)
d_plus(data) = d_plus!(similar(data, size(data)..., ndims(data)), data)
=#

"""
    tgv_denoise_mono(
        image,
        alpha,
        beta;
        iterations = 1000,
        tolmean = 1e-6,
        tolsup = 1e-4,
        strength = 1
    )

Performs total generalized variation (TGV) denoising on a monochrome image.

The `alpha` and `beta` parameters are the regularization parameters for the optimization.

The `tolmean` and `tolsup` parameters are used to determine whether the iterations should terminate
early.
`tolmean` is the mean change in pixel values between iterations, and `tolsup` is the maximum change
that occurred at a particular iteration.

`strength` is either a value between 0 and 1 that determines the proportion of the denoised image to
combine with the final image, or an array of values between 0 and 1 with the same dimensions as the
input image determining the denoising strength for each individual pixel.
"""
function tgv_denoise_mono(
    image,
    alpha,
    beta;
    iterations = 10,
    tolmean = 1e-6,
    tolsup = 1e-4,
    strength = 1
)
    all(iszero, strength) && return copy(image)
    if any(x -> x < 0, strength) || any(x -> x > 1, strength)
        throw(ArgumentError("Denoising strength must be between 0 and 1"))
    end
    # initializations
    u_new = copy(image)
    u_old = similar(u_new)
    u_bar = copy(image)
    p_new = zeros(eltype(image), size(image)..., 2)
    p_old = similar(p_new)
    p_bar = zeros(eltype(image), size(image)..., 2)
    v = zeros(eltype(image), size(image)..., 2)
    w = zeros(eltype(image), size(image)..., 2, 2)
    du_bar = similar(v) # reuse to store derivatives
    dp_bar = similar(w)
    z = similar(image)  # temporary array
    L2 = 8
    k = 1
    while k <= iterations
        u_old .= u_new
        p_old .= p_new

        tau = inv(k+1)
        sigma = (k+1) / L2
        # TODO: for the steps z[z .< 1], we effectively have a ReLU function + 1
        # What happens if we change it to something different?
        v .+= sigma .* d_central!(du_bar, u_bar) .- p_bar
        z .= reduce_drop(hypot, v, dims=3, init=zero(eltype(v))) ./ alpha
        z[z .< 1] .= 1
        for n in 1:ndims(image)
            v[:,:,n] ./= z
        end

        # Take the unsymmetrized derivative of p_bar
        for n in 1:ndims(image)
            d_central!(view(dp_bar, :, :, n, :), view(p_bar, :, :, n))
        end
        # Add the derivative in a symmetrized fashion
        w .+= (sigma / 2) .* dp_bar
        w .+= (sigma / 2) .* permutedims(dp_bar, (1, 2, 4, 3))
        z .= reduce_drop(hypot, w, dims=(3,4), init=zero(eltype(v))) ./ beta
        z[z .< 1] .= 1
        for (a,b) in Iterators.product(1:ndims(image), 1:ndims(image))
            w[:,:,a,b] ./= z
        end

        u_new .+= tau .* (d_central(@view(v[:,:,1]), 1) .+ d_central(@view(v[:,:,2]), 2) .+ image)
        u_new ./= (1 + tau)
        
        # Check if we need to exit early
        change = abs.(u_new - u_old)
        max_change = maximum(change)
        mean_change = sum(change) / length(change)
        level_change = (sum(u_new) ./ sum(u_old))
        if (max_change <= tolsup || mean_change <= tolmean)
            @info "Reached termination criteria at $k iterations!\n" *
                "\n\tmaximum change: $max_change" *
                "\n\tmean change: $mean_change" *
                "\n\tmean pixel value ratio: $level_change"
            break
        end
        
        p_new .+= tau .* v
        p_new[:,:,1] .+= tau .* (d_central(@view(w[:,:,1,1]), 1) .+ d_central(@view(w[:,:,1,2]), 2))
        p_new[:,:,2] .+= tau .* (d_central(@view(w[:,:,1,2]), 1) .+ d_central(@view(w[:,:,2,2]), 2))
        p_bar .= 2 .* p_new .- p_old
        u_bar .= 2 .* u_new .- u_old

        @info "At iteration $k:" * 
            "\n\tmaximum change: $max_change" * 
            "\n\tmean change: $mean_change" *
            "\n\tmean pixel value ratio: $level_change"
        k += 1
    end
    all(isone, strength) && return u_new
    # Don't promote the element type of the result
    _strength = convert.(eltype(image), strength)
    return (u_new .* _strength) .+ (image .* (1 .- _strength))
end

"""
    tgv_denoise_channels(
        image,
        alpha,
        beta;
        iterations = 1000,
        tolmean = 1e-6,
        tolsup = 1e-4
    )

Performs total generalized variation (TGV) denoising on each channel of a color image independently.

The `alpha` and `beta` parameters are the regularization parameters for the optimization.

The `tolmean` and `tolsup` parameters are used to determine whether the iterations should terminate
early.
`tolmean` is the mean change in pixel values between iterations, and `tolsup` is the maximum change
that occurred at a particular iteration.
"""
function tgv_denoise_channels(
    image,
    alpha,
    beta;
    kwargs...
)
    if ndims(image) == 2 && eltype(image) <: Number
        # Treat image as monochrome
        return tgv_denoise_mono(image, alpha, beta; kwargs...)
    elseif ndims(image) == 3
        # Try to treat image as RGB
        # Sum RGB data to create luminance channel
        if last(size(image)) == 3
            return stack(tgv_denoise_mono.(eachslice(image, dims=3), alpha, beta; kwargs...))
        elseif first(size(image)) == 3
            return stack(tgv_denoise_mono.(eachslice(image, dims=1), alpha, beta; kwargs...))
        end
    end
    throw(DimensionMismatch("Unreadable image format")) # TODO: better explanation
end

"""
    tgv_denoise_luminance(
        image,
        alpha,
        beta;
        iterations = 1000,
        tolmean = 1e-6,
        tolsup = 1e-4
    )

Performs total generalized variation (TGV) denoising on only the luminance component of a monochrome
image.
The luminance component is calculated by summing the RGB values of the image, denoised with the TGV
algorithm, and combined with the 

The `alpha` and `beta` parameters are the regularization parameters for the optimization.

The `tolmean` and `tolsup` parameters are used to determine whether the iterations should terminate
early.
`tolmean` is the mean change in pixel values between iterations, and `tolsup` is the maximum change
that occurred at a particular iteration.
"""
function tgv_denoise_luminance(
    image,
    alpha,
    beta;
    kwargs...
)
    if ndims(image) == 2 && eltype(image) <: Number
        # Treat image as monochrome
        return tgv_denoise_mono(image, alpha, beta; kwargs...)
    elseif ndims(image) == 3
        # Try to treat image as RGB
        # Sum RGB data to create luminance channel
        if last(size(image)) == 3
            image_L = sum(eachslice(image, dims=3))
            denoised_L = tgv_denoise_mono(image_L, alpha, beta; kwargs...)
            return stack(denoised_L .* c ./ image_L for c in eachslice(image, dims=3))
        elseif first(size(image)) == 3
            image_L = sum(eachslice(image, dims=1))
            denoised_L = tgv_denoise_mono(image_L, alpha, beta; kwargs...)
            return stack(denoised_L .* c ./ image_L for c in eachslice(image, dims=1))
        end
    end
    throw(DimensionMismatch("Unreadable image format")) # TODO: better explanation
end

include("color.jl")
include("multithread.jl")

export tgv_denoise_mono, tgv_denoise_channels, tgv_denoise_luminance, tgv_denoise_color

end # module TGVDenoise
