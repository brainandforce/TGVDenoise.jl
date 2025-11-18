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
    v[1:(s-1),:] = u[2:s,:] - u[1:(s-1),:]
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
    v[:,1:(s-1)] = u[:,2:s] - u[:,1:(s-1)]
    return v
end

function dy_minus(u)
    v = copy(u)
    s = size(v,2)
    v[:,2:(s-1)] .= u[:,2:(s-1)] - u[:,1:(s-2)]
    v[:,s] .= -u[:,s]
    return v
end

function d_plus(u, dim)
    v = zero(u)
    s = size(v, dim)
    i = ntuple(n -> ifelse(n === dim, 1, :), dim)
    v[i...] .= u[i...]
    j = ntuple(n -> ifelse(n === dim, firstindex(v, dim):(lastindex(v, dim) - 1), :), dim)
    v[j...] .= u[(j .+ 1)...] - u[j...]
    return v
end

function d_minus(u, dim)
    v = copy(u)
    s = size(v, dim)
    i = ntuple(n -> ifelse(n === dim, (firstindex(v, dim)+ 1):(lastindex(v, dim) - 1), :), dim)
    v[i...] .= u[i...] - u[(i .- 1)...]
    j = ntuple(n -> ifelse(n === dim, s, :), dim)
    v[j...] .= -u[j...]
    return v
end

"""
    sfdiff(image, dim)

Calculates a symmetric finite difference of an image along dimension `dim`.
(This is the average of the left and right finite differences)
"""
function sfdiff(image, dim)
    result = zero(image)
    i1 = ntuple(
        n -> ifelse(n === dim, firstindex(image, dim):(lastindex(image, dim) - 1), :),
        Val(ndims(image))
    )
    i2 = ntuple(
        n -> ifelse(n === dim, (firstindex(image, dim) + 1):lastindex(image, dim), :),
        Val(ndims(image))
    )
    tmp = (image[i2...] .- image[i1...]) / 2
    result[i1...] .+= tmp
    result[i2...] .+= tmp
    return result
end

sfdiff(image) = stack(sfdiff(image, n) for n in 1:ndims(image))

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
    u_old = copy(image)
    u_bar = copy(image)
    p_old = zeros(eltype(image), size(image)..., 2)
    p_bar = zeros(eltype(image), size(image)..., 2)
    v = zeros(eltype(image), size(image)..., 2)
    w = zeros(eltype(image), size(image)..., 2, 2)
    L2 = 8
    k = 1
    while k <= iterations
        tau = inv(k+1)
        sigma = (k+1) / L2
        # TODO: for the steps z[z .< 1], we effectively have a ReLU function + 1
        # What happens if we change it to something different?
        v[:,:,1] .+= sigma .* dx_plus(u_bar) .- p_bar[:,:,1]
        v[:,:,2] .+= sigma .* dy_plus(u_bar) .- p_bar[:,:,2]
        z = hypot.(v[:,:,1], v[:,:,2]) / alpha
        z[z .< 1] .= 1
        # v ./= z
        v[:,:,1] ./= z
        v[:,:,2] ./= z

        w[:,:,1,1] .+= sigma * dx_minus(p_bar[:,:,1])
        w[:,:,2,2] .+= sigma * dy_minus(p_bar[:,:,2])
        w[:,:,1,2] .+= sigma * (dy_minus(p_bar[:,:,1]) + dx_minus(p_bar[:,:,2])) / 2
        w[:,:,2,1] .= w[:,:,1,2]
        z = hypot.(w[:,:,1,1], w[:,:,1,2], w[:,:,1,2]) / beta
        z[z .< 1] .= 1
        # w ./ z
        w[:,:,1,1] ./= z
        w[:,:,2,2] ./= z
        w[:,:,1,2] ./= z
        w[:,:,2,1] .= w[:,:,1,2]

        u_new = (u_old .+ tau .* (dx_minus(v[:,:,1]) .+ dy_minus(v[:,:,2]) .+ image)) ./ (1+tau)
        # Check if we need to exit early
        change = abs.(u_new - u_old)
        max_change = maximum(change)
        mean_change = sum(change) / length(change)
        if (max_change <= tolsup || mean_change <= tolmean)
            @info "Reached termination criteria at $k iterations!\n" *
                "\n\tmaximum change: $max_change" * "\n\tmean change: $mean_change"
            break
        end

        p_new = zeros(eltype(p_old), size(p_old))
        p_new[:,:,1] .= p_old[:,:,1] .+ tau .* (v[:,:,1] .+ dx_plus(w[:,:,1,1]) .+ dy_plus(w[:,:,1,2]))
        p_new[:,:,2] .= p_old[:,:,2] .+ tau .* (v[:,:,2] .+ dx_plus(w[:,:,1,2]) .+ dy_plus(w[:,:,2,2]))
        p_bar[:,:,1] .= 2*p_new[:,:,1] - p_old[:,:,1]
        p_bar[:,:,2] .= 2*p_new[:,:,2] - p_old[:,:,2]
        u_bar .= 2 * u_new - u_old
        u_old .= u_new
        p_old .= p_new

        @info "At iteration $k:\n\tmaximum change: $max_change\n\tmean change: $mean_change"
        k += 1
    end
    all(isone, strength) && return u_old
    # Don't promote the element type of the result
    _strength = convert.(eltype(image), strength)
    return (u_old .* _strength) .+ (image .* (1 - _strength))
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
