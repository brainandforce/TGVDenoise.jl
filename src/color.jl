# Use YCoCg color space to perform the denoising

#=
# We don't depend on StaticArrays
const RGB_TO_YCOCG = SMatrix{3,3}(1//4, 1, -1//2, 1//2, 0, 1, 1//4, -1, -1//2)
const YCOCG_TO_RGB = inv(RGB_TO_YCOCG)

# Scalar conversion functions which we don't seem to need at the moment
function rgb_to_ycocg(R::T, G::T, B::T) where T<:Real
    Co  = R - B
    tmp = B + Co/2
    Cg  = G - tmp
    Y   = tmp + Cg/2
    (Y, Co, Cg)
end

rgb_to_ycocg(R::Real, G::Real, B::Real) = rgb_to_ycocg(promote(R, G, B)...)
rgb_to_ycocg(rgb::Tuple{Real, Real, Real}) = rgb_to_ycocg(rgb...)

function ycocg_to_rgb(Y::T, Co::T, Cg::T)
    tmp = Y - Cg/2;
    G   = Cg + tmp;
    B   = tmp - Co/2;
    R   = B + Co;
    (R, G, B)
end

ycocg_to_rgb(Y::Real, Co::Real, Cg::Real) = ycocg_to_rgb(promote(Y, Co, Cg)...)
ycocg_to_rgb(ycocg::Tuple{Real, Real, Real}) = ycocg_to_rgb(ycocg...)
=#

"""
    rgb_to_ycocg(image; colordim = 3)

Converts an image from the RGB to YCoCg color space.
"""
function rgb_to_ycocg(image; colordim = 3)
    # Do not use mapslices to do this; it's super slow!
    result = similar(image)
    (R, G, B) = selectdim.(tuple(image), colordim, 1:3)
    (Y, Co, Cg) = selectdim.(tuple(result), colordim, 1:3)
    Co .= R - B     # tmp
    Y  .= B + Co/2
    Cg .= G - Y
    Y .+= Cg/2
    return result
end

function ycocg_to_rgb(image; colordim = 3)
    # Do not use mapslices to do this; it's super slow!
    result = similar(image)
    (Y, Co, Cg) = selectdim.(tuple(image), colordim, 1:3)
    (R, G, B) = selectdim.(tuple(result), colordim, 1:3)
    R  .= Y - Cg/2  # tmp
    G  .= Cg + R
    B  .= R - Co/2
    R .+= Co/2
    return result
end

"""
    function tgv_denoise_color(
        image,
        alpha,
        beta;
        iterations = 10,
        tolmean = 1e-6,
        tolsup = 1e-4,
        strength = 1,
        highest_snr_channel = 2
    )

Performs total generalized variation (TGV) denoising on a color image by transforming it to the
YCoCg color space and denoising the luminance (Y) and two chrominance (Co and Cg) components before
transforming back to an RGB image.
For color images, this transform should provide superior denoising compared to denoising of the
individual RGB channels.
For monochrome images, the image is denoised normally.

The `alpha` and `beta` parameters are the regularization parameters for the optimization.

The `iterations` parameter determines the maximum number of iterations per channel.
The `tolmean` and `tolsup` parameters are used to determine whether the iterations should terminate
early.
`tolmean` is the mean change in pixel values between iterations, and `tolsup` is the maximum change
that occurred at a particular iteration.

# Notes on color space transform

The [YCoCg](https://en.wikipedia.org/wiki/YCoCg) color space allows for lossless transformation of
an RGB image to a luminance-based representation.
However, the luminance channel is not a sum of the RGB channels; instead it doubles the weight of
the green channel.
For unfiltered one-shot color cameras with a standard Bayer matrix (which matches the luminance 
weighting of YCoCg), this is beneficial.
But for monochrome cameras and exotic palettes, it may be wise to swap the order of the channels so
that the highest SNR channel (often H-alpha for narrowband images) is green.
This can be accomplished by changing the `highest_snr_channel` parameter to either 1 or 3 (by 
default it is 2, assuming the green channel is second).

This advice is only based on a hunch, and may not hold true in all cases.
"""
function tgv_denoise_color(
    image,
    alpha,
    beta;
    highest_snr_channel = 2,
    kwargs...
)
    inds = collect(1:3)
    inds[2], inds[highest_snr_channel] = inds[highest_snr_channel], inds[2]
    if ndims(image) == 2 && eltype(image) <: Number
        # Treat image as monochrome
        return tgv_denoise_mono(image, alpha, beta; kwargs...)
    elseif ndims(image) == 3
        # Try to treat image as RGB
        # Sum RGB data to create luminance channel
        if last(size(image)) == 3
            ycocg_image = rgb_to_ycocg(view(image, :, :, inds); colordim = 3)
            result_ycocg = tgv_denoise_channels(ycocg_image, alpha, beta; kwargs...)
            return ycocg_to_rgb(result_ycocg)[:,:,inds]
        elseif first(size(image)) == 3
            ycocg_image = rgb_to_ycocg(view(image, inds, :, :); colordim = 1)
            result_ycocg = tgv_denoise_channels(ycocg_image, alpha, beta; kwargs...)
            return ycocg_to_rgb(result_ycocg)[inds,:,:]
        end
    end
    throw(DimensionMismatch("Unreadable image format")) # TODO: better explanation
end
