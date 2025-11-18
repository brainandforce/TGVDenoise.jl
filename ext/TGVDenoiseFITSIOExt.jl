module TGVDenoiseFITSIOExt

using TGVDenoise
using FITSIO

using Base.Filesystem: cp

function tgv_denoise_channels!(
    hdu::ImageHDU,
    alpha,
    beta;
    kwargs...
)
    denoised = TGVDenoise.tgv_denoise_channels(read(hdu), alpha, beta; kwargs...)
    write(hdu, denoised)
    return denoised
end

function tgv_denoise_channels!(
    fits::FITS,
    alpha,
    beta;
    hdu_index = 1,
    kwargs...
)
    hdu = fits[hdu_index]
    tgv_denoise_channels!(hdu, alpha, beta, kwargs...)
end

"""
    tgv_denoise_channels(
        fits::FITS,
        alpha,
        beta;
        prefix = "tgv",
        hdu_index = 1,
        force = false,
        kwargs...
    )

Performs total TGV denoising independently on each color channel on the selected header data unit
(HDU) of a FITS file.
By default, the first HDU is chosen, but this can be changed with the `hdu_index` keyword.

The original file is copied to a file appended with a prefix given by the `prefix` parameter plus
an underscore.
If the file already exists, it will not be overwritten unless `force` is set to `true`
"""
function TGVDenoise.tgv_denoise_channels(
    fits::FITS,
    alpha,
    beta;
    prefix = "tgv",
    hdu_index = 1,
    force = false,
    kwargs...
)
    # Create a copy of the target file
    new_filename = let 
        sf = splitdir(fits.filename)
        joinpath(first(sf), string(ifelse(isnothing(prefix), "", prefix)) * '_' * last(sf))
    end
    cp(fits.filename, new_filename; force)
    FITS(new_filename, "r+") do new_fits
        tgv_denoise_channels!(new_fits[hdu_index], alpha, beta; kwargs...)
    end
end

function tgv_denoise_color!(
    hdu::ImageHDU,
    alpha,
    beta;
    kwargs...
)
    denoised = TGVDenoise.tgv_denoise_color(read(hdu), alpha, beta; kwargs...)
    write(hdu, denoised)
    return denoised
end

function tgv_denoise_color!(
    fits::FITS,
    alpha,
    beta;
    hdu_index = 1,
    kwargs...
)
    hdu = fits[hdu_index]
    tgv_denoise_color!(hdu, alpha, beta, kwargs...)
end

"""
    tgv_denoise_color(
        fits::FITS,
        alpha,
        beta;
        prefix = "tgv",
        hdu_index = 1,
        force = false,
        kwargs...
    )

Performs TGV denoising on a color FITS file by transforming it to the YCoCg color space and
denoising the luminance and chrominance components.
By default, the first HDU is chosen, but this can be changed with the `hdu_index` keyword.

The original file is copied to a file appended with a prefix given by the `prefix` parameter plus
an underscore.
If the file already exists, it will not be overwritten unless `force` is set to `true`.
"""
function TGVDenoise.tgv_denoise_color(
    fits::FITS,
    alpha,
    beta;
    prefix = "tgv",
    hdu_index = 1,
    force = false,
    kwargs...
)
    # Create a copy of the target file
    new_filename = let 
        sf = splitdir(fits.filename)
        joinpath(first(sf), string(ifelse(isnothing(prefix), "", prefix)) * '_' * last(sf))
    end
    cp(fits.filename, new_filename; force)
    FITS(new_filename, "r+") do new_fits
        tgv_denoise_color!(new_fits[hdu_index], alpha, beta; kwargs...)
    end
end

export tgv_denoise_channels!, tgv_denoise_color!

end
