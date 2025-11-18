function tgv_denoise_multithreaded(
    image,
    alpha,
    beta;
    colordim = 3,
    kwargs...
)
    result = similar(image)
    image_channels = eachslice(image, dims = colordim)
    result_channels = eachslice(result, dims = colordim)
    Threads.@threads for n in axes(image, colordim)
        result_channels[n] .= tgv_denoise_mono(image_channels[n], alpha, beta; kwargs...)
    end
    return result
end
