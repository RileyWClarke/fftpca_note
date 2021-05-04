from scipy.signal import fftconvolve
import numpy as np

def _window_sum_1d(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    return window_sum

#1d method
def match_template1d(ts, template, pad_input=False, mode='constant',
                   constant_values=0):
   
    assert (ts.ndim == 1 and template.ndim == 1, "can only be use with time series")
    
    assert (ts.shape[0] > template.shape[0], "Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
  
    ts = np.array(ts, dtype=np.float64, copy=False)

    pad_width = tuple((width, width) for width in template.shape)
    if mode == 'constant':
        ts = np.pad(ts, pad_width=pad_width, mode=mode,
                       constant_values=constant_values)
    else:
        st = np.pad(ts, pad_width=pad_width, mode=mode)

    # Use special case for 2-D images for much better performance in
    # computation of integral images
    if ts.ndim == 1:
        ts_window_sum = _window_sum_1d(ts, template.shape)
        ts_window_sum2 = _window_sum_1d(ts**2, template.shape)
    '''
    if image.ndim == 2:
        image_window_sum = _window_sum_2d(image, template.shape)
        image_window_sum2 = _window_sum_2d(image ** 2, template.shape)
    elif image.ndim == 3:
        image_window_sum = _window_sum_3d(image, template.shape)
        image_window_sum2 = _window_sum_3d(image ** 2, template.shape)
    '''
    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)
    #=========#
    xcorr = fftconvolve(ts, template[::-1],
                            mode="valid")[1:-1]
    


    numerator = xcorr - ts_window_sum * template_mean

    denominator = ts_window_sum2
    np.multiply(ts_window_sum, ts_window_sum, out=ts_window_sum)
    np.divide(ts_window_sum, template_volume, out=ts_window_sum)
    denominator -= ts_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=np.float64)

    # avoid zero-division
    mask = denominator > np.finfo(np.float64).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        if pad_input:
            d0 = (template.shape[i] - 1) // 2
            d1 = d0 + ts.shape[i]
        else:
            d0 = template.shape[i] - 1
            d1 = d0 + ts.shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))

    return response[tuple(slices)]
