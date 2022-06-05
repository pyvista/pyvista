"""
.. _image_fft_perlin_example:

Fast Fourier Transform with Perlin Noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to apply a Fast Fourier Transform (FFT) to a
:class:`pyvista.UniformGrid` using :func:`pyvista.UniformGridFilters.fft`
filter.

Here, we demonstrate FFT usage by first generating Perlin noise using
:func:`pyvista.sample_function() <pyvista.core.imaging.sample_function>` to
sample :func:`pyvista.perlin_noise <pyvista.core.common_data.perlin_noise>`,
and then performing FFT of the sampled noise to show the frequency content of
that noise.

"""

import numpy as np

import pyvista as pv

###############################################################################
# Start by generating some `Perlin Noise <https://en.wikipedia.org/wiki/Perlin_noise>`_.
#
# Note that we are generating it in a flat plane and using 10 Hz in the x
# direction and 5 Hz in the y direction.
#

freq = [10, 5, 0]
noise = pv.perlin_noise(1, freq, (0, 0, 0))
xdim, ydim = (500, 500)
sampled = pv.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(xdim, ydim, 1))

# plot the sampled noise
sampled.plot(cpos='xy', show_scalar_bar=False, text='Perlin Noise')

###############################################################################
# Next, perform a FFT of the noise and plot the frequency content.
# For the sake of simplicity, we will only plot the content in the first
# quadrant.
#
# Note the usage of :func:`numpy.fft.fftfreq` to get the frequencies.
#
sampled_fft = sampled.fft()
freq = np.fft.fftfreq(sampled.dimensions[0], sampled.spacing[0])

# only show the first quadrant
subset = sampled_fft.extract_subset((0, xdim // 4, 0, ydim // 4, 0, 0))

# shift the position of the uniform grid to match the frequency
subset.origin = (0, 0, 0)
spacing = np.diff(freq[:2])[0]
subset.spacing = (spacing, spacing, spacing)


###############################################################################
# Now, plot the noise in the frequency domain. Note how there is more high
# frequency content in the x direction and this matches the frequencies given
# to :func:`pyvista.perlin_noise <pyvista.core.common_data.perlin_noise>`.

pl = pv.Plotter()
pl.add_mesh(subset, cmap='gray', show_scalar_bar=False)
pl.camera_position = 'xy'
pl.show_bounds(xlabel='X Frequency', ylabel='Y Frequency')
pl.add_text('Frequency Domain of the Perlin Noise')
pl.show()

###############################################################################
# Low Pass Filter
# ~~~~~~~~~~~~~~~
# For fun, let's perform a low pass filter on the frequency content and then
# convert it back into the "time" domain by immediately applying a reverse FFT.
#
# As expected, we only see low frequency noise.

low_pass = sampled_fft.low_pass(0.5, 0.5, 0.5).rfft()
low_pass['ImageScalars'] = low_pass['ImageScalars'][:, 0]  # remove the complex data
low_pass.plot(cpos='xy', show_scalar_bar=False, text='Low Pass of the Perlin Noise')


###############################################################################
# High Pass Filter
# ~~~~~~~~~~~~~~~~
# This time, let's perform a high pass filter on the frequency content and then
# convert it back into the "time" domain by immediately applying a reverse FFT.
#
# As expected, we only see the high frequency noise content as the low
# frequency noise has been attenuated.

high_pass_noise = sampled_fft.high_pass(5, 5, 5).rfft()
high_pass_noise['ImageScalars'] = high_pass_noise['ImageScalars'][:, 0]  # remove the complex data
high_pass_noise.plot(cpos='xy', show_scalar_bar=False, text='High Pass of the Perlin Noise')
