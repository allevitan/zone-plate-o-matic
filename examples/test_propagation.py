from zpom.propagation import FFT_DI_tiled

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from time import time

# Okay, so I need to do a test to confirm that the tile FFT_DI matches the
# non-tiled FFT_DI. I'd also like to test that the new non-tiled FFT_DI
# matches the old non-tiled FFR_DI.

# 40nm grid
# I need a 800 um field of view to keep the optic in frame

input_shape = [1024,1024]
output_shape = [3000,3000]
offset = (np.array(output_shape) - np.array(input_shape)) / 2

U_0 = t.exp(2j*np.pi*t.rand(*input_shape))
step = [50e-9,50e-9]

wavelength = 20.7e-9 #wavelength of light, in nm, at the Co L3 edge (59.9 eV)

input_extent = [0,step[0]*input_shape[0]*1e6,0,step[1]*input_shape[1]*1e6]
output_extent = [0,step[0]*output_shape[0]*1e6,0,step[1]*output_shape[1]*1e6]

# Major Parameters Here
N_ZP = 300
delta_r = 100e-9
f = 4 * N_ZP * delta_r**2 / wavelength
r = 2 * N_ZP * delta_r
NA = f / r

N_speckle = 30

sample_r = N_speckle * delta_r / 2


print('Number of zones',N_ZP)
print('Numerical Aperture', NA)
print('Focal Distance',f*1e3, 'mm')
print('ZP Diameter', 2 * r*1e6, 'um')
print('Number of speckles', N_speckle)
print('Focal Spot Diameter', 2*sample_r*1e3, 'um')

#save_location = '/home/abe/Dropbox (MIT)/Photon Scattering Group/Data/Fabrication/FERMI/20211101_RPI/'

#filename = (save_location + 'dr_%dnm_D_%dum_NS_%d_f_%.2fmm' %
#            (delta_r, 2 * r/1e3, N_speckle, f/1e6))


xs = t.arange(0, U_0.shape[0]) * step[0]
ys = t.arange(0, U_0.shape[1]) * step[1]
xs = xs - t.mean(xs)
ys = ys - t.mean(ys)
Xs, Ys = t.meshgrid(xs, ys)
Rs2 = (xs**2)[:,None] + (ys**2)[None,:]

U_0[Rs2>(sample_r**2)] = 0
offset = offset.astype(int)

U_0_untiled = t.zeros(*output_shape, dtype=U_0.dtype)
U_0_untiled[offset[0]:-offset[0],offset[1]:-offset[1]] = U_0

tim = time()

#propagated_untiled = FFT_DI(U_0_untiled,step,wavelength, -f, verbose=True)
propagated = FFT_DI_tiled(U_0, -f, wavelength, step,
                          output_shape=output_shape, offset=offset, verbose=True,
                          tile_shape=[530,570])


#print(t.sum(t.abs(propagated-propagated_untiled)) / t.sum(t.abs(propagated)))
mask = (t.angle(propagated)<0).to(dtype=t.complex64)
#mask_untiled = (t.angle(propagated_untiled)<0).to(dtype=t.complex64)


# We need to redefine this because propagated has a different shape
xs = t.arange(0, propagated.shape[0]) * step[0]
ys = t.arange(0, propagated.shape[1]) * step[1]
xs = xs - t.mean(xs)
ys = ys - t.mean(ys)
Xs, Ys = t.meshgrid(xs, ys)
Rs2 = (xs**2)[:,None] + (ys**2)[None,:]


mask[Rs2>r**2] = 0
mask[Rs2<(r/2)**2] = 0

#repropagated = FFT_DI(mask,step,wavelength, f, verbose=True)
repropagated = FFT_DI_tiled(mask, f, wavelength, step,
                          output_shape=output_shape, offset=-offset, verbose=True,
                          tile_shape=[530,570])

# To understand what an optical inspection would look like
#repropagated = FFT_DI(mask,step,532, f, verbose=True, threads=4)

mask = (mask * 255).to(dtype=t.uint8)


#np.save(filename + '_binary.npy', mask)
#image = Image.fromarray(mask.numpy())
#image.save('test_tiled_binary.png')
#np.save(filename + '_phase.npy', phase)
#image = Image.fromarray(mask)
#image.save(filename + '_phase.png')
#np.save('pytorch_result.npy',repropagated)

#plt.imshow(np.abs(U_0),extent=extent)
#plt.figure()
#plt.imshow(np.angle(propagated),extent=extent)
#plt.figure()
plt.imshow(np.abs(mask),extent=output_extent)
plt.xlabel('X (um)')
plt.ylabel('Y (um)')
plt.title('Optic Design')
plt.figure()
plt.imshow(np.abs(U_0),extent=input_extent)
plt.xlabel('X (um)')
plt.ylabel('Y (um)')
plt.title('Initial design focus amplitude')
plt.figure()
#plt.imshow(np.log(np.abs(repropagated)),extent=extent)
#plt.colorbar()
#plt.figure()
plt.imshow(np.abs(repropagated),extent=input_extent)
plt.show()



