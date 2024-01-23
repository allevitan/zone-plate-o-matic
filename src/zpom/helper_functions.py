import torch as t

__all__ = ['linterp_1d']


def linterp_1d(y, xs, x0, dx, const=0):
    """Performs a 1d linear interpolation into y.

    Basically, this acts like torch.take(y,xs), but it interpolates into y
    instead of indexing into y

    """
    if dx <= 0:
        raise KeyError('dx cannot be less than or equal to 0')

    # Padding y with two copies of const at the end avoids an annoying bug
    # that occurs if any of the values in xs turn out to index exactly into
    # what would have been the last entry if we only padded it once
    padded_y = t.cat([t.as_tensor([const]),
                      y,
                      t.as_tensor([const,const])])

    # + 1 to acount for the fact that we padded y with const
    # we add it here, before the floor, so we don't need to worry
    # about how floor deals with negative numbers
    scaled_xs = ((xs - x0) / dx) + 1
    scaled_xs = t.clamp(scaled_xs, min=0, max=len(y) + 1)

    idx = t.floor(scaled_xs).to(dtype=t.int64)
    alpha = scaled_xs - idx

    return (1-alpha) * t.take(padded_y, idx) + alpha * t.take(padded_y, idx + 1)
