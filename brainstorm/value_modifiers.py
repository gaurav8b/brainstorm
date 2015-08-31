#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.describable import Describable


class ValueModifier(Seedable, Describable):

    __undescribed__ = {'layer_name', 'view_name'}

    def __init__(self):
        super(ValueModifier, self).__init__()
        self.layer_name = ''
        self.view_name = ''

    def __call__(self, handler, view):
        raise NotImplementedError()

    def __repr__(self):
        return "<{}.{}.{}}>".format(self.layer_name, self.view_name,
                                    self.__class__.__name__)


class ConstrainL2Norm(ValueModifier):

    """
    Consrrains the L2 norm of the incoming weights to every neuron/unit to be
    less than or equal to a limit.
    If the L2 norm for any unit exceeds the limit, the weights are
    rescaled such that the squared L2 norm equals the limit.
    Ignores Biases.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HX': ConstrainL2Norm()})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    __default_values__ = {'limit': 1.0}

    def __init__(self, limit=1.0):
        super(ConstrainL2Norm, self).__init__()
        self.limit = limit

    def __call__(self, handler, view):
        if len(view.shape) < 2:
            return
        mat = handler.reshape(view, (view.shape[0], view.size // view.shape[0]))
        sq_norm = handler.allocate((view.shape[0], 1))
        divisor = handler.allocate(sq_norm.shape)

        handler.sum_t(mat * mat, axis=1, out=sq_norm)
        handler.mult_st(1 / self.limit, sq_norm ** 0.5, out=divisor)
        handler.clip_t(divisor, a_min=1.0,
                       a_max=np.finfo(handler.dtype).max, out=divisor)
        handler.divide_mv(mat, divisor, mat)

    def __repr__(self):
        return "<{}.{}.ConstrainL2Norm to {:0.4f}>"\
            .format(self.layer_name, self.view_name, self.limit)


class ClipValues(ValueModifier):

    """
    Clips (limits) the weights to be between low and high.
    Defaults to low=-1 and high=1.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HR': ClipValues()})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    def __init__(self, low=-1., high=1.):
        super(ClipValues, self).__init__()
        self.low = low
        self.high = high

    def __call__(self, handler, view):
        handler.clip_t(view, self.low, self.high, view)

    def __repr__(self):
        return "<{}.{}.ClipValues [{:0.4f}; {:0.4f}]>"\
            .format(self.layer_name, self.view_name, self.low, self.high)


class MaskValues(ValueModifier):

    """
    Multiplies the weights elementwise with the mask.

    This can be used to clamp some of the weights to zero.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HR': MaskValues(M)})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    __undescribed__ = {'device_mask'}

    def __init__(self, mask):
        super(MaskValues, self).__init__()
        assert isinstance(mask, np.ndarray)
        self.mask = mask
        self.device_mask = None

    def __call__(self, handler, view):
        if (self.device_mask is None or
                not isinstance(self.device_mask, handler.array_type)):
            self.device_mask = handler.allocate(self.mask.shape)
            handler.set_from_numpy(self.device_mask, self.mask)
        handler.mult_tt(view, self.device_mask, view)


class FreezeValues(ValueModifier):

    """
    Prevents the weights from changing at all.

    If the weights argument is left at None it will remember the first weights
    it sees and resets them to that every time.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': FreezeValues()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """

    __undescribed__ = {'weights', 'device_weights'}

    def __init__(self, weights=None):
        super(FreezeValues, self).__init__()
        self.weights = weights
        self.device_weights = None

    def __call__(self, handler, view):
        if self.weights is None:
            self.weights = handler.get_numpy_copy(view)

        if (self.device_weights is None or
                not isinstance(self.device_weights, handler.array_type)):
            self.device_weights = handler.allocate(self.weights.shape)
            handler.set_from_numpy(self.device_weights, self.weights)

        handler.copy_to(view, self.device_weights)