#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers import default_handler
from brainstorm.structure.buffer_views import BufferView
from brainstorm.structure.layout import get_buffer_type
from brainstorm.utils import sort_by_index_key


def create_buffer_views_from_layout(layout, buffers):
    if '@slice' in layout:
        start, stop = layout['@slice']
        shape = layout['@shape']
        buffer_type = get_buffer_type(shape)
        full_buffer = buffers[buffer_type][..., start:stop]
        full_buffer = full_buffer.reshape(full_buffer.shape[:-1] + shape[buffer_type:])
    else:
        full_buffer = None

    if layout['@type'] == 'BufferView':
        children = [(n, create_buffer_views_from_layout(sub_node, buffers))
                    for n, sub_node in sorted(layout.items(), key=sort_by_index_key)
                    if not n.startswith('@')]
        if children:
            names, child_buffers = zip(*children)
        else:
            names, child_buffers = [], []
        return BufferView(names, child_buffers, full_buffer)
    else:  # layout['@type'] == 'array':
        assert full_buffer is not None, layout
        return full_buffer


class BufferManager(object):
    def __init__(self, layout, sizes, handler=default_handler):
        self.feature_sizes = sizes
        self.handler = handler
        self.layout = layout
        self.time_size = -1
        self.batch_size = -1
        self.size = -1
        self.full_buffer = None
        self.forward = None
        self.backward = None
        self.resize(0, 0)

    def get_total_size_slices_and_shapes(self):
        shapes = [
            (self.feature_sizes[0],),
            (self.batch_size, self.feature_sizes[1]),
            (self.time_size, self.batch_size, self.feature_sizes[2]),
        ]
        totals = np.cumsum([0] + [int(np.prod(s)) for s in shapes]*2)
        size = totals[-1]
        slices = [slice(i, j) for i, j in zip(totals[:-1], totals[1:])]
        return size, slices, shapes

    def resize(self, time_size, batch_size):
        if time_size == self.time_size and batch_size == self.batch_size:
            return  # lazy

        self.time_size = time_size
        self.batch_size = batch_size
        total_size, slices, shapes = self.get_total_size_slices_and_shapes()

        if total_size > self.size:
            self.full_buffer = self.handler.allocate(total_size)
            self.size = total_size

        full_forward_buffers = [
            self.full_buffer[slices[0]].reshape(shapes[0]),
            self.full_buffer[slices[1]].reshape(shapes[1]),
            self.full_buffer[slices[2]].reshape(shapes[2])
        ]
        self.forward = create_buffer_views_from_layout(
            self.layout, full_forward_buffers)

        # TODO optimization: allocate the backward pass only if needed
        full_backward_buffers = [
            self.full_buffer[slices[3]].reshape(shapes[0]),
            self.full_buffer[slices[4]].reshape(shapes[1]),
            self.full_buffer[slices[5]].reshape(shapes[2])
        ]
        self.backward = create_buffer_views_from_layout(
            self.layout, full_backward_buffers)

    def set_memory_handler(self, new_handler):
        # TODO: Preserve at least the weights
        self.full_buffer = None
        self.size = -1
        self.time_size = -1
        self.batch_size = -1
        self.handler = new_handler
        self.resize(0, 0)
