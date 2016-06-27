#!/usr/bin/env python3

# Copyright 2015 asarcar Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Softmax."""

import numpy as np
# Plot softmax curves
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    tmp = np.exp(x)
    # tmp2 = np.array(np.sum(tmp, tmp.ndim-1))
    # tmp3 = tmp2.reshape(tmp2.shape+(1,))
    tmp3 = np.sum(tmp, 0)
    return tmp/tmp3


def main_fn():
    scores = [3.0, 1.0, 0.2]
    print(softmax(scores))
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

    plt.plot(x, softmax(scores)[0], '-b', label='x', linewidth=2)
    plt.plot(x, softmax(scores)[1], '-r', label='1', linewidth=2)
    plt.plot(x, softmax(scores)[2], '-g', label='0.2', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()

main_fn()
