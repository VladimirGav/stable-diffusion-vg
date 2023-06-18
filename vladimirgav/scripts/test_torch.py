#!/usr/bin/env python # [1]
"""
 * VladimirGav
 * GitHub Website: https://vladimirgav.github.io/
 * GitHub: https://github.com/VladimirGav
 * Source https://github.com/VladimirGav/stable-diffusion-vg
 * Copyright (c)
"""

import torch
x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())
print(torch.version.cuda)