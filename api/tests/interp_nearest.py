#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from common_import import *


@benchmark_registry.register("interp_nearest")
class InterpNearestConfig(APIConfig):
    def __init__(self):
        super(InterpNearestConfig, self).__init__('interp_nearest')
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(InterpNearestConfig, self).init_from_json(filename, config_id,
                                                        unknown_dim)
        if self.data_format == 'NHWC':
            self.run_torch = False


@benchmark_registry.register("interp_nearest")
class PaddleInterpNearest(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.interpolate(
            x,
            size=config.size,
            mode="nearest",
            scale_factor=config.scale_factor,
            data_format=config.data_format)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


@benchmark_registry.register("interp_nearest")
class TorchInterpNearest(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.interpolate(
            x,
            size=config.size,
            mode="nearest",
            scale_factor=config.scale_factor)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
