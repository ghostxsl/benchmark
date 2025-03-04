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


class TruncatedGaussianRandomConfig(APIConfig):
    def __init__(self):
        super(TruncatedGaussianRandomConfig,
              self).__init__("truncated_gaussian_random")
        self.run_torch = False
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


class PaddleTruncatedGaussianRandom(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        paddle.seed(config.seed)
        normal = paddle.nn.initializer.TruncatedNormal(
            mean=config.mean, std=config.std)
        w = self.variable(name="w", shape=config.w_shape, dtype=config.w_dtype)
        result = normal(w)
        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleTruncatedGaussianRandom(),
        config=TruncatedGaussianRandomConfig())
