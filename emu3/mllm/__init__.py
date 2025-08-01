# Copyright 2024 BAAI and the HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration": ["Emu3Config"],
    "tokenization": ["Emu3Tokenizer"],
    "processing_ti2ti": ["Emu3Processor"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_emu3"] = [
        "Emu3Model",
        "Emu3PretrainedModel",
        "Emu3ForCausalLM",
    ]

if TYPE_CHECKING:
    from .configuration import Emu3Config
    from .tokenization import Emu3Tokenizer
    from .processing_ti2ti import Emu3Processor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ti2ti import (
            Emu3Model,
            Emu3PreTrainedModel,
            Emu3ForCausalLM,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
