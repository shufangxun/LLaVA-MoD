from .model import LlavaLlamaForCausalLM
from .model import LLaVAMoDLlamaForCausalLM
from .model import LlavaQWenForCausalLM
from .model import LLaVAMoDLlamaForCausalLM
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .model import LlavaMistralForCausalLM
    from .model import LLaVAMoDMistralForCausalLM
if a == '4' and int(b) >= 36:
    from .model import LlavaMiniCPMForCausalLM
    from .model import LLaVAMoDMiniCPMForCausalLM
    from .model import LlavaPhiForCausalLM
    from .model import LLaVAMoDPhiForCausalLM
    from .model import LlavaStablelmForCausalLM
    from .model import LLaVAMoDStablelmForCausalLM
if a == '4' and int(b) >= 37:
    from .model import LlavaQwen1_5ForCausalLM
    from .model import LLaVAMoDQwen1_5ForCausalLM
