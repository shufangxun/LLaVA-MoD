from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .language_model.llava_llama_moe import LLaVAMoDLlamaForCausalLM, LLaVAMoDLlamaConfig
from .language_model.llava_qwen import LlavaQWenForCausalLM, LlavaQWenConfig
from .language_model.llava_qwen_moe import LLaVAMoDQWenForCausalLM, LLaVAMoDQWenForCausalLMFineTune, LLaVAMoDQWenConfig
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
    from .language_model.llava_mistral_moe import LLaVAMoDMistralForCausalLM, LLaVAMoDMistralConfig
if a == '4' and int(b) >= 36:
    from .language_model.llava_minicpm import LlavaMiniCPMForCausalLM, LlavaMiniCPMConfig
    from .language_model.llava_minicpm_moe import LLaVAMoDMiniCPMForCausalLM, LLaVAMoDMiniCPMConfig
    from .language_model.llava_phi import LlavaPhiForCausalLM, LlavaPhiConfig
    from .language_model.llava_phi_moe import LLaVAMoDPhiForCausalLM, LLaVAMoDPhiConfig
    from .language_model.llava_stablelm import LlavaStablelmForCausalLM, LlavaStablelmConfig
    from .language_model.llava_stablelm_moe import LLaVAMoDStablelmForCausalLM, LLaVAMoDStablelmConfig
if a == '4' and int(b) >= 37:
    from .language_model.llava_qwen1_5 import LlavaQwen1_5ForCausalLM, LlavaQwen1_5Config
    from .language_model.llava_qwen1_5_moe import LLaVAMoDQwen1_5ForCausalLM, LLaVAMoDQwen1_5Config
    from .language_model.llava_qwen1_5_moe import LLaVAMoDQwen1_5ForCausalLMFineTune
    from .language_model.llava_qwen2 import LlavaQwen2ForCausalLM, LlavaQwen2Config
    from .language_model.llava_qwen2_moe import LLaVAMoDQwen2ForCausalLM, LLaVAMoDQwen2Config
    from .language_model.llava_qwen2_moe import LLaVAMoDQwen2ForCausalLMFineTune
    from .language_model.llava_gemma2 import LlavaGemma2ForCausalLM, LlavaGemma2Config
    from .language_model.llava_gemma2_moe import LLaVAMoDGemma2ForCausalLM, LLaVAMoDGemma2Config
    from .language_model.llava_gemma2_moe import LLaVAMoDGemma2ForCausalLMFineTune
if a == '4' and int(b) <= 31:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
