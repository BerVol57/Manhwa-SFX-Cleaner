import huggingface_hub
import logging

if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    logging.info("[PATCH] cached_download successfully emulated for diffusers compatibility")