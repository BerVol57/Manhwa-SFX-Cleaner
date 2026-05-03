import gc
import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

class VLMInterpreter:
    """
    Wraps Qwen2.5-VL + ChromaDB RAG pipeline.
    Loads models lazily on first call to save VRAM until needed.
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    DICT_PATH = "manhwa_onomatopoeia.json"
    CHROMA_PATH = "./chroma_db"
    EMBED_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._embed_model = None
        self._collection = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _load_vlm(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        logger.info("Loading Qwen2.5-VL (4-bit, bfloat16) as in notebook...")
        
        # 1. Точнісінько як у vlm_models.ipynb
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # 2. Обмеження пікселів (як у зошиті, щоб не зависала)
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28 
        
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels
        )
        
        # 3. Завантаження моделі з квантизацією та bfloat16
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logger.info("VLM loaded successfully.")

    def _load_rag(self) -> None:
        if self._collection is not None:
            return
        import json
        import chromadb
        from sentence_transformers import SentenceTransformer

        logger.info("Loading RAG components …")
        self._embed_model = SentenceTransformer(self.EMBED_MODEL_ID)

        client = chromadb.PersistentClient(path=self.CHROMA_PATH)
        self._collection = client.get_or_create_collection(name="onomatopoeia")

        # Index dictionary if collection is empty
        if self._collection.count() == 0 and Path(self.DICT_PATH).exists():
            with open(self.DICT_PATH, "r", encoding="utf-8") as f:
                dictionary = json.load(f)
            for cat in dictionary["categories"]:
                for entry in cat["entries"]:
                    text = f"{entry['meaning']} {cat['label']} {entry['transliteration']}"
                    emb = self._embed_model.encode(text).tolist()
                    self._collection.upsert(
                        ids=[entry["korean"]],
                        embeddings=[emb],
                        metadatas=[{
                            "ukrainian": ", ".join(entry["ukrainian"]),
                            "meaning": entry["meaning"],
                            "category": cat["label"],
                        }],
                        documents=[text],
                    )
            logger.info("RAG indexed %d entries.", self._collection.count())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ask_vlm(self, image_path: str, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        # 2. Генерація відповіді (як у ask_model_simple)
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                temperature=0
            )
            
        # 3. Декодування результату
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _rag_candidates(self, description: str, n: int = 3) -> str:
        vec = self._embed_model.encode(description).tolist()
        res = self._collection.query(query_embeddings=[vec], n_results=n)
        lines = ["Reference candidates (Korean → Ukrainian):"]
        for i in range(len(res["ids"][0])):
            m = res["metadatas"][0][i]
            lines.append(f"- {res['ids'][0][i]} ({m['meaning']}): {m['ukrainian']}")
        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self, image_path: str) -> str:
        """
        Given a path to a manhwa panel (or cropped SFX region), return the
        best Ukrainian onomatopoeia translation.
        """
        self._load_vlm()
        self._load_rag()

        # Step 1 – describe the visual scene in English for the embed model
        desc_prompt = (
            "Describe ONLY the visual action and sound style in this manga panel. "
            "Focus on: type of motion (sharp/flowing), intensity (loud/quiet), "
            "and what physical action is happening. Answer in English, 1-2 sentences."
        )
        visual_desc = self._ask_vlm(image_path, desc_prompt)
        logger.info("[VLM] Scene description: %s", visual_desc)

        # Step 2 – vector search for candidate translations
        candidates = self._rag_candidates(visual_desc)
        logger.info("[VLM] Candidates:\n%s", candidates)

        # Step 3 – final selection
        final_prompt = (
            f"The action in this image was described as: '{visual_desc}'\n\n"
            f"{candidates}\n"
            "Choose the single most appropriate Ukrainian sound effect from the "
            "candidates above. Output ONLY the Ukrainian word, nothing else."
        )
        result = self._ask_vlm(image_path, final_prompt)
        logger.info("[VLM] Translation: %s", result)
        return result

    def unload(self) -> None:
        """Free VRAM after translation is done."""
        del self._model, self._processor, self._embed_model
        self._model = self._processor = self._embed_model = None
        self._collection = None  # chromadb не займає VRAM, але скидаємо стан
        gc.collect()
        torch.cuda.empty_cache()