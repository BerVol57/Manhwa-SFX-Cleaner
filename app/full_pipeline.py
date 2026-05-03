import subprocess
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullPipeline")

class ManhwaPipeline:
    def __init__(self, venv_cleanup: str, venv_render: str):
        self.venv_cleanup = Path(venv_cleanup)
        self.venv_render = Path(venv_render)
        self.temp_dir = Path("temp_cache")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def _get_py(self, venv: Path):
        return str(venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python")

    def run_step(self, venv: Path, script: str, args: list):
        py = self._get_py(venv)
        cmd = [py, script] + args
        logger.info(f"Running: {script} in {venv.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            logger.error(f"Error in {script}:\n{result.stderr}")
            raise RuntimeError(f"Step failed: {script}")
        return result.stdout

    def process(self, image_path: str):
        img_abs = str(Path(image_path).absolute())
        
        # Stage 1: Segmentation + Inpainting
        logger.info(">>> Stage 1: Cleaning Image...")
        self.run_step(self.venv_cleanup, "app/sub_cleanup.py", [img_abs])

        # Stage 2: VLM + Rendering
        logger.info(">>> Stage 2: Translating and Rendering...")
        self.run_step(self.venv_render, "app/sub_render.py", [img_abs])
        
        logger.info(f"Done! Result at: {self.output_dir / 'final_page.png'}")

if __name__ == "__main__":
    # Приклад запуску:
    pipe = ManhwaPipeline(venv_cleanup="./venv_cleanup", venv_render="./venv_render")
    pipe.process("imgs/3.jpg")