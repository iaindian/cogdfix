from cog import BasePredictor, Input, Path as CogPath
import subprocess
import time
import logging
import uuid
from pathlib import Path
from comfyrun import (
    prepare_output_dir,
    upload_image,
    load_workflow,
    inject_input_image,
    inject_parameters,
    find_output_node,
    queue_workflow,
    await_completion,
    download_output,
)


class Predictor(BasePredictor):
    def setup(self):
        # Launch ComfyUI in server mode
        self.proc = subprocess.Popen([
            "python", "ComfyUI/main.py",
            "--listen", "0.0.0.0", "--port", "8188",
        ])
        host = "http://127.0.0.1:8188"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                import requests
                if requests.get(host, timeout=5).status_code == 200:
                    logging.info("ComfyUI server ready")
                    return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI failed to start within timeout")

    def predict(
        self,
        workflow: CogPath = Input(description="Path to workflow JSON"),
        image: CogPath = Input(description="Path to input image file"),
    ) -> CogPath:
        """
        Executes the image-to-image workflow and returns the generated image path.
        """
        logging.basicConfig(level=logging.INFO)
        host = "http://127.0.0.1:8188"

        # Prepare directories and upload
        in_path = Path(image)
        out_dir = Path("output")
        prepare_output_dir(out_dir)
        unique_name = f"{uuid.uuid4().hex}_{in_path.name}"
        upload_image(in_path, host, unique_name)

        # Load and inject workflow
        wf = load_workflow(Path(workflow))
        inject_input_image(wf, unique_name)
        inject_parameters(wf)

        # Determine output node
        node = find_output_node(wf)
        logging.info(f"Using output node {node}")

        # Queue and process
        pid = queue_workflow(wf, host, node)
        images = await_completion(pid, host, interval=1.0, timeout=300.0)

        # Download and return first result
        result = download_output(images[0], host, out_dir)
        return CogPath(str(result))
