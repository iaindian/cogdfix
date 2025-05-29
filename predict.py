#!/usr/bin/env python3
import subprocess
import time
import logging
import uuid
from pathlib import Path
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath
from comfyrunbatch import (
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
        # Launch ComfyUI server
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
        image: CogPath = Input(description="Path to the input image file"),
        workflow: Optional[CogPath] = Input(
            default=None,
            description="Optional path to the workflow JSON (uses default if omitted)",
        ),
    ) -> CogPath:
        """
        Executes the image-to-image workflow and returns the generated image path.
        """
        logging.basicConfig(level=logging.INFO)
        host = "http://127.0.0.1:8188"

        # Validate and locate input image
        in_path = Path(image)
        if not in_path.is_file():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        # Determine workflow file: user-specified or default
        if workflow:
            wf_path = Path(workflow)
        else:
            wf_path = Path("workflow_api/default-workflow-api.json")
        if not wf_path.is_file():
            raise FileNotFoundError(f"Workflow file not found: {wf_path}")
        logging.info(f"Using workflow: {wf_path}")

        # Prepare output directory
        out_dir = Path("output")
        prepare_output_dir(out_dir)

        # Upload input image with unique name to avoid caching
        unique_name = f"{uuid.uuid4().hex}_{in_path.name}"
        upload_image(in_path, host, unique_name)

        # Load workflow and inject inputs
        wf = load_workflow(wf_path)
        inject_input_image(wf, unique_name)
        inject_parameters(wf)

        # Detect output node
        node = find_output_node(wf)
        logging.info(f"Using output node: {node}")

        # Queue workflow and await completion
        pid = queue_workflow(wf, host, node)
        images = await_completion(pid, host, interval=1.0, timeout=300.0)

        # Download and return the first result image
        result_path = download_output(images[0], host, out_dir)
        logging.info(f"Prediction complete: {result_path}")
        return CogPath(str(result_path))
