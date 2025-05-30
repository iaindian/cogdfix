#!/usr/bin/env python3
import subprocess
import time
import logging
import uuid
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath
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

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Absolute default workflow path
default_workflow_path = (Path(__file__).parent / "workflow_api/default-workflow-api.json").resolve()

class Predictor(BasePredictor):
    def setup(self):
        logger.info("Starting ComfyUI server...")
        self.proc = subprocess.Popen([
            "python", "ComfyUI/main.py",
            "--listen", "0.0.0.0", "--port", "8188",
        ])
        host = "http://127.0.0.1:8188"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                import requests
                r = requests.get(host, timeout=5)
                logger.debug(f"ComfyUI ping status: {r.status_code}")
                if r.status_code == 200:
                    logger.info("ComfyUI server is ready")
                    return
            except Exception as e:
                logger.debug(f"ComfyUI not ready yet: {e}")
                time.sleep(1)
        raise RuntimeError("ComfyUI failed to start within 60 seconds")

    def predict(
        self,
        input: CogPath = Input(description="Path to the input image file"),
    ) -> CogPath:
        try:
            logger.info(f"Predict called with input={input}")
            host = "http://127.0.0.1:8188"

            # Validate input image
            in_path = Path(input)
            logger.info(f"Validating input image at: {in_path}")
            if not in_path.is_file():
                logger.error(f"Input file not found: {in_path}")
                raise FileNotFoundError(f"Input file not found: {in_path}")

            # Use default workflow
            wf_path = default_workflow_path
            logger.info(f"Using default workflow file: {wf_path}")
            logger.debug(f"Exists? {wf_path.exists()}, CWD={Path().resolve()}")
            if not wf_path.is_file():
                logger.error(f"Default workflow file not found: {wf_path}")
                raise FileNotFoundError(f"Default workflow file not found: {wf_path}")

            # Prepare output directory
            out_dir = Path("ComfyUI/output")
            logger.info(f"Preparing output directory: {out_dir}")
            prepare_output_dir(out_dir)

            # Upload input image
            unique_name = f"{uuid.uuid4().hex}_{in_path.name}"  # unique upload key
            logger.info(f"Uploading image as: {unique_name}")
            upload_image(in_path, host, unique_name)

            # Load workflow
            logger.info("Loading workflow JSON")
            wf = load_workflow(wf_path)

            # Inject input image
            logger.info(f"Injecting image {unique_name} into workflow nodes")
            inject_input_image(wf, unique_name)

            # Inject parameters (defaults)
            logger.info("Injecting default parameters into workflow")
            inject_parameters(wf)

            # Detect output node
            logger.info("Detecting output node in workflow")
            node = find_output_node(wf)
            logger.info(f"Output node selected: {node}")

            # Queue workflow
            logger.info("Queueing workflow for execution")
            pid = queue_workflow(wf, host, node)
            logger.info(f"Workflow queued with prompt_id: {pid}")

            # Await completion
            logger.info("Awaiting workflow completion")
            images = await_completion(pid, host, interval=1.0, timeout=300.0)
            logger.info(f"Received {len(images)} image(s) from server")

            # Download first output image
            first = images[0]
            filename = first.get('filename')
            logger.info(f"Downloading first output image: {filename}")
            download_output(first, host, out_dir)

            # Construct and return path
            result_path = out_dir / filename
            logger.info(f"Downloaded output image to: {result_path}")
            return CogPath(str(result_path))

        except Exception as e:
            logger.exception("Prediction failed with exception")
            raise

if __name__ == '__main__':
    # Local test
    pred = Predictor()
    pred.setup()
    output = pred.predict(input="test.jpg")
    print("Output:", output)
