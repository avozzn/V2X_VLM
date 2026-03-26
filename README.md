# UniV2X + VLM: Sequential Perception and Decision-making with Multimodal LLMs

This project extends the [UniV2X](https://www.google.com/search?q=https://github.com/OpenDriveLab/UniV2X) framework by integrating Vision-Language Models (VLM) for enhanced V2X (Vehicle-to-Everything) perception and reasoning. By leveraging the **LLaVA-Interleave** framework, we transform traditional V2X sensor data into a multimodal textual-visual format, enabling the model to process sequential V2X data with powerful language reasoning capabilities.

## 🌟 Key Features

  - **VLM Integration**: Incorporates [LLaVA-Interleave](https://github.com/LLaVA-VL/LLaVA-NeXT) to handle multi-frame and multi-view V2X sensor inputs.
  - **Dataset Support**: Specifically designed for the **V2X-Seq-SPD** dataset.
  - **Data Conversion Pipeline**: A specialized tool to convert raw V2X-Seq-SPD data into a VLM-friendly text+image instruction-following format.

## 📊 Dataset

This project uses the **V2X-Seq-SPD** dataset.

  - **Download**: [Google Drive Link](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e)
  - **Description**: V2X-Seq-SPD is a comprehensive sequential perception dataset for vehicle-infrastructure cooperative autonomous driving.

## 🛠️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd UniV2X

# Follow UniV2X and LLaVA-Interleave installation guides
pip install -r requirements.txt
# Additional dependencies for LLaVA-Interleave
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
```

## 🔄 Data Preparation

We provide a specialized script to convert the V2X-Seq-SPD dataset into the format required for VLM training/inference.

### SPD to VLM Conversion

The conversion script is located at `UniV2X/tools/spd_data_converter/spd_to_vlm.py`. It generates textual descriptions (questions/answers) corresponding to the sequential point cloud or image data.

```bash
python UniV2X/tools/spd_data_converter/spd_to_vlm.py \
    --data-path /path/to/V2X-Seq-SPD \
    --output-path /path/to/vlm_text_dataset \
    --config configs/conversion_config.yaml
```

**What this script does:**

1.  Parses V2X-Seq-SPD metadata and sequential frames.
2.  Extracts key perception targets (vehicles, infrastructure, traffic signs).
3.  Generates interleaved text-image pairs (e.g., "Given the view from the roadside and vehicle units, what is the current traffic situation?").
4.  Formats the output to be compatible with LLaVA-Interleave training requirements.

## 🚀 Model Architecture

Instead of traditional bounding box outputs, our modified UniV2X-VLM framework:

1.  **Inputs**: Multi-view images from both Vehicle (V) and Infrastructure (I) sides.
2.  **Backbone**: Uses LLaVA-Interleave to fuse multi-modal temporal features.
3.  **Output**: Natural language descriptions of the scene, spatial coordinates, and driving decisions.

## 📝 TODO

  - [ ] Support for more VLM backbones.
  - [ ] Pre-trained weights release.
  - [ ] Evaluation scripts for SPD-VLM tasks.

## Acknowledgements

  - [UniV2X](https://www.google.com/search?q=https://github.com/OpenDriveLab/UniV2X)
  - [LLaVA](https://github.com/haotian-liu/LLaVA)
  - [V2X-Seq Dataset](https://www.google.com/search?q=https://v2x-seq-by-air.github.io/)
